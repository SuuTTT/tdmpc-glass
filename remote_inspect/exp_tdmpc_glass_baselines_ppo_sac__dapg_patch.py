"""DAPG-style demo-augmented PPO (alternative (a) in the plan).

Monkeypatches brax ppo.losses.compute_ppo_loss to ADD a behavior-cloning
auxiliary term on offline demos to the PPO total loss:

    total_loss += BC_COEF * mean( (tanh(policy_mean(demo_obs)) - demo_act)^2 )

This biases PPO's policy toward the abstraction's competent behavior WITHOUT
relying on standalone BC being a good policy (PPO's online rollouts correct the
compounding error; the demo term only provides directed gradient). Unconstrained
raw-action policy -> can exceed the heuristic's structural cap.

Env-gated (only active when DAPG_DEMOS is set), so importing this is a no-op for
plain PPO. BC_COEF is constant (simplest working DAPG variant); brax's jitted
loss has no easy access to the global step for annealing, so we use a small fixed
coefficient and rely on PPO's own loss growing as it learns to dominate later.

A fixed demo minibatch (DAPG_BC_BATCH rows) is sampled once at import and held
as a module constant captured into the jitted loss (static across updates).
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from brax.training.agents.ppo import losses as _L

_DEMOS = os.environ.get("DAPG_DEMOS", "")
_BC_COEF = float(os.environ.get("DAPG_BC_COEF", "0.1"))
_BC_BATCH = int(os.environ.get("DAPG_BC_BATCH", "8192"))

if _DEMOS:
    _d = np.load(_DEMOS)
    _obs = _d["obs"].astype(np.float32)
    _act = np.clip(_d["act"].astype(np.float32), -0.999, 0.999)
    # fixed subsample for a stable, jit-friendly demo minibatch
    _rng = np.random.default_rng(0)
    _idx = _rng.choice(_obs.shape[0], size=min(_BC_BATCH, _obs.shape[0]), replace=False)
    _DEMO_OBS = jnp.asarray(_obs[_idx])
    _DEMO_ACT = jnp.asarray(_act[_idx])
    _AD = _act.shape[1]
    print(f"[DAPG] loaded {_DEMO_OBS.shape[0]} demo rows, BC_COEF={_BC_COEF}", flush=True)

    _orig_compute = _L.compute_ppo_loss

    def compute_ppo_loss_dapg(params, normalizer_params, data, rng, ppo_network, **kw):
        total_loss, metrics = _orig_compute(
            params, normalizer_params, data, rng, ppo_network, **kw)
        # BC auxiliary term on demos (raw-action space, matches det inference)
        policy_apply = ppo_network.policy_network.apply
        logits = policy_apply(normalizer_params, params.policy, _DEMO_OBS)
        mean = logits[:, :_AD]
        pred = jnp.tanh(mean)
        bc_loss = jnp.mean((pred - _DEMO_ACT) ** 2)
        total_loss = total_loss + _BC_COEF * bc_loss
        metrics = dict(metrics)
        metrics["bc_loss"] = bc_loss
        metrics["bc_coef"] = jnp.asarray(_BC_COEF)
        return total_loss, metrics

    _L.compute_ppo_loss = compute_ppo_loss_dapg
    # ppo.train imports the symbol at call time from the module, so patching the
    # module attribute is sufficient. Also patch any already-bound reference.
    try:
        from brax.training.agents.ppo import train as _T
        if hasattr(_T, "compute_ppo_loss"):
            _T.compute_ppo_loss = compute_ppo_loss_dapg
    except Exception:
        pass
    print("[DAPG] compute_ppo_loss patched.", flush=True)
else:
    print("[DAPG] DAPG_DEMOS unset -> no-op (plain PPO).", flush=True)
