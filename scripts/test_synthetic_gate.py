#!/usr/bin/env python3
"""Tiny CPU-runnable selftest for the synthetic mechanism-check infra.

Runs WITHOUT a GPU; needs only ``jax`` + ``flax`` + ``optax`` on CPU. It does
NOT touch the queue, daemons, or any training. Run on the control box only if
those deps happen to be installed; otherwise run it on a worker.

Checks:
  1. env builds at N=4; obs / per-entity shapes correct; reset/step jit.
  2. ground-truth sparsity: perturbing a VALUE-IRRELEVANT entity leaves the
     reward UNCHANGED; perturbing a VALUE-RELEVANT entity CHANGES the reward.
  3. N-monotonicity of coupling: the N=4 sub-block of C is identical inside the
     N=6 coupling (same rules, more objects).
  4. EntityWM builds; forward returns the probe contract (attn/tokens/q/...);
     OOD apply at N=6 with N=4-trained params works (N-agnostic).
  5. two train steps on mock data run with finite (non-NaN) loss.

Usage:
    PYTHONPATH=<repo>/src python3 <repo>/scripts/test_synthetic_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from helios.dynamics.entity_wm import EntityWM, entity_wm_loss
from helios.envs import synthetic_entities as se


def check_env_shapes():
    env = se.make_env(n_entities=4, seed=0)
    state = env.reset(jax.random.PRNGKey(0))
    obs = env.observe(state)
    ent = env.observe_entities(state)
    assert obs.shape == (4 * se.ENTITY_DIM,), obs.shape
    assert ent.shape == (4, se.ENTITY_DIM), ent.shape

    # jit reset/step
    jreset = jax.jit(lambda k: env.reset(k))
    jstep = jax.jit(lambda s, a: env.step(s, a))
    s = jreset(jax.random.PRNGKey(1))
    s2, o2, r, d, info = jstep(s, jnp.array([0.5, -0.5]))
    assert o2.shape == (4 * se.ENTITY_DIM,)
    assert jnp.isfinite(r)
    assert "agent_goal_dist" in info and "pair_dist" in info
    print("[ok] env shapes + jit reset/step")
    return env


def check_reward_sparsity(_env):
    """Perturb irrelevant vs relevant entity; reward should only change for the
    relevant one. Uses N=6 so there is at least one distractor (at N=4 every
    entity is value-relevant: {agent,goal,p,q})."""
    env = se.make_env(n_entities=6, seed=0)
    state = env.reset(jax.random.PRNGKey(3))
    base_r = se._reward(env, state.pos)

    relevant = set(env.value_relevant_entities)
    all_idx = set(range(env.n_entities))
    irrelevant = sorted(all_idx - relevant)
    assert irrelevant, (
        "test needs at least one value-irrelevant entity; "
        f"relevant={relevant} N={env.n_entities}"
    )

    # Perturb an irrelevant entity's position -> reward unchanged.
    irr = irrelevant[0]
    pos_irr = state.pos.at[irr].add(jnp.array([5.0, -3.0]))
    r_irr = se._reward(env, pos_irr)
    assert jnp.allclose(r_irr, base_r), (
        f"reward changed when perturbing IRRELEVANT entity {irr}: "
        f"{float(base_r)} -> {float(r_irr)}"
    )

    # Perturb a relevant entity (the value pair's p) -> reward changes.
    p = env.value_pair[0]
    pos_rel = state.pos.at[p].add(jnp.array([5.0, -3.0]))
    r_rel = se._reward(env, pos_rel)
    assert not jnp.allclose(r_rel, base_r), (
        f"reward did NOT change when perturbing RELEVANT entity {p}"
    )
    print(
        f"[ok] reward sparsity: relevant={sorted(relevant)} "
        f"irrelevant={irrelevant} (perturb irr -> unchanged, rel -> changed)"
    )


def check_n_monotonic():
    env4 = se.make_env(n_entities=4, seed=0)
    env6 = se.make_env(n_entities=6, seed=0)
    C4 = np.asarray(env4.coupling)
    C6 = np.asarray(env6.coupling)
    # NOTE: the value pair is force-set in BOTH, and for n>=4 it is (2,3) in
    # both, so the forced edge lands in the shared 4x4 block consistently.
    assert np.allclose(C4, C6[:4, :4]), (
        "N=4 coupling sub-block differs inside N=6 coupling:\n"
        f"{C4}\n vs \n{C6[:4, :4]}"
    )
    # value pair stable across N
    assert env4.value_pair == env6.value_pair, (env4.value_pair, env6.value_pair)
    print(
        f"[ok] N-monotonic coupling (N=4 block preserved in N=6); "
        f"value_pair stable = {env4.value_pair}"
    )


def check_model_and_train():
    env = se.make_env(n_entities=4, seed=0)
    model = EntityWM(
        entity_dim=se.ENTITY_DIM,
        action_dim=2,
        n_entities=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        max_entities=16,
    )
    B, N = 8, 4
    rng = jax.random.PRNGKey(7)
    ent = jax.random.normal(rng, (B, N, se.ENTITY_DIM))
    act = jax.random.normal(rng, (B, 2))
    params = model.init(rng, ent[:2], act[:2])["params"]

    out = model.apply({"params": params}, ent, act, return_attn=True)
    assert out["next_ent"].shape == (B, N, se.ENTITY_DIM), out["next_ent"].shape
    assert out["reward"].shape == (B,), out["reward"].shape
    assert out["q"].shape == (B,), out["q"].shape
    assert out["attn"].shape == (2, B, 4, N, N), out["attn"].shape
    assert out["tokens"].shape == (3, B, N, 32), out["tokens"].shape
    # attention rows sum to 1 (softmax over keys)
    assert jnp.allclose(jnp.sum(out["attn"], axis=-1), 1.0, atol=1e-4)
    print("[ok] EntityWM forward + probe contract (attn/tokens/q/reward/next_ent)")

    # OOD apply: same params at N=6 (N-agnostic).
    ent6 = jax.random.normal(rng, (B, 6, se.ENTITY_DIM))
    out6 = model.apply({"params": params}, ent6, act, return_attn=True)
    assert out6["q"].shape == (B,)
    assert out6["attn"].shape == (2, B, 4, 6, 6), out6["attn"].shape
    print("[ok] EntityWM OOD apply at N=6 with N=4 params")

    # Q-grad smoke (the probe will differentiate Q wrt entity inputs).
    def q_scalar(e):
        return jnp.sum(model.apply({"params": params}, e, act)["q"])

    g = jax.grad(q_scalar)(ent)
    assert g.shape == ent.shape and jnp.all(jnp.isfinite(g))
    print("[ok] dQ/dent gradient finite (probe affordance)")

    # Two train steps on mock data; assert finite loss.
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)
    batch = {
        "ent": ent,
        "action": act,
        "next_ent": ent + 0.01,
        "reward": jnp.zeros((B,)),
        "q_target": jnp.zeros((B,)),
    }

    @jax.jit
    def step(params, opt_state):
        (loss, _), grads = jax.value_and_grad(entity_wm_loss, has_aux=True)(
            params, model.apply, batch
        )
        upd, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss

    for i in range(2):
        params, opt_state, loss = step(params, opt_state)
        assert jnp.isfinite(loss), f"non-finite loss at step {i}: {loss}"
    print(f"[ok] 2 train steps; final mock loss = {float(loss):.4f}")


def main():
    print("=== synthetic_gate selftest ===")
    env = check_env_shapes()
    check_reward_sparsity(env)
    check_n_monotonic()
    check_model_and_train()
    print("=== ALL SELFTESTS PASSED ===")


if __name__ == "__main__":
    main()
