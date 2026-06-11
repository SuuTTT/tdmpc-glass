#!/usr/bin/env python3
"""Standalone DreamerV4 (small transformer-WM) launcher for MuJoCo Playground.

A SELF-CONTAINED driver for the transformer-world-model agent in
``src/helios/algorithms/dreamer4.py`` + ``src/helios/dynamics/transformer_wm.py``.
It mirrors scripts/run_dreamer.py exactly (same env build, same CSV layout) but
uses a block-causal Transformer dynamics core instead of the RSSM, and tags eval
rows with ``eval_type="dreamer4"``.

It does NOT import or modify the hot path (run_benchmark.py / tdmpc2.py /
tdmpc_glass.py / task_queue_daemon.py).

CSV outputs (mirrors run_benchmark / run_dreamer conventions):

    exp/tdmpc_glass/<env_id>[_<TAG>]/seed_<seed>.csv
        header: step,reward,eval_type,seed
        rows:   <env_steps>,<reward>,dreamer4,<seed>
    exp/benchmark/dreamer4_<task>[_<TAG>].csv   (rollup: task,seed,step,reward)

Launch convention (matches run_dreamer_baseline.sh env vars):

    PROBE_ID=<id> ALGO=dreamer4 TASK=PandaPickCube SEEDS="1 2 3" \
    TOTAL_STEPS=3000000 CODE_SHA=$(git rev-parse --short HEAD) \
    TDMPC_GLASS_OUTPUT_TAG=<tag> python3 scripts/run_dreamer4.py

NOTE: this box (EC2) has no GPU; the author validates with py_compile only and
training runs on remote workers via the queue. The transformer WM is sized for a
single 16 GB GPU (RTX 5070 Ti): d_model=256, n_layers=4, n_heads=4, ctx=32.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO.parents[1] / "wiki/learn_mujoco_playground/repo"))

import jax
import jax.numpy as jnp

from mujoco_playground import registry, wrapper

from helios.algorithms.dreamer4 import DreamerV4Agent
from helios.core.distributions import TanhNormal
from helios.memory.trajectory import TrajectoryBuffer


EXP_DIR = _REPO / "exp" / "benchmark"


# ---------------------------------------------------------------------------
# DreamerV4 config (transformer WM, state-based MuJoCo Playground)
# ---------------------------------------------------------------------------

def make_config(
    *,
    gamma: float = 0.997,
    horizon: int = 15,
    batch_size: int = 16,
    batch_length: int = 64,
) -> SimpleNamespace:
    """Config consumed by DreamerV4Agent. Transformer sized for a single 16 GB GPU."""
    transformer = SimpleNamespace(
        embed_dim=256,
        d_model=256,
        n_layers=4,
        n_heads=4,
        context_len=32,
        mlp_ratio=4,
        pos_encoding="learned",
    )
    return SimpleNamespace(
        transformer=transformer,
        # Head / actor / critic widths
        actor_hidden_dims=(512, 512),
        critic_hidden_dims=(512, 512),
        # World-model loss weights
        reconstruction_loss_weight=1.0,
        embed_loss_weight=1.0,
        reward_loss_weight=1.0,
        continue_loss_weight=1.0,
        # Actor-critic in imagination
        imagination_horizon=horizon,
        gamma=gamma,
        gae_lambda=0.95,
        actor_entropy_scale=3e-4,
        slow_target_fraction=0.02,
        # Optimizers
        model_lr=1e-4,
        model_eps=1e-8,
        actor_lr=3e-5,
        actor_eps=1e-5,
        critic_lr=3e-5,
        critic_eps=1e-5,
        max_grad_norm=100.0,
        # Replay / training cadence (read by this launcher)
        batch_size=batch_size,
        batch_length=batch_length,
    )


# ---------------------------------------------------------------------------
# Minimal gymnasium-like spaces (agent reads .shape / .n)
# ---------------------------------------------------------------------------

class _BoxSpace:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.low = -np.ones(shape, dtype=np.float32)
        self.high = np.ones(shape, dtype=np.float32)


def _patch_gym_box(action_dim: int, obs_dim: int):
    try:
        import gymnasium as gym

        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        return obs_space, act_space
    except Exception:
        import types as _types

        fake_gym = _types.ModuleType("gymnasium")
        fake_spaces = _types.ModuleType("gymnasium.spaces")
        fake_spaces.Box = _BoxSpace
        fake_gym.spaces = fake_spaces
        sys.modules.setdefault("gymnasium", fake_gym)
        sys.modules.setdefault("gymnasium.spaces", fake_spaces)
        return _BoxSpace((obs_dim,)), _BoxSpace((action_dim,))


# ---------------------------------------------------------------------------
# CSV helpers (mirror run_dreamer.py)
# ---------------------------------------------------------------------------

def eval_csv_path(env_id: str, seed: int) -> Path:
    tag = os.environ.get("TDMPC_GLASS_OUTPUT_TAG", "").strip()
    env_dir = f"{env_id}{('_' + tag) if tag else ''}"
    p = EXP_DIR.parent / "tdmpc_glass" / env_dir / f"seed_{seed}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, "w") as f:
            f.write("step,reward,eval_type,seed\n")
    return p


def rollup_csv_path(task: str) -> Path:
    tag = os.environ.get("TDMPC_GLASS_OUTPUT_TAG", "").strip()
    suffix = f"_{tag}" if tag else ""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    p = EXP_DIR / f"dreamer4_{task}{suffix}.csv"
    if not p.exists():
        with open(p, "w") as f:
            f.write("task,seed,step,reward\n")
    return p


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_dreamer4(
    env_id: str,
    total_steps: int,
    seed: int,
    *,
    num_envs: int = 16,
    warmup_env_steps: int = 25_000,
    train_every: int = 16,
    updates_per_step: int = 1,
) -> None:
    cfg = make_config()
    episode_length = 1000
    eval_interval = 250_000 if env_id == "HopperHop" else 50_000
    _ei_env = os.environ.get("DREAMER_EVAL_INTERVAL", "").strip()
    if _ei_env:
        eval_interval = int(_ei_env)
    ctx = int(cfg.transformer.context_len)

    # ── Environment (identical construction to run_dreamer / run_benchmark) ──
    env = registry.load(env_id)
    env = wrapper.wrap_for_brax_training(env, episode_length=episode_length, action_repeat=1)
    obs_dim = int(env.observation_size)
    act_dim = int(env.action_size)
    print(f"  [dreamer4] env={env_id} obs={obs_dim} act={act_dim} ctx={ctx}", flush=True)

    obs_space, act_space = _patch_gym_box(act_dim, obs_dim)

    # ── Agent ──
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    agent = DreamerV4Agent(cfg, obs_space, act_space)
    state = agent.initial_state(init_key)

    # ── Replay buffer (sequence buffer for the transformer) ──
    buffer = TrajectoryBuffer(
        capacity=max(total_steps + 10_000, 100_000),
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
        seq_len=int(cfg.batch_length),
    )
    rng_np = np.random.default_rng(seed)

    @jax.jit
    def batch_reset(k):
        return env.reset(jax.random.split(k, num_envs))

    @jax.jit
    def batch_step(st, acts):
        return env.step(st, acts)

    @jax.jit
    def single_reset(k):
        return env.reset(jax.random.split(k, 1))

    @jax.jit
    def single_step(st, act):
        return env.step(st, act[None])

    enc = state["encoder"]
    wm = state["wm"]
    actor = state["actor"]

    # ── Policy step over a rolling (embed, action) window ──
    # emb_hist/act_hist: (B, ctx, *) padded windows; we keep the full window and
    # the transformer's causal mask + last-position read gives the current state.
    @jax.jit
    def policy_step(wm_params, actor_params, emb_hist, act_hist, k, deterministic):
        out = wm.apply(wm_params["wm"], emb_hist, act_hist)
        feat = out["z"][:, -1]                            # (B, d_model)
        a_out = actor.apply(actor_params, feat)
        dist = TanhNormal(a_out["mean"], a_out["log_std"])
        det_a = dist.mode()
        smp_a, _ = dist.sample(k)
        action = jnp.where(deterministic, det_a, smp_a)
        return action

    @jax.jit
    def encode(wm_params, obs):
        return enc.apply(wm_params["encoder"], obs)       # (B, e)

    embed_dim = int(cfg.transformer.embed_dim)

    def roll_window(hist, new_step):
        """Drop oldest, append newest along the time axis. hist:(B,ctx,*)."""
        return np.concatenate([hist[:, 1:, :], new_step[:, None, :]], axis=1)

    def roll_window_j(hist, new_step):
        """jnp version of roll_window for use inside the scanned collector."""
        return jnp.concatenate([hist[:, 1:, :], new_step[:, None, :]], axis=1)

    # ── Vectorised on-device collection: lax.scan K env steps per call ──
    # Carry = (env_state, emb_hist, act_hist, key). Per step:
    #   1. encode current obs -> append to emb_hist rolling window
    #   2. policy_step on the full (emb_hist, act_hist) window (transformer fwd)
    #   3. (warmup) replace action with uniform-random actions
    #   4. append action to act_hist; env.step; emit (obs,action,reward,done)
    #   5. zero both windows per-env on done via jnp.where on the mask
    # ONE host transfer + ONE batched buffer push happens AFTER the scan.
    def _make_collect(chunk_len: int):
        @jax.jit
        def collect_chunk(env_state, emb_hist, act_hist, key, use_random):
            def body(carry, _):
                es, emb_hist, act_hist, k = carry
                obs_t = es.obs
                emb = enc.apply(state["wm_params"]["encoder"], obs_t)
                emb_hist = roll_window_j(emb_hist, emb)
                k, sk, rk = jax.random.split(k, 3)
                pol_a = policy_step(
                    state["wm_params"], state["actor_params"],
                    emb_hist, act_hist, sk, jnp.bool_(False),
                )
                rand_a = jax.random.uniform(rk, pol_a.shape, minval=-1.0, maxval=1.0)
                action = jnp.where(use_random, rand_a, pol_a)
                act_hist = roll_window_j(act_hist, action)
                nes = env.step(es, action)
                done = (nes.done > 0.5)
                # Zero rolling windows per-env on episode end (auto-reset obs).
                keep = (~done)[:, None, None]
                emb_hist = jnp.where(keep, emb_hist, 0.0)
                act_hist = jnp.where(keep, act_hist, 0.0)
                emit = (obs_t, action, nes.reward, done.astype(jnp.float32))
                return (nes, emb_hist, act_hist, k), emit

            (nes, emb_hist, act_hist, nk), traj = jax.lax.scan(
                body, (env_state, emb_hist, act_hist, key), None, length=chunk_len
            )
            return nes, emb_hist, act_hist, nk, traj

        return collect_chunk

    # ── Evaluation: deterministic actor rollout, single env, real env reward ──
    def eval_actor(n_eps: int = 5) -> float:
        nonlocal key
        rets = []
        for _ in range(n_eps):
            key, rk = jax.random.split(key)
            st = single_reset(rk)
            obs = jnp.asarray(st.obs)                     # (1, obs_dim)
            emb_hist = np.zeros((1, ctx, embed_dim), np.float32)
            act_hist = np.zeros((1, ctx, act_dim), np.float32)
            er = 0.0
            for _ in range(episode_length):
                emb = np.array(encode(state["wm_params"], obs))
                emb_hist = roll_window(emb_hist, emb)
                key, sk = jax.random.split(key)
                action = policy_step(
                    state["wm_params"], state["actor_params"],
                    jnp.asarray(emb_hist), jnp.asarray(act_hist), sk, True,
                )
                act_np = np.array(action, np.float32)
                act_hist = roll_window(act_hist, act_np)
                st = single_step(st, action[0])
                er += float(st.reward[0])
                if bool(st.done[0] > 0.5):
                    break
                obs = jnp.asarray(st.obs)
            rets.append(er)
        return float(np.mean(rets))

    # ── CSV outputs ──
    ev_csv = eval_csv_path(env_id, seed)
    roll_csv = rollup_csv_path(env_id)

    def write_eval(step: int, reward: float) -> None:
        with open(ev_csv, "a") as f:
            f.write(f"{step},{reward:.1f},dreamer4,{seed}\n")
        with open(roll_csv, "a") as f:
            f.write(f"{env_id},{seed},{step},{reward:.4f}\n")

    # ── Collection + training loop (chunked, on-device scan) ──
    # The transformer forward runs per scan-step on the full rolling window, but
    # the scan keeps the whole chunk on-device so there is ONE host sync + ONE
    # batched buffer push per chunk instead of a per-step device→host sync. We
    # cut each chunk at the next {warmup, eval, total} boundary to keep the
    # warmup/policy branch and the eval cadence byte-identical to the per-step
    # version. Chunk default is smaller than DreamerV3 because each scan-step is
    # a full transformer forward (memory/compile cost grows with chunk_len).
    base_chunk = max(train_every // num_envs, 16)
    _chunk_cache: dict[int, "callable"] = {}

    def get_collector(n):
        c = _chunk_cache.get(n)
        if c is None:
            c = _make_collect(n)
            _chunk_cache[n] = c
        return c

    key, rk = jax.random.split(key)
    env_state = batch_reset(rk)
    emb_hist = jnp.zeros((num_envs, ctx, embed_dim), np.float32)
    act_hist = jnp.zeros((num_envs, ctx, act_dim), np.float32)

    env_steps = 0
    next_eval = eval_interval
    t0 = time.time()
    last_metrics: dict[str, float] = {}

    print(f"  [dreamer4] training to {total_steps:,} env-steps "
          f"(num_envs={num_envs}, warmup={warmup_env_steps:,}, chunk={base_chunk} scan-steps)",
          flush=True)

    while env_steps < total_steps:
        boundaries = [total_steps]
        if env_steps < warmup_env_steps:
            boundaries.append(warmup_env_steps)
        boundaries.append(next_eval)
        next_boundary = min(b for b in boundaries if b > env_steps)
        steps_to_boundary = next_boundary - env_steps
        scan_steps = min(base_chunk, max(steps_to_boundary // num_envs, 1))
        chunk_env_steps = scan_steps * num_envs

        use_random = env_steps < warmup_env_steps
        collector = get_collector(scan_steps)
        key, ck = jax.random.split(key)
        env_state, emb_hist, act_hist, _, traj = collector(
            env_state, emb_hist, act_hist, ck, jnp.bool_(use_random)
        )
        obs_t, act_t, rew_t, done_t = traj
        # ONE host transfer + ONE batched buffer push for the whole chunk.
        buffer.add_chunk(
            np.asarray(obs_t), np.asarray(act_t),
            np.asarray(rew_t), np.asarray(done_t),
        )

        # -- Train -- (replicate per-step trigger cadence over the chunk)
        prev_steps = env_steps
        env_steps += chunk_env_steps
        if buffer.can_sample(int(cfg.batch_size)):
            s = prev_steps + num_envs
            while s <= env_steps:
                if s >= warmup_env_steps and s % train_every < num_envs:
                    for _ in range(updates_per_step):
                        batch = buffer.sample(int(cfg.batch_size), rng_np)
                        state, last_metrics = agent.update(batch, state)
                s += num_envs

        # -- Eval --
        if env_steps >= next_eval:
            ret = eval_actor(n_eps=5)
            write_eval(env_steps, ret)
            sps = int(env_steps / max(time.time() - t0, 1))
            wm_loss = float(last_metrics.get("wm/total", 0.0))
            print(f"  es={env_steps:>9,}  sps={sps}  eval={ret:7.1f}  wm_loss={wm_loss:.3f}", flush=True)
            # iter-29: persist a checkpoint so se_attention_graph.py can analyze the trained
            # transformer-WM's attention graph (run_dreamer4 previously saved nothing).
            try:
                import pickle as _pk
                from pathlib import Path as _P
                _twm = state["wm"]
                _ckdir = _P(ev_csv).parent / f"seed_{seed}" / "checkpoints"
                _ckdir.mkdir(parents=True, exist_ok=True)
                _cfg = {k: getattr(_twm, k) for k in (
                    "embed_dim", "action_dim", "d_model", "n_layers", "n_heads",
                    "context_len", "mlp_ratio", "activation", "pos_encoding", "max_seq_len")
                    if hasattr(_twm, k)}
                _pk.dump({"wm_params": jax.device_get(state["wm_params"]),
                          "actor_params": jax.device_get(state["actor_params"]),
                          "transformer_config": _cfg, "obs_dim": obs_dim, "act_dim": act_dim,
                          "env_id": env_id, "seed": seed, "step": env_steps},
                         open(_ckdir / "latest.pkl", "wb"))
            except Exception as _e:
                print(f"  [dreamer4] ckpt save failed: {_e}", flush=True)
            next_eval += eval_interval

    print(f"  [dreamer4] done {env_id} seed={seed} at {env_steps:,} env-steps "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# CLI / env-var glue (queue convention)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Standalone DreamerV4 transformer-WM launcher.")
    ap.add_argument("--task", default=os.environ.get("TASK", "PandaPickCube"),
                    help="MuJoCo Playground env id (default from $TASK or PandaPickCube).")
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", os.environ.get("SEED", "1")),
                    help="Space-separated seed list (default from $SEEDS/$SEED or '1').")
    ap.add_argument("--total_steps", type=int,
                    default=int(os.environ.get("TOTAL_STEPS", "3000000")),
                    help="Total env steps per seed (default $TOTAL_STEPS or 3,000,000).")
    ap.add_argument("--num_envs", type=int, default=int(os.environ.get("NUM_ENVS", "16")))
    ap.add_argument("--warmup_env_steps", type=int,
                    default=int(os.environ.get("WARMUP_ENV_STEPS", "25000")))
    ap.add_argument("--no_plot", action="store_true", help="Accepted for queue compatibility; no-op.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    probe = os.environ.get("PROBE_ID", "dreamer4")
    sha = os.environ.get("CODE_SHA", "unknown")
    seeds = [int(s) for s in str(args.seeds).split() if s.strip()]
    print(f"[{probe}] DreamerV4(transformer)  task={args.task}  seeds={seeds}  "
          f"total_steps={args.total_steps:,}  sha={sha}  "
          f"tag={os.environ.get('TDMPC_GLASS_OUTPUT_TAG', '')}", flush=True)

    failed = False
    for seed in seeds:
        try:
            train_dreamer4(
                args.task,
                args.total_steps,
                seed,
                num_envs=args.num_envs,
                warmup_env_steps=args.warmup_env_steps,
            )
        except Exception as e:  # noqa: BLE001 - report per-seed and continue
            failed = True
            print(f"\nERROR in dreamer4/{args.task} seed={seed}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        import gc
        gc.collect()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
