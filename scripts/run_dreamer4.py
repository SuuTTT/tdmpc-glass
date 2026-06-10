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

    # ── Collection + training loop ──
    key, rk = jax.random.split(key)
    env_state = batch_reset(rk)
    obs_np = np.array(env_state.obs)
    emb_hist = np.zeros((num_envs, ctx, embed_dim), np.float32)
    act_hist = np.zeros((num_envs, ctx, act_dim), np.float32)

    env_steps = 0
    next_eval = eval_interval
    t0 = time.time()
    last_metrics: dict[str, float] = {}

    print(f"  [dreamer4] training to {total_steps:,} env-steps "
          f"(num_envs={num_envs}, warmup={warmup_env_steps:,})", flush=True)

    while env_steps < total_steps:
        # Encode current obs and append to the rolling embedding window.
        emb = np.array(encode(state["wm_params"], jnp.asarray(obs_np)))
        emb_hist = roll_window(emb_hist, emb)

        # -- Act --
        if env_steps < warmup_env_steps:
            acts_np = rng_np.uniform(-1.0, 1.0, (num_envs, act_dim)).astype(np.float32)
        else:
            key, sk = jax.random.split(key)
            action = policy_step(
                state["wm_params"], state["actor_params"],
                jnp.asarray(emb_hist), jnp.asarray(act_hist), sk, False,
            )
            acts_np = np.array(action, dtype=np.float32)

        # Record the action taken into the action window.
        act_hist = roll_window(act_hist, acts_np)

        env_state = batch_step(env_state, jnp.asarray(acts_np))
        new_obs = np.array(env_state.obs)
        rews_np = np.array(env_state.reward, np.float32)
        done_np = np.array(env_state.done > 0.5, np.float32)

        for i in range(num_envs):
            buffer.add_transition(obs_np[i], acts_np[i], float(rews_np[i]), bool(done_np[i]))

        # Reset the rolling windows for envs that terminated (auto-reset obs).
        if done_np.any():
            m = done_np > 0.5
            emb_hist[m] = 0.0
            act_hist[m] = 0.0

        obs_np = new_obs
        env_steps += num_envs

        # -- Train --
        if (
            env_steps >= warmup_env_steps
            and env_steps % train_every < num_envs
            and buffer.can_sample(int(cfg.batch_size))
        ):
            for _ in range(updates_per_step):
                batch = buffer.sample(int(cfg.batch_size), rng_np)
                state, last_metrics = agent.update(batch, state)

        # -- Eval --
        if env_steps >= next_eval:
            ret = eval_actor(n_eps=5)
            write_eval(env_steps, ret)
            sps = int(env_steps / max(time.time() - t0, 1))
            wm_loss = float(last_metrics.get("wm/total", 0.0))
            print(f"  es={env_steps:>9,}  sps={sps}  eval={ret:7.1f}  wm_loss={wm_loss:.3f}", flush=True)
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
