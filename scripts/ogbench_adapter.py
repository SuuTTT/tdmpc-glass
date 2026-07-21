"""OGBench -> mujoco_playground-style env adapter (Phase 1 cross-benchmark bridge).

Goal: let scripts/run_benchmark.py train our TD-MPC2 ONLINE on an OGBench
gymnasium env without touching the MJX/brax training loop. The loop only ever
touches the env through:
    env.observation_size, env.action_size
    state = env.reset(keys)              # keys ignored; we seed internally
    state = env.step(state, acts)        # acts: (N, act_dim) in [-1,1]
    state.obs   (N, obs_dim)  float32 jax array
    state.reward(N,)          float32 jax array
    state.done  (N,)          float32 jax array  (1.0 on terminal transition)
plus single-env eval which calls reset/step on an N=1 env and reads [0].

OGBench envs are CPU gymnasium MuJoCo envs (reset()/step()). We run N of them in
a python loop and expose the brax-style batched interface, including brax-style
AUTO-RESET (the training loop never resets mid-episode — it relies on the env
auto-resetting a sub-env on the step where it terminates, and returning the new
episode's first obs in state.obs with done=1.0 for that transition).

This module is import-light (numpy + gymnasium/ogbench only). It returns plain
numpy arrays wrapped in a tiny State; run_benchmark uses non-jit env wrappers on
the OGBench path (a CPU gym step cannot be traced by jax.jit).
"""

import os
import numpy as np


class State:
    """Minimal brax-style env state. Carries numpy arrays; the training loop does
    np.array(state.obs) etc., so numpy is fine (no jax tracing on this path)."""
    __slots__ = ("obs", "reward", "done", "info")

    def __init__(self, obs, reward, done, info=None):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info or {}


class OGBenchBatchEnv:
    """Vector of N CPU OGBench gym envs with brax-style auto-reset.

    success: OGBench step() returns info with a 'success' key for singletask
    envs; we surface the per-episode success as part of done bookkeeping and
    expose last-episode success via .last_success for eval logging.
    """

    def __init__(self, name, n_envs, seed=0, max_episode_steps=None):
        import ogbench
        self.name = name
        self.n_envs = int(n_envs)
        self._make = lambda: ogbench.make_env_and_datasets(name, compact_dataset=False)[0]
        self.envs = [self._make() for _ in range(self.n_envs)]
        self._seeds = [int(seed) + i for i in range(self.n_envs)]
        # probe dims + action bounds from a temporary reset
        o0, _ = self.envs[0].reset(seed=self._seeds[0])
        o0 = np.asarray(o0, dtype=np.float32).reshape(-1)
        self.observation_size = int(o0.shape[0])
        self.action_size = int(np.asarray(self.envs[0].action_space.shape).prod())
        self._alow = np.asarray(self.envs[0].action_space.low, dtype=np.float32)
        self._ahigh = np.asarray(self.envs[0].action_space.high, dtype=np.float32)
        # max steps per episode (truncation) — OGBench sets this internally; we
        # also keep a hard cap so a stuck env doesn't run forever.
        env_max = getattr(self.envs[0], "_max_episode_steps", None)
        try:
            env_max = env_max or self.envs[0].spec.max_episode_steps
        except Exception:
            pass
        self.max_episode_steps = int(max_episode_steps or env_max or 1000)
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)
        self._ep_succ = np.zeros(self.n_envs, dtype=np.float32)
        self.last_success = 0.0      # success rate of episodes that ended this batch
        self._reset_all_pending = True

    # actions arrive in [-1,1]; OGBench action spaces are already [-1,1] for these
    # tasks, but rescale defensively to the env bounds.
    def _scale(self, a):
        a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
        return self._alow + (a + 1.0) * 0.5 * (self._ahigh - self._alow)

    def reset(self, keys=None):
        n = self.n_envs if keys is None else (len(keys) if hasattr(keys, "__len__") else int(getattr(keys, "shape", [self.n_envs])[0]))
        # keys may be a jax array of shape (n, 2); we only use its length.
        n = self.n_envs
        obs = np.zeros((n, self.observation_size), dtype=np.float32)
        for i in range(n):
            o, _ = self.envs[i].reset(seed=self._seeds[i])
            self._seeds[i] += 10_000  # advance per-env seed stream
            obs[i] = np.asarray(o, dtype=np.float32).reshape(-1)
        self._ep_len[:] = 0
        self._ep_succ[:] = 0.0
        self._reset_all_pending = False
        return State(obs,
                     np.zeros(n, dtype=np.float32),
                     np.zeros(n, dtype=np.float32))

    def step(self, state, acts):
        acts = np.asarray(acts, dtype=np.float32)
        n = self.n_envs
        obs = np.zeros((n, self.observation_size), dtype=np.float32)
        rew = np.zeros(n, dtype=np.float32)
        done = np.zeros(n, dtype=np.float32)
        ended_succ = []
        step_succ = np.zeros(n, dtype=np.float32)
        scaled = self._scale(acts)
        for i in range(n):
            o, r, term, trunc, info = self.envs[i].step(scaled[i])
            self._ep_len[i] += 1
            s = float(info.get("success", 0.0))
            step_succ[i] = s
            self._ep_succ[i] = max(self._ep_succ[i], s)
            d = bool(term) or bool(trunc) or (self._ep_len[i] >= self.max_episode_steps)
            rew[i] = float(r)
            if d:
                done[i] = 1.0
                ended_succ.append(float(self._ep_succ[i]))
                # brax-style auto-reset: return next episode's first obs
                o2, _ = self.envs[i].reset(seed=self._seeds[i])
                self._seeds[i] += 10_000
                obs[i] = np.asarray(o2, dtype=np.float32).reshape(-1)
                self._ep_len[i] = 0
                self._ep_succ[i] = 0.0
            else:
                obs[i] = np.asarray(o, dtype=np.float32).reshape(-1)
        if ended_succ:
            self.last_success = float(np.mean(ended_succ))
        return State(obs, rew, done, info={"success": step_succ})


def make_ogbench_env(name, n_envs, seed=0):
    """Factory used by run_benchmark when TDMPC_ENV_SRC=ogbench:<name>."""
    return OGBenchBatchEnv(name, n_envs, seed=seed)
