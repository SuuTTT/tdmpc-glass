#!/usr/bin/env python3
"""iter-19 Stage-2 — skill-reach-rate probe (controllability test).

Loads a FROZEN Glass checkpoint + its community file, builds goal-conditioned MPPI
(make_goal_mppi_fn), and for each community measures how often the planner can NAVIGATE
the agent into that community from random starts — vs a random-action baseline.

GATE: goal-MPPI reach-rate > random clearly on >=3 communities -> Stage-2 PASS (the learned
dynamics+planner can execute community-reaching skills -> Stage-3 can sequence them).

Run ON A BOX (needs jax + mujoco_playground):
  PYTHONPATH=src python3 scripts/skill_eval.py --ckpt <pkl> --community_file <npz> \
       --task CartpoleSwingupSparse [--horizon 12 --episodes 10 --cap 200]
"""
import argparse, os, pickle, sys, importlib.util
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/root/mujoco_playground_repo")
from mujoco_playground import registry, wrapper
from helios.algorithms.tdmpc_glass import Encoder, Dynamics, Pi, make_goal_mppi_fn, DEFAULTS

# skills.py without the jax-heavy package __init__
spec = importlib.util.spec_from_file_location("skills", str(Path(__file__).resolve().parents[1] / "src/helios/algorithms/skills.py"))
skills = importlib.util.module_from_spec(spec); spec.loader.exec_module(skills)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--community_file", required=True)
    ap.add_argument("--task", default="CartpoleSwingupSparse")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=512)
    args = ap.parse_args()

    d = dict(DEFAULTS)
    latent_dim, hidden, V = d["latent_dim"], d["hidden"], d["V"]

    env = registry.load(args.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    print(f"task={args.task} obs={obs_dim} act={act_dim} H={args.horizon}", flush=True)

    enc = Encoder(latent_dim=latent_dim, hidden=hidden, V=V)
    dyn = Dynamics(latent_dim=latent_dim, hidden=hidden, V=V)
    pi = Pi(action_dim=act_dim, hidden=hidden)

    ck = pickle.load(open(args.ckpt, "rb"))
    params = ck["params"]
    comm = np.load(args.community_file)
    protos, labels, centroids = comm["prototypes"], comm["labels"], comm["centroids"]
    n_comm = int(comm["n_comm"])

    plan = make_goal_mppi_fn(enc, dyn, pi, horizon=args.horizon, n_samples=args.n_samples,
                             num_elites=64, num_pi_trajs=24, n_iter=6, act_dim=act_dim)

    @jax.jit
    def enc_apply(p, obs):
        return enc.apply(p["enc"], obs[None])[0]

    @jax.jit
    def step(state, act):
        return env.step(state, act[None])

    @jax.jit
    def reset(key):
        return env.reset(jax.random.split(key, 1))

    key = jax.random.PRNGKey(0)
    H = args.horizon

    def membership(obs):
        z = np.asarray(enc_apply(params, jnp.asarray(obs)))
        return int(skills.assign_community(z[None], protos, labels)[0])

    def run_episode(target, mode):
        nonlocal key
        key, rk = jax.random.split(key)
        state = reset(rk)
        obs = jnp.asarray(state.obs[0])
        goal = jnp.asarray(centroids[target])
        mu = jnp.zeros((H, act_dim)); std = jnp.full((H, act_dim), d["MAX_STD"])
        t0 = jnp.bool_(True)
        reached = False
        for _ in range(args.cap):
            if mode == "goal":
                key, pk = jax.random.split(key)
                act, mu, std = plan(params, obs, goal, mu, std, pk, t0); t0 = jnp.bool_(False)
            else:
                key, pk = jax.random.split(key)
                act = jax.random.uniform(pk, (act_dim,), minval=-1.0, maxval=1.0)
            state = step(state, act)
            obs = jnp.asarray(state.obs[0])
            if membership(np.asarray(obs)) == target:
                reached = True; break
            if bool(state.done[0] > 0.5):
                break
        return reached

    print(f"\nreach-rate over {args.episodes} eps/community (goal-MPPI vs random):", flush=True)
    g_tot = r_tot = wins = 0
    for c in range(n_comm):
        g = sum(run_episode(c, "goal") for _ in range(args.episodes)) / args.episodes
        r = sum(run_episode(c, "rand") for _ in range(args.episodes)) / args.episodes
        win = g > r + 0.15
        wins += win
        g_tot += g; r_tot += r
        print(f"  community {c} (size {int((labels==c).sum())}): goal={g:.2f} random={r:.2f} {'WIN' if win else ''}", flush=True)
    print(f"\nSUMMARY: goal-MPPI mean reach={g_tot/n_comm:.2f} vs random {r_tot/n_comm:.2f}; "
          f"clear wins (>random+0.15) on {wins}/{n_comm} communities", flush=True)
    print("STAGE-2 GATE:", "PASS" if wins >= 3 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
