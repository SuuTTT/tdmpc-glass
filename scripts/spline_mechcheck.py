#!/usr/bin/env python3
"""P2 mechanism-check (standalone; does NOT touch run_benchmark hot path).

Gates the "Hermite-spline action bottleneck" lever (docs/research/abstraction-axes-plan.md §P2,
RESEARCH_LEDGER "NEXT PROBES #1"): does a cubic-Hermite-spline restriction of the action sequence
preserve achievable return on Franka manipulation?

Procedure
  1. Roll out a trained TD-MPC2 checkpoint CLOSED-loop (policy reacts each step, mirrors
     run_benchmark / value_probe eval exactly), recording per-step actions and the TRUE env return.
  2. Per episode, fit a cubic Hermite spline to each action dim with knots every k steps
     (knot values = recorded actions at knot timesteps, knot tangents = Catmull-Rom finite
     differences), reconstruct the full-rate action sequence.
  3. REPLAY the episode OPEN-loop from the SAME initial state (identical reset key — MJX is
     deterministic given (state, actions)) with the splined actions; record splined return.
  4. CONTROL: piecewise-constant (zero-order hold) actions at the same knot spacing. If ZOH
     preserves return as well as splines, the spline structure is unnecessary.

Metric: return_preservation = splined_return / reference_return per episode.
GO rule (pre-registered in the plan): mean preservation >= 0.95 at knot_every=4.

HONEST CAVEAT (also stored in the JSON): the reference rollout is CLOSED-loop while the
spline/ZOH replays are OPEN-loop (fixed action sequence, no reaction). That asymmetry is the
point — it upper-bounds how much return the spline restriction loses *on the expert trajectory*.
But open-loop replay UNDERSTATES what spline-MPPI could achieve, because spline-MPPI replans
every step and can correct drift. So a PASS here is meaningful evidence the lever can work;
a marginal FAIL is not necessarily fatal.

CPU/GPU worker friendly; no training. Output = JSON (read-from-JSON discipline).
  python scripts/spline_mechcheck.py --ckpt <best_mppi.pkl> --task PandaPickCube
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np

# match run_benchmark's sys.path so mujoco_playground + helios resolve on a worker
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))

import jax
import jax.numpy as jnp
from helios.algorithms.tdmpc2 import Encoder, Pi
from mujoco_playground import registry, wrapper


# ── spline machinery (pure numpy, deterministic) ─────────────────────────────────────────────

def knot_indices(T, k):
    """Knot timesteps 0, k, 2k, ... plus the final step T-1 (so the spline spans the episode)."""
    idx = list(range(0, T, k))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    return np.asarray(idx, dtype=np.int64)


def catmull_rom_tangents(t, y):
    """Standard Catmull-Rom / finite-difference tangents at the knots.
    t: (K,) knot times, y: (K, d) knot values -> (K, d) tangents (dy/dt).
    Interior: (y[i+1]-y[i-1])/(t[i+1]-t[i-1]); one-sided differences at the ends."""
    K = len(t)
    m = np.zeros_like(y)
    if K == 1:
        return m
    m[0] = (y[1] - y[0]) / (t[1] - t[0])
    m[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    for i in range(1, K - 1):
        m[i] = (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])
    return m


def hermite_eval(t_knots, y, m, t_query):
    """Evaluate the cubic Hermite spline at integer times t_query.
    t_knots: (K,), y/m: (K, d), t_query: (T,) -> (T, d)."""
    seg = np.clip(np.searchsorted(t_knots, t_query, side="right") - 1, 0, len(t_knots) - 2)
    t0 = t_knots[seg].astype(np.float64)
    t1 = t_knots[seg + 1].astype(np.float64)
    h = t1 - t0
    s = ((t_query - t0) / h)[:, None]                       # (T,1) in [0,1]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return (h00 * y[seg] + h10 * h[:, None] * m[seg]
            + h01 * y[seg + 1] + h11 * h[:, None] * m[seg + 1])


def spline_actions(A, k):
    """Cubic-Hermite reconstruction of the action sequence A (T, d) with knots every k steps."""
    T = A.shape[0]
    ki = knot_indices(T, k)
    if len(ki) < 2:                                         # degenerate (T <= 1): passthrough
        return A.copy()
    t = ki.astype(np.float64)
    y = A[ki].astype(np.float64)
    m = catmull_rom_tangents(t, y)
    rec = hermite_eval(t, y, m, np.arange(T, dtype=np.float64))
    return np.clip(rec, -1.0, 1.0).astype(np.float32)       # actions are tanh-bounded


def zoh_actions(A, k):
    """Zero-order-hold control: hold the action sampled at the most recent knot."""
    T = A.shape[0]
    ki = knot_indices(T, k)
    seg = np.clip(np.searchsorted(ki, np.arange(T), side="right") - 1, 0, len(ki) - 1)
    return A[ki[seg]].astype(np.float32)


# ── main ─────────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="PandaPickCube")
    ap.add_argument("--n_ep", type=int, default=8)
    ap.add_argument("--knot_every", default="4,8", help="comma list of knot spacings to test")
    ap.add_argument("--out", default=None,
                    help="default: exp/tdmpc_glass/mechcheck/spline_mechcheck_<task>.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode_length", type=int, default=1000)
    ap.add_argument("--latent_dim", type=int, default=512)
    ap.add_argument("--V", type=int, default=8)
    args = ap.parse_args()
    knot_list = [int(x) for x in args.knot_every.split(",") if x.strip()]
    out_path = Path(args.out) if args.out else (
        _ROOT / "exp/tdmpc_glass/mechcheck" / f"spline_mechcheck_{args.task}.json")
    hidden = (512, 512)

    # ── env (mirror value_probe / run_benchmark: registry.load -> wrap_for_brax_training,
    #         single env via split-1). MJX => deterministic given (reset key, action sequence),
    #         which is what makes the open-loop replay from the same key meaningful.
    env = registry.load(args.task)
    env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1)
    obs_dim, act_dim = env.observation_size, env.action_size
    reset = jax.jit(lambda k: env.reset(jax.random.split(k, 1)))
    step = jax.jit(lambda st, a: env.step(st, a[None]))

    # ── nets + checkpoint params (same pattern as value_probe; only enc + pi needed here)
    enc_net = Encoder(latent_dim=args.latent_dim, hidden=hidden, V=args.V)
    pi_net = Pi(action_dim=act_dim, hidden=hidden)
    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    @jax.jit
    def policy_action(obs):  # deterministic policy action tanh(mu), as in eval
        z = enc_net.apply(params["enc"], obs[None])
        mu, _ = pi_net.apply(params["pi"], z)
        return jnp.tanh(mu)[0]

    def rollout_closed(reset_key):
        """Reference: closed-loop policy rollout. Returns (actions (T,d), return, T)."""
        st = reset(reset_key)
        obs = st.obs[0]
        acts, ret = [], 0.0
        for _t in range(args.episode_length):
            a = np.asarray(policy_action(obs), np.float32)
            acts.append(a)
            st = step(st, jnp.asarray(a))
            ret += float(st.reward[0])
            if bool(st.done[0] > 0.5):
                break
            obs = st.obs[0]
        return np.stack(acts), ret, len(acts)

    def rollout_open(reset_key, actions):
        """Open-loop replay of a fixed action sequence from the SAME reset key."""
        st = reset(reset_key)
        ret, steps = 0.0, 0
        for a in actions:
            st = step(st, jnp.asarray(a))
            ret += float(st.reward[0])
            steps += 1
            if bool(st.done[0] > 0.5):
                break
        return ret, steps

    # ── episodes (deterministic given --seed; identical key reused for every replay)
    key = jax.random.PRNGKey(args.seed)
    episodes = []
    for ep in range(args.n_ep):
        key, rk = jax.random.split(key)
        A, ref_ret, T = rollout_closed(rk)
        rec = {"ep": ep, "T": T, "reference_return": round(ref_ret, 4)}
        for k in knot_list:
            As = spline_actions(A, k)
            Az = zoh_actions(A, k)
            sp_ret, sp_T = rollout_open(rk, As)
            zh_ret, zh_T = rollout_open(rk, Az)
            denom = ref_ret if abs(ref_ret) > 1e-8 else None
            rec[f"k{k}"] = {
                "spline_return": round(sp_ret, 4),
                "zoh_return": round(zh_ret, 4),
                "spline_preservation": round(sp_ret / denom, 4) if denom else None,
                "zoh_preservation": round(zh_ret / denom, 4) if denom else None,
                "spline_steps": sp_T, "zoh_steps": zh_T,
                "spline_l2_per_step": round(float(np.linalg.norm(As - A, axis=1).mean()), 5),
                "zoh_l2_per_step": round(float(np.linalg.norm(Az - A, axis=1).mean()), 5),
            }
            print(f"[ep {ep}] T={T} ref={ref_ret:.3f} k={k} "
                  f"spline={sp_ret:.3f} zoh={zh_ret:.3f}", flush=True)
        episodes.append(rec)

    # ── aggregate + verdict (pre-registered: GO iff mean spline preservation >= 0.95 at k=4)
    summary = {}
    for k in knot_list:
        sp = [e[f"k{k}"]["spline_preservation"] for e in episodes
              if e[f"k{k}"]["spline_preservation"] is not None]
        zh = [e[f"k{k}"]["zoh_preservation"] for e in episodes
              if e[f"k{k}"]["zoh_preservation"] is not None]
        summary[f"k{k}"] = {
            "spline_preservation_mean": round(float(np.mean(sp)), 4) if sp else None,
            "spline_preservation_min": round(float(np.min(sp)), 4) if sp else None,
            "zoh_preservation_mean": round(float(np.mean(zh)), 4) if zh else None,
            "zoh_preservation_min": round(float(np.min(zh)), 4) if zh else None,
            "spline_l2_mean": round(float(np.mean(
                [e[f"k{k}"]["spline_l2_per_step"] for e in episodes])), 5),
            "zoh_l2_mean": round(float(np.mean(
                [e[f"k{k}"]["zoh_l2_per_step"] for e in episodes])), 5),
            "n_episodes_scored": len(sp),
        }

    gate_k = 4 if 4 in knot_list else knot_list[0]
    gate_mean = summary[f"k{gate_k}"]["spline_preservation_mean"]
    go = (gate_mean is not None) and (gate_mean >= 0.95)
    zoh_mean = summary[f"k{gate_k}"]["zoh_preservation_mean"]
    spline_adds_over_zoh = (gate_mean is not None and zoh_mean is not None
                            and gate_mean > zoh_mean + 0.02)

    out = {
        "probe": "P2 spline mechanism-check (abstraction-axes-plan §P2 / ledger NEXT PROBES #1)",
        "config": {
            "ckpt": args.ckpt, "task": args.task, "n_ep": args.n_ep,
            "knot_every": knot_list, "seed": args.seed,
            "episode_length": args.episode_length, "latent_dim": args.latent_dim,
            "obs_dim": int(obs_dim), "act_dim": int(act_dim),
            "spline": "cubic Hermite, knot values = recorded actions, "
                      "Catmull-Rom finite-difference tangents, clipped to [-1,1]",
            "control": "zero-order hold at the same knot spacing",
        },
        "episodes": episodes,
        "summary": summary,
        "verdict": {
            "rule": f"GO iff mean spline return-preservation >= 0.95 at knot_every={gate_k} "
                    "(pre-registered in abstraction-axes-plan.md §P2)",
            "gate_knot_every": gate_k,
            "spline_preservation_mean": gate_mean,
            "zoh_preservation_mean": zoh_mean,
            "GO": bool(go),
            "spline_adds_over_zoh": bool(spline_adds_over_zoh),
            "zoh_note": "if ZOH preserves return as well as splines, the spline structure is "
                        "unnecessary — the bottleneck could be even simpler",
        },
        "caveat": ("Reference rollout is CLOSED-loop (policy reacts); spline/ZOH replays are "
                   "OPEN-loop (fixed action sequence). This upper-bounds the return lost by the "
                   "spline restriction on the expert trajectory, but UNDERSTATES what spline-MPPI "
                   "could achieve since it replans every step. A PASS is meaningful; a marginal "
                   "FAIL is not necessarily fatal."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"summary": summary, "verdict": out["verdict"]}, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
