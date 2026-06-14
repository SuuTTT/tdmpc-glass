#!/usr/bin/env python3
"""CONTROL-BENEFIT test for the GWM result — the DECISIVE GATE.

============================================================================
THE QUESTION (the campaign's core lesson, made operational)
============================================================================
The companion mechanism check (``scripts/gwm_simulator_mechcheck.py``) measures
REPRESENTATION properties of the graph world model (GWM): compositional-OOD
value-decodability (can you linearly read the return out of the latent at
held-out object counts?) and contact-conditioned prediction. A graph latent may
WIN on those.

But the campaign's hard-won lesson is that **representation properties do not
auto-convert to control** ("basin-entry wins were procedure tricks, not
abstraction"). So this script asks the only question that decides whether the GWM
result matters for an agent:

    Does the graph WM's OOD value-decodability advantage translate into better
    CONTROL RETURN at held-out object counts?

We answer it WITHOUT any policy RL: we do model-predictive control (MPC) using
the LEARNED world model directly. Better WM -> better plans -> higher TRUE-env
return. The graph WM and the (fair) monolithic baseline are driven by the SAME
planner; the only thing that differs is the model whose ``next_ent`` + ``reward``
heads the planner queries. If the graph WM's representation edge is real for
control, its TRUE-env return at OOD N must beat the monolithic baseline's.

============================================================================
WHAT IS COMPARED (FAIR baseline)
============================================================================
  * GRAPH WM   = ``helios.dynamics.entity_wm.EntityWM`` (transformer over
                 entities; N-agnostic).
  * MONOLITHIC = ``helios.dynamics.monolithic_wm.MonolithicWM`` with
                 ``mode="pad"`` — the FAIR baseline (preserves per-entity slots
                 up to max_entities, no pooling-collapse artifact). This is the
                 same fairness control the mechcheck flags as the apples-to-apples
                 unstructured comparator. We do NOT use mode="pool" (its OOD
                 degradation could be a pooling artifact, not a graph win).
  * RANDOM     = uniform random actions in [-1,1]^2 — the sanity FLOOR.

Both WMs are trained on the SAME data at N_train with the EXACT protocol of
``gwm_simulator_mechcheck`` (same data collection, same standardized Q target,
same steps / episodes / batch / lr / seed, param-matched at N_train). The
mechcheck training functions are imported and reused verbatim — this script adds
ONLY the planner + the true-env return evaluation.

============================================================================
THE MPC PLANNER (model-predictive control with the learned WM)
============================================================================
Short-horizon RANDOM-SHOOTING (a.k.a. random-shooting MPC / one-shot CEM-0):

  At each TRUE-env step, given the current per-entity state ``ent`` (N, d):
    1. Sample M action SEQUENCES of horizon H: ``acts`` ~ U([-1,1]^2),
       shape (M, H, 2).
    2. Roll all M sequences forward THROUGH THE LEARNED WM (receding-horizon
       imagination). At each imagined step h:
         out = wm.apply(params, ent_batch (M,N,d), acts[:, h])  # batched over M
         rew_h = out["reward"]      # (M,) the WM's reward head (MODEL reward)
         ent_batch = out["next_ent"]  # (M,N,d) the WM's next-state head
       Score each sequence by the discounted (gamma) sum of MODEL reward over H.
    3. Pick the best sequence (argmax score) and EXECUTE only its FIRST action in
       the TRUE env (receding horizon — replan every step).

This is GENERIC over the WM: it only touches ``out["next_ent"]`` and
``out["reward"]``, which BOTH EntityWM and MonolithicWM expose with identical
shapes ((B,N,d) and (B,)). The SAME ``plan_action`` function is therefore called
with each model's (apply_fn, params) — see ``ModelPlanner``. No per-model special
casing; the only difference between the two control runs is which model's heads
the rollout queries. Planning uses MODEL reward (the WM is the simulator); the
VERDICT is scored on TRUE-ENV reward only.

============================================================================
PRE-REGISTERED VERDICT (stated before running)
============================================================================
CONTROL-GO iff:
  (1) graph mean TRUE-env return exceeds pad-monolithic mean TRUE-env return by a
      CLEAR MARGIN (>= 15% RELATIVE) at >= 1 held-out (OOD) N, AND
  (2) graph >= mono IN-DISTRIBUTION (N_train) — no in-dist regression.
NO-GO otherwise -> the OOD value-decode advantage does NOT convert to control
(representation-only; the redundancy story holds at the controller).

Relative margin is computed on a SHIFTED return scale so it is well-defined for
the env's (negative, distance-based) returns: rel = (graph - mono) / |mono_floor|
where mono_floor anchors to the random-action return at that N (the sanity
floor) — i.e. we measure improvement OVER the random floor. The raw and shifted
numbers are both persisted so the verdict is auditable. (See ``_rel_gain``.)

============================================================================
HONEST CAVEATS (persisted in the JSON)
============================================================================
  * Random-shooting MPC is a WEAK planner (no CEM refinement / no policy prior);
    it is a coarse WM-quality -> control transducer, not a strong controller.
  * The planner uses MODEL reward to plan (the WM IS the simulator); only the
    reported return uses TRUE-env reward. A WM with a poor reward head will plan
    badly regardless of its value-decodability — this is part of "does it convert
    to control", but it means a NO-GO can come from the reward head, not the
    latent. Reward-head R^2 (from the mechcheck) is the relevant diagnostic.
  * Synthetic elastic disks are NOT real manipulation; controlled proxy only.
  * SINGLE training seed per run (--seed); no multi-seed error bars on training.
    Eval episodes ARE paired across models (same eval seeds) for a clean A/B.
  * Models have MODEST R^2; short-horizon plans can be dominated by compounding
    model error. If returns hug the random floor for BOTH models (no separation
    from RANDOM at ANY N), the test is UNINFORMATIVE — flagged in the JSON.

============================================================================
IF THE SIGNAL IS WEAK (reward density tuning — read this)
============================================================================
The env reward is DENSE-ish for short-horizon MPC: it is
``-dist(target, goal) - w_agent*dist(agent, target) + bonus*1[success]``. The two
distance terms give a smooth per-step gradient the planner can climb on EVERY
step (not just at sparse success), so a horizon-H plan that moves the agent toward
the target / the target toward the goal already scores higher model reward. That
is why short-horizon random-shooting can show signal here. If in practice ALL
returns hug the random floor (uninformative), tune toward MORE per-step signal:
  - raise ``--mpc_horizon`` (longer imagined credit assignment),
  - raise ``--mpc_samples`` (better argmax over the action space),
  - the env's ``w_agent`` shaping weight and ``action_scale`` make the dense
    terms bite harder (would require an env change — out of scope here; flagged).
These are reported so a weak result is diagnosable rather than silently null.

Runs on a GPU/CPU worker with jax+flax+optax. NOT on the control box (no GPU);
EC2 does py_compile only. Worker jobs set XLA_PYTHON_CLIENT_PREALLOCATE=false.

CLI:
    --n_train 5 --n_ood 7,9 --steps 6000 --episodes 256 \
    --mpc_samples 512 --mpc_horizon 8 --eval_episodes 20 --seed 0 \
    --out exp/tdmpc_glass/mechcheck/gwm_control_benefit_s0.json
Deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helios.dynamics.entity_wm import EntityWM, entity_wm_loss
from helios.dynamics.monolithic_wm import (
    MonolithicWM,
    matched_hidden,
    monolithic_wm_loss,
    _count_params,
)
from helios.envs import contact_entities as ce

# Reuse the mechcheck's data-collection + training loop EXACTLY (same protocol).
import gwm_simulator_mechcheck as mech  # noqa: E402  (sibling script, same dir)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "exp/tdmpc_glass/mechcheck/gwm_control_benefit.json"


# ---------------------------------------------------------------------------
# Generic model-predictive controller (random-shooting MPC) over a learned WM.
# ---------------------------------------------------------------------------


class ModelPlanner:
    """Random-shooting MPC driven by a LEARNED world model.

    GENERIC over the WM: it queries ONLY ``out["next_ent"]`` (B,N,d) and
    ``out["reward"]`` (B,), the two heads BOTH EntityWM and MonolithicWM expose
    with identical shapes. The SAME class therefore plans on both models — the
    only thing that changes is the (apply_fn, params) passed in. No per-model
    branching anywhere below.

    Args:
        apply_fn: model ``.apply`` (Flax bound method); called as
                  ``apply_fn({"params": params}, ent (B,N,d), action (B,A))``.
        params:   trained params for that model.
        action_dim: action dimensionality (2 for contact_entities).
        n_samples: M action sequences sampled per replan.
        horizon:  H imagined steps per sequence.
        gamma:    discount over the imagined horizon (model-reward scoring).
    """

    def __init__(self, apply_fn, params, action_dim, n_samples, horizon, gamma=0.99):
        self.action_dim = int(action_dim)
        self.n_samples = int(n_samples)
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        disc = self.gamma ** jnp.arange(self.horizon)  # (H,) static discounts

        def _plan(ent, key):
            """One receding-horizon replan from current per-entity state ``ent``.

            Args:
                ent: (N, d) current TRUE per-entity state.
                key: PRNG key for sampling the M action sequences.
            Returns:
                first_action: (action_dim,) first action of the best sequence.
            """
            N, d = ent.shape
            M, H = self.n_samples, self.horizon
            # Sample M action sequences ~ U([-1,1])^(M,H,A).
            acts = jax.random.uniform(
                key, (M, H, self.action_dim), minval=-1.0, maxval=1.0
            )
            # Initial imagined state: broadcast current ent to the M-batch.
            ent_b = jnp.broadcast_to(ent[None], (M, N, d))

            def step(carry, h):
                eb = carry
                a_h = acts[:, h, :]  # (M, A)
                out = apply_fn({"params": params}, eb, a_h)
                return out["next_ent"], out["reward"]  # (M,N,d), (M,)

            _, rews = jax.lax.scan(step, ent_b, jnp.arange(H))  # rews (H, M)
            scores = jnp.sum(rews * disc[:, None], axis=0)      # (M,) discounted
            best = jnp.argmax(scores)
            return acts[best, 0, :]

        # jit once per planner instance (graph captures M/H/params statically).
        self._plan = jax.jit(_plan)

    def act(self, ent, key):
        """Plan + return the first action (numpy (action_dim,))."""
        return self._plan(ent, key)


def random_planner_factory(action_dim):
    """A planner-shaped callable that ignores the model and returns uniform
    random actions (the sanity FLOOR). Same call signature as ModelPlanner.act."""

    def act(ent, key):
        return jax.random.uniform(key, (action_dim,), minval=-1.0, maxval=1.0)

    return act


# ---------------------------------------------------------------------------
# True-env episode rollout under a given planner (TRUE reward is scored).
# ---------------------------------------------------------------------------


def run_episode(env, planner_act, key, ep_len):
    """Roll one TRUE-env episode under ``planner_act``; return total TRUE reward.

    ``planner_act(ent (N,d), plan_key) -> action (action_dim,)`` is either a
    ModelPlanner.act (MPC) or the random-floor callable. The env reset + step are
    the genuine contact_entities dynamics; reward summed is the TRUE env reward
    (NEVER the model reward — that is only used inside the planner to choose).
    """
    k_reset, k_plan = jax.random.split(key)
    state = env.reset(k_reset)
    # jit the single env step for speed (pure function of state+action).
    step_fn = jax.jit(env.step)
    total = 0.0
    for _t in range(ep_len):
        ent = env.observe_entities(state)         # (N, d)
        k_plan, ak = jax.random.split(k_plan)
        action = planner_act(ent, ak)             # (action_dim,)
        state, _obs, reward, done, _info = step_fn(state, action)
        total += float(reward)
        if bool(done):
            break
    return total


def eval_return_at_n(env, planner_act, key, eval_episodes, ep_len):
    """Mean/std/per-seed TRUE-env return over ``eval_episodes`` PAIRED episodes.

    The per-episode keys are derived deterministically from ``key`` so the SAME
    eval seeds are reused across the graph / mono / random planners at this N
    (paired A/B). Returns a dict of stats."""
    ep_keys = jax.random.split(key, eval_episodes)
    rets = [run_episode(env, planner_act, ek, ep_len) for ek in ep_keys]
    rets_a = np.asarray(rets, dtype=np.float64)
    return {
        "return_mean": float(rets_a.mean()),
        "return_std": float(rets_a.std()),
        "return_per_seed": [float(x) for x in rets_a],
        "n_episodes": int(rets_a.size),
    }


# ---------------------------------------------------------------------------
# Verdict helper: relative gain on a floor-anchored scale.
# ---------------------------------------------------------------------------


def _rel_gain(graph_mean, mono_mean, floor_mean):
    """Relative improvement of graph over mono, anchored to the random floor.

    The env returns are negative (distance-based) so a raw (g-m)/|m| ratio is
    ill-conditioned. We measure improvement OVER the random-action floor:
        gain_graph = graph_mean - floor_mean
        gain_mono  = mono_mean  - floor_mean
        rel        = (gain_graph - gain_mono) / |gain_mono|
    i.e. how much more the graph planner beats the floor than the mono planner
    does, relative to mono's own beat-the-floor margin. Falls back to a raw
    relative gain on |mono_mean| when mono does not separate from the floor.
    """
    gain_g = graph_mean - floor_mean
    gain_m = mono_mean - floor_mean
    if abs(gain_m) > 1e-9:
        return float((gain_g - gain_m) / abs(gain_m)), "floor_anchored"
    denom = abs(mono_mean) if abs(mono_mean) > 1e-9 else 1e-9
    return float((graph_mean - mono_mean) / denom), "raw_fallback"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="CONTROL-BENEFIT test: does GWM OOD value-decode edge convert "
        "to better MPC control return at held-out object counts?"
    )
    # --- shared with mechcheck (same training protocol) ---
    p.add_argument("--n_train", type=int, default=5)
    p.add_argument("--n_ood", type=str, default="7,9")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--episodes", type=int, default=256,
                   help="training-data collection episodes (same as mechcheck).")
    p.add_argument("--ep_len", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mono_layers", type=int, default=2)
    # The FAIR baseline is the pad monolithic. Fixed here (not pool) by design.
    p.add_argument("--mono_mode", type=str, default="pad", choices=["pool", "pad"],
                   help="monolithic baseline encoder; default 'pad' = the FAIR "
                        "control (no pooling-collapse artifact).")
    # --- MPC / control eval ---
    p.add_argument("--mpc_samples", type=int, default=512,
                   help="M action sequences sampled per replan.")
    p.add_argument("--mpc_horizon", type=int, default=8,
                   help="H imagined steps per sequence.")
    p.add_argument("--mpc_gamma", type=float, default=0.99,
                   help="discount over the imagined horizon (model-reward score).")
    p.add_argument("--eval_episodes", type=int, default=20,
                   help="TRUE-env episodes per N (paired across models).")
    p.add_argument("--eval_ep_len", type=int, default=100,
                   help="TRUE-env episode length for control eval.")
    # --- verdict threshold (pre-registered) ---
    p.add_argument("--rel_margin", type=float, default=0.15,
                   help="required relative control-return margin (graph over mono) "
                        "at >=1 OOD N for CONTROL-GO. Pre-registered at 0.15.")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = p.parse_args()
    args.n_ood = [int(x) for x in str(args.n_ood).split(",") if x.strip()]
    # Tag default --out with the seed so multi-seed runs don't clobber each other.
    if args.out == DEFAULT_OUT:
        stem, ext = os.path.splitext(args.out)
        args.out = f"{stem}_s{args.seed}{ext}"
    return args


def main():
    args = parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    key = jax.random.PRNGKey(args.seed)

    # === TRAIN BOTH WMs — REUSE the mechcheck protocol verbatim ==============
    # (same data collection, same standardized Q target, same trainer, same
    #  steps/episodes/batch/lr/seed, param-matched at N_train.)
    env_train = ce.make_env(n_entities=args.n_train, seed=args.seed)
    k_data, k_gtrain, k_mtrain, key = jax.random.split(key, 4)
    data = mech.collect_dataset(env_train, k_data, args.episodes, args.ep_len)

    # Standardize the Q/return target exactly as mechcheck does.
    ret = np.asarray(data["ret"])
    ret_mean, ret_std = float(ret.mean()), float(ret.std() + 1e-8)
    data = dict(data)
    data["ret"] = (data["ret"] - ret_mean) / ret_std

    # GRAPH WM (uses mech.make_graph_model + mech.train_model).
    graph_model = mech.make_graph_model(args, args.n_train)
    graph_params, g_metrics, g_nparams = mech.train_model(
        graph_model, entity_wm_loss, data, k_gtrain, args
    )

    # MONOLITHIC (pad) WM, param-matched at N_train (mech.matched_hidden path).
    hidden, mono_target_nparams = matched_hidden(
        g_nparams, ce.ENTITY_DIM, 2, args.n_train, n_layers=args.mono_layers,
        mode=args.mono_mode, max_entities=mech.mono_max_entities(args),
    )
    mono_model = mech.make_mono_model(args, args.n_train, hidden)
    mono_params, m_metrics, m_nparams = mech.train_model(
        mono_model, monolithic_wm_loss, data, k_mtrain, args
    )

    # === BUILD the SAME planner over each WM ================================
    graph_planner = ModelPlanner(
        graph_model.apply, graph_params, action_dim=2,
        n_samples=args.mpc_samples, horizon=args.mpc_horizon, gamma=args.mpc_gamma,
    )
    mono_planner = ModelPlanner(
        mono_model.apply, mono_params, action_dim=2,
        n_samples=args.mpc_samples, horizon=args.mpc_horizon, gamma=args.mpc_gamma,
    )
    random_act = random_planner_factory(action_dim=2)

    # === EVALUATE TRUE-env control return at N_train AND held-out N =========
    per_n = {}
    for n in [args.n_train] + args.n_ood:
        env_n = ce.make_env(n_entities=n, seed=args.seed)
        # ONE eval base key per N, REUSED across graph / mono / random. Because
        # split() is deterministic, episode i then gets the SAME reset seed (and
        # the same planner-noise sub-stream) for all three planners -> strictly
        # paired A/B on identical initial states.
        k_eval, key = jax.random.split(key)
        base = k_eval  # shared base => paired episodes across the three planners
        g_stats = eval_return_at_n(env_n, graph_planner.act, base,
                                   args.eval_episodes, args.eval_ep_len)
        m_stats = eval_return_at_n(env_n, mono_planner.act, base,
                                   args.eval_episodes, args.eval_ep_len)
        r_stats = eval_return_at_n(env_n, random_act, base,
                                   args.eval_episodes, args.eval_ep_len)
        gm = g_stats["return_mean"]
        mm = m_stats["return_mean"]
        rm = r_stats["return_mean"]
        rel, rel_mode = _rel_gain(gm, mm, rm)
        per_n[str(n)] = {
            "is_ood": bool(n in args.n_ood),
            "graph": g_stats,
            "monolithic_pad": m_stats,
            "random_floor": r_stats,
            "ood_return_gap_graph_minus_mono": float(gm - mm),
            "graph_minus_random": float(gm - rm),
            "mono_minus_random": float(mm - rm),
            "relative_gain_graph_over_mono": rel,
            "relative_gain_mode": rel_mode,
            "value_relevant_entities": list(env_n.value_relevant_entities),
        }
        print(
            f"N={n}{' (OOD)' if n in args.n_ood else ''}: "
            f"return graph={gm:.3f} mono(pad)={mm:.3f} random={rm:.3f} | "
            f"gap(g-m)={gm - mm:+.3f} rel={rel:+.3f} ({rel_mode})",
            flush=True,
        )

    # === PRE-REGISTERED VERDICT =============================================
    # CONTROL-GO iff:
    #   (1) graph beats mono(pad) by >= rel_margin RELATIVE at >=1 OOD N, AND
    #   (2) graph >= mono IN-DISTRIBUTION (no in-dist regression).
    indist = per_n[str(args.n_train)]
    indist_no_regression = bool(
        indist["graph"]["return_mean"] >= indist["monolithic_pad"]["return_mean"]
    )
    ood_pass_ns = [
        n for n in args.n_ood
        if per_n[str(n)]["relative_gain_graph_over_mono"] >= args.rel_margin
        and per_n[str(n)]["ood_return_gap_graph_minus_mono"] > 0.0
    ]
    crit_ood = bool(len(ood_pass_ns) >= 1)
    go = bool(crit_ood and indist_no_regression)

    # Uninformative-test flag: if NEITHER model separates from the random floor at
    # ANY N, the planner/model is too weak to score control — the test can't speak.
    sep_thresh = 1e-3
    any_separation = any(
        (per_n[str(n)]["graph_minus_random"] > sep_thresh)
        or (per_n[str(n)]["mono_minus_random"] > sep_thresh)
        for n in [args.n_train] + args.n_ood
    )
    uninformative = bool(not any_separation)

    verdict = {
        "control_go": go,
        "criteria_preregistered": (
            "CONTROL-GO iff (graph mean TRUE-env return exceeds pad-monolithic by "
            ">=15% RELATIVE at >=1 held-out N) AND (graph >= mono in-distribution "
            "[no in-dist regression]). NO-GO otherwise -> the OOD value-decode "
            "advantage does NOT convert to control (representation-only; the "
            "redundancy story holds at the controller)."
        ),
        "criterion_1_ood_control_margin": {
            "rel_margin_threshold": args.rel_margin,
            "ood_N_passing": ood_pass_ns,
            "passed": crit_ood,
            "per_ood_relative_gain": {
                str(n): per_n[str(n)]["relative_gain_graph_over_mono"]
                for n in args.n_ood
            },
        },
        "criterion_2_no_indist_regression": {
            "n_train": args.n_train,
            "graph_indist_return": indist["graph"]["return_mean"],
            "mono_indist_return": indist["monolithic_pad"]["return_mean"],
            "passed": indist_no_regression,
        },
        "uninformative_test_flag": {
            "flagged": uninformative,
            "note": (
                "True iff NEITHER model's mean return separates from the random "
                "floor at ANY N (separation threshold {:.0e}). When flagged, the "
                "MPC/WM is too weak to convert ANY representation edge to control "
                "-> the test cannot speak; tune --mpc_horizon / --mpc_samples "
                "(see module docstring)."
            ).format(sep_thresh),
        },
        "interpretation": (
            "CONTROL-GO: the graph WM's OOD value-decodability edge DOES convert "
            "to better MPC control return at held-out object counts -> the GWM "
            "result matters for an agent. NO-GO: representation-only; the edge "
            "does not reach the controller (redundancy story holds), the "
            "campaign's core lesson once more."
        ),
    }

    caveats = [
        "Random-shooting MPC is a WEAK planner (no CEM refinement / no policy "
        "prior); a coarse WM-quality->control transducer, not a strong controller.",
        "The planner uses MODEL reward to plan (the WM is the simulator); only the "
        "reported return uses TRUE-env reward. A poor reward head can cause a "
        "NO-GO independent of the latent — check reward-head R^2 in the mechcheck.",
        "Synthetic elastic disks are NOT real manipulation; controlled proxy only.",
        f"Single training seed (seed={args.seed}); no multi-seed error bars on "
        "training. Eval episodes ARE paired across models (same eval seeds).",
        "Models have MODEST R^2; short-horizon plans can be dominated by "
        "compounding model error. If returns hug the random floor for BOTH models "
        "at ALL N, the test is uninformative (see uninformative_test_flag).",
    ]

    out = {
        "tag": "gwm_control_benefit",
        "config": vars(args),
        "ret_standardization": {"mean": ret_mean, "std": ret_std},
        "param_counts": {
            "graph_entity_wm": g_nparams,
            "monolithic_wm": m_nparams,
            "monolithic_hidden": hidden,
            "monolithic_target_at_init": mono_target_nparams,
            "match_ratio_mono_over_graph": float(m_nparams / g_nparams),
        },
        "train_metrics": {"graph": g_metrics, "monolithic_pad": m_metrics},
        "mpc": {
            "planner": "random-shooting MPC (receding horizon); SAME planner over "
                       "both WMs (queries only next_ent + reward heads).",
            "n_samples": args.mpc_samples,
            "horizon": args.mpc_horizon,
            "gamma": args.mpc_gamma,
            "plan_reward": "MODEL reward (WM as simulator)",
            "scored_reward": "TRUE-env reward",
        },
        "per_n": per_n,
        "verdict": verdict,
        "caveats": caveats,
        "wall_time_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gwm_control_benefit] wrote {out_path}")
    print(json.dumps(verdict, indent=2))
    print(json.dumps(out["param_counts"], indent=2))


if __name__ == "__main__":
    main()
