#!/usr/bin/env python3
"""Experiment #2: HL phases as a learned SKILL-OPTIONS abstraction on PandaPickCube.

CONTRAST to experiment #1 (raw-action residual). Here we learn at the OPTION /
PHASE level: a high-level policy pi_hi(context) chooses the OPTION PARAMETERS
(the continuous Knobs + small per-phase target nudges) of the analytic phase
controller, which then EXECUTES the option (reach->descend->grasp->lift->place).
The low-level execution is the unchanged analytic controller (param_controller).

  context  = [initial_box_xyz, target_xyz]            (6-dim)
  d        = pi_hi(context)  in [-1,1]^KV_DIM         (bounded knob deltas)
  kv       = clip(KV_BASE + KV_SCALE*d, lo, hi)       (effective option params)
  action_t = ParamPick.act(state_t, cstate_t, kv)     (analytic option exec)

This is the VALUE-AWARE / STATE-CONDITIONED abstraction the campaign asks about:
the abstraction's parameters ADAPT to the situation (box/target geometry), vs the
HL controller's FIXED knobs (~6.25% true success). The thesis under test: a
SEMANTIC, task-grounded, value-aware temporal abstraction works where flat fixed-k
'jumpy' abstraction (null->harmful) and structural-entropy abstraction failed.

TWO MODES:
  --mode global  : pi_hi is a single learned global d vector (context-free).
                   This is the SMOKE/BASELINE: kv constant across envs. d==0
                   reproduces the v9 HL controller (~6% success) at init.
  --mode mlp     : pi_hi is a tiny MLP(6 -> H -> KV_DIM) with tanh output, so
                   the option params are conditioned on (box0,target). Output
                   init ~0 so it starts at the v9 HL controller, then adapts.

TRAINER: antithetic OpenAI-ES, rank-normalized (same as residual_train.py).
Fitness is a shaped proxy used only to RANK; we SELECT/REPORT TRUE success
(box_target>=0.9 at any step) on a held-out eval set. Reports PEAK and FINAL.
Nothing fabricated: all numbers come from real rollouts -> RESULTS.json/PROGRESS.md.

Usage (smoke, global):
  python options_train.py --mode global --pop 32 --envs_per 2 --iters 6 \
      --eval_envs 128 --tag smoke --out_dir /path
Usage (full, mlp):
  python options_train.py --mode mlp --pop 128 --envs_per 2 --iters 350 \
      --eval_envs 512 --sigma 0.05 --lr 0.03 --tag full --out_dir /path
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJPG_IMPL", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")

import argparse, json, time, sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jp
from mujoco_playground import registry

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import param_controller as PC

CTX_DIM = 6
KV_DIM = PC.KV_DIM

# HL ceilings / comparison anchors (from files: curve_v9_rhold.json etc.)
CMP = {
    "HL_heuristic_v9_curve": 0.0625,   # curve_v9_rhold.json success_rate
    "HL_heuristic_v9_harness": 0.0977,
    "flat_jumpy_TDMPC2": 0.0,          # jumpy temporal abstraction: null->harmful
    "structural_entropy": 0.0,         # failed (campaign)
    "exp1_residual_peak": 0.11,        # experiment #1 raw-action residual (~0.10-0.11)
    "PPO_official": 0.66,
    "PPO_env_steps": 33_000_000,
}


# ---------------- high-level policy pi_hi ----------------
def init_theta_global(key):
    # context-free: a single d vector, start at 0 (==v9 HL controller)
    return jp.zeros((KV_DIM,))


def pi_hi_global(theta, ctx):
    # ignore ctx; same d for every env. tanh keeps it bounded.
    return jp.tanh(theta)


MLP_H = 32  # tiny hidden layer


def init_theta_mlp(key):
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (CTX_DIM, MLP_H)) * (1.0 / np.sqrt(CTX_DIM))
    b1 = jp.zeros((MLP_H,))
    # tiny last layer so output ~0 -> starts at v9 HL controller
    W2 = jax.random.normal(k2, (MLP_H, KV_DIM)) * 0.01
    b2 = jp.zeros((KV_DIM,))
    return [(W1, b1), (W2, b2)]


def pi_hi_mlp(theta, ctx):
    (W1, b1), (W2, b2) = theta
    h = jp.tanh(ctx @ W1 + b1)
    return jp.tanh(h @ W2 + b2)  # in [-1,1]^KV_DIM


def flatten(theta):
    flat, tree = jax.tree_util.tree_flatten(theta)
    shapes = [a.shape for a in flat]
    vec = jp.concatenate([a.ravel() for a in flat])
    return vec, (tree, shapes)


def unflatten(vec, aux):
    tree, shapes = aux
    arrs, i = [], 0
    for s in shapes:
        n = int(np.prod(s)) if s else 1
        arrs.append(vec[i:i + n].reshape(s))
        i += n
    return jax.tree_util.tree_unflatten(tree, arrs)


# ---------------- rollout: HL options executed by analytic controller ----------
def build_rollout(env, n_steps, mode):
    ctrl = PC.make_param_controller(env)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    act_vmap = jax.vmap(ctrl.act)  # (state, cstate, kv) all batched
    init_cs_single = ctrl.init_state()
    pi_hi = pi_hi_global if mode == "global" else pi_hi_mlp

    def rollout(theta_vec, aux, keys):
        theta = unflatten(theta_vec, aux)
        n = keys.shape[0]
        state = reset(keys)
        state = step(state, jp.zeros((n, 8)))
        cs = jax.tree_util.tree_map(
            lambda x: jp.broadcast_to(x, (n,) + x.shape), init_cs_single)
        box0 = state.data.xpos[:, env._obj_body]            # (n,3) initial box xyz
        box0z = box0[:, 2]
        target = state.info["target_pos"]                   # (n,3)
        # high-level policy emits per-env option params ONCE from context.
        ctx = jax.vmap(PC.context_of)(box0, target)         # (n,6)
        d = jax.vmap(lambda c: pi_hi(theta, c))(ctx)        # (n,KV_DIM) in [-1,1]
        kv = jax.vmap(PC.d_to_knobs)(d)                      # (n,KV_DIM) effective

        def body(carry, _):
            state, cs = carry
            action, cs = act_vmap(state, cs, kv)
            state = step(state, action)
            box = state.data.xpos[:, env._obj_body]
            site = state.data.site_xpos[:, env._gripper_site]
            fw = state.data.qpos[:, 7]
            dd = jp.linalg.norm(box - site, axis=-1)
            tele = {
                "box_target": state.metrics["box_target"],
                "reached": state.info["reached_box"],
                "box_z": box[:, 2],
                "fw": fw,
                "d": dd,
                "phase": cs["phase"],
            }
            return (state, cs), tele

        (state, cs), tele = jax.lax.scan(body, (state, cs), None, length=n_steps)
        bt_max = tele["box_target"].max(axis=0)
        success = (tele["box_target"] >= 0.9).any(axis=0).astype(jp.float32)
        reached = (tele["reached"] > 0.5).any(axis=0).astype(jp.float32)
        lift = tele["box_z"] - box0z[None, :]
        # phase-reached telemetry (which phase the abstraction improves)
        ph = tele["phase"]
        ph_descend = (ph >= PC.DESCEND).any(axis=0).astype(jp.float32)
        grasp = ((tele["fw"] < PC.FIX_close_threshold) & (tele["fw"] > 0.005)
                 & (tele["d"] < 0.04)).any(axis=0).astype(jp.float32)
        lift_ok = ((lift > 0.06) & (tele["d"] < 0.05)).any(axis=0).astype(jp.float32)
        ph_lift = (ph >= PC.LIFT).any(axis=0).astype(jp.float32)
        ph_place = (ph >= PC.PLACE).any(axis=0).astype(jp.float32)
        max_lift = lift.max(axis=0)
        return {
            "bt_max": bt_max, "success": success, "reached": reached,
            "grasp": grasp, "lift_ok": lift_ok, "max_lift": max_lift,
            "ph_descend": ph_descend, "ph_lift": ph_lift, "ph_place": ph_place,
        }

    return rollout, ctrl


def fitness_from(summ):
    """Shaped proxy correlated with TRUE success (RANK only). success dominates;
    box_target + lift + place-phase give dense gradient below the ceiling."""
    return (4.0 * summ["success"] + summ["bt_max"]
            + 0.5 * jp.clip(summ["max_lift"], 0.0, 0.2) / 0.2
            + 0.25 * summ["lift_ok"] + 0.15 * summ["ph_place"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["global", "mlp"], default="mlp")
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--envs_per", type=int, default=2)
    ap.add_argument("--iters", type=int, default=350)
    ap.add_argument("--eval_envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_every", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
    print(f"env obs={env.observation_size} act={env.action_size} dt={env.dt} "
          f"mode={args.mode} KV_DIM={KV_DIM}", flush=True)
    rollout, ctrl = build_rollout(env, args.steps, args.mode)

    key = jax.random.PRNGKey(args.seed)
    key, kp = jax.random.split(key)
    init_theta = init_theta_global if args.mode == "global" else init_theta_mlp
    theta0, aux = flatten(init_theta(kp))
    D = theta0.shape[0]
    print(f"pi_hi params D={D} (mode={args.mode}), pop={args.pop}(x2 antithetic), "
          f"envs_per={args.envs_per}, eval_envs={args.eval_envs}, steps={args.steps}",
          flush=True)

    pop_full = args.pop * 2

    def eval_population(theta, eps, keys_pe):
        thetas = jp.concatenate([theta[None] + args.sigma * eps,
                                 theta[None] - args.sigma * eps], axis=0)
        per_member = jax.vmap(lambda th, ks: rollout(th, aux, ks))(thetas, keys_pe)
        fit = jax.vmap(fitness_from)(per_member)
        return fit.mean(axis=1), per_member

    eval_population_j = jax.jit(eval_population)

    def eval_mean(theta, keys):
        summ = rollout(theta, aux, keys)
        return {k: jp.mean(v) for k, v in summ.items()}
    eval_mean_jit = jax.jit(eval_mean)

    def eval_mean_j(theta, keys):
        d = eval_mean_jit(theta, keys)
        return {k: float(v) for k, v in d.items()}

    theta = theta0
    hist = []
    best_success = 0.0
    best_theta = theta0
    env_steps_total = 0
    t_start = time.time()

    APPROACH_STR = ("HL phases as a learned SKILL-OPTIONS abstraction: "
                    "pi_hi(context=[box0,target]) -> option-params "
                    "(continuous Knobs + per-phase xyz nudges); analytic phase "
                    "controller executes the option. Learns the ABSTRACTION "
                    "(value-aware, state-conditioned), NOT raw per-step deltas.")

    def write_progress(it, last_eval, final_eval=None, done=False):
        peak = best_success
        final = (final_eval or last_eval or {}).get("success")
        L = []
        L.append("# Experiment #2: HL phases as a learned SKILL-OPTIONS abstraction (ES) on PandaPickCube")
        L.append("")
        L.append(f"approach: {APPROACH_STR}")
        L.append(f"param space: mode={args.mode} ({'context-free single global d vector' if args.mode=='global' else 'tiny MLP(6->%d->%d), context-conditioned'%(MLP_H,KV_DIM)}); "
                 f"KV_DIM={KV_DIM} (16 continuous knobs + 9 per-phase xyz nudges); pi_hi params D={D}")
        L.append(f"trainer=Evolution Strategies (antithetic OpenAI-ES, rank-normalized); "
                 f"pop={args.pop}(x2), envs_per={args.envs_per}, eval_envs={args.eval_envs}, "
                 f"steps={args.steps}, sigma={args.sigma}, lr={args.lr}")
        L.append("")
        L.append(f"iter {it}/{args.iters}  env_steps={env_steps_total:,}  "
                 f"elapsed={time.time()-t_start:.0f}s  {'DONE' if done else 'RUNNING'}")
        L.append("")
        L.append("## TRUE SUCCESS (held-out eval, box_target>=0.9)")
        L.append(f"- PEAK success : {peak:.4f}")
        L.append(f"- FINAL success: {final if final is not None else 'n/a'}")
        if last_eval:
            L.append(f"- last eval: success={last_eval.get('success'):.4f} "
                     f"reached={last_eval.get('reached'):.4f} grasp={last_eval.get('grasp'):.4f} "
                     f"lift_ok={last_eval.get('lift_ok'):.4f} bt_max={last_eval.get('bt_max'):.4f} "
                     f"| phase: descend={last_eval.get('ph_descend'):.3f} "
                     f"lift={last_eval.get('ph_lift'):.3f} place={last_eval.get('ph_place'):.3f}")
        L.append("")
        L.append("## CEILINGS / COMPARISON (TRUE success)")
        L.append(f"- HL heuristic fixed-knobs   : 0.0625 (curve) / 0.0977 (harness)")
        L.append(f"- flat fixed-k 'jumpy'       : 0.0 (null->harmful)")
        L.append(f"- structural-entropy abstr.  : 0.0 (failed)")
        L.append(f"- exp #1 raw-action residual : ~0.11 (peak)")
        L.append(f"- PPO (official brax)        : 0.66 @ 33M env-steps")
        L.append(f"- HL-OPTIONS (THIS, peak)    : {peak:.4f}  @ {env_steps_total:,} env-steps")
        (out_dir / "PROGRESS.md").write_text("\n".join(L) + "\n")

    def write_results(final_eval=None, status="running", last_eval=None):
        res = {
            "experiment": "HL_phases_as_learned_SKILL_OPTIONS_abstraction_ES",
            "approach": APPROACH_STR,
            "param_space": {
                "mode": args.mode,
                "description": ("context-free single global d vector" if args.mode == "global"
                                else f"context-conditioned tiny MLP(6->{MLP_H}->{KV_DIM})"),
                "context": "[initial_box_xyz, target_xyz] (6-dim)",
                "kv_dim": KV_DIM,
                "kv_names": PC.KV_NAMES,
                "n_continuous_knobs": 16,
                "n_phase_offsets": 9,
                "pi_hi_param_count": int(D),
                "fixed_structural_knobs": {"use_aik": 1, "aik_from": "GRASP",
                                           "ik_gain": 1.0, "ik_iters": 12},
            },
            "trainer": "EvolutionStrategies (antithetic OpenAI-ES, rank-normalized)",
            "sigma": args.sigma, "lr": args.lr, "pop": args.pop,
            "pop_antithetic": pop_full, "envs_per_member": args.envs_per,
            "eval_envs": args.eval_envs, "steps_per_ep": args.steps,
            "seed": args.seed,
            "status": status,
            "env_steps_used": int(env_steps_total),
            "iters_done": len(hist),
            "success_peak": round(best_success, 4),
            "success_final": (round(final_eval["success"], 4) if final_eval else None),
            "final_eval": ({k: round(v, 4) for k, v in final_eval.items()} if final_eval else None),
            "phase_telemetry": ({
                "descend_reached": round((final_eval or last_eval or {}).get("ph_descend", float("nan")), 4),
                "grasp_ok": round((final_eval or last_eval or {}).get("grasp", float("nan")), 4),
                "lift_reached": round((final_eval or last_eval or {}).get("ph_lift", float("nan")), 4),
                "lift_ok": round((final_eval or last_eval or {}).get("lift_ok", float("nan")), 4),
                "place_reached": round((final_eval or last_eval or {}).get("ph_place", float("nan")), 4),
                "reached_box": round((final_eval or last_eval or {}).get("reached", float("nan")), 4),
            } if (final_eval or last_eval) else None),
            "history": hist[-200:],
            "comparison": dict(CMP, HL_OPTIONS_peak=round(best_success, 4),
                               HL_OPTIONS_env_steps=int(env_steps_total)),
            "elapsed_sec": round(time.time() - t_start, 1),
            "wrote_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        txt = json.dumps(res, indent=2)
        json.loads(txt)  # validate
        (out_dir / "RESULTS.json").write_text(txt)

    last_eval = None
    for it in range(1, args.iters + 1):
        key, ke, kk = jax.random.split(key, 3)
        eps = jax.random.normal(ke, (args.pop, D))
        keys_flat = jax.random.split(kk, pop_full * args.envs_per)
        keys_pe = keys_flat.reshape(pop_full, args.envs_per, 2)
        fit, _ = eval_population_j(theta, eps, keys_pe)
        fit = jax.device_get(fit)
        env_steps_total += pop_full * args.envs_per * args.steps

        f = np.asarray(fit)
        ranks = np.empty_like(f)
        ranks[np.argsort(f)] = np.arange(len(f))
        u = ranks / (len(f) - 1) - 0.5
        u_plus = u[:args.pop]
        u_minus = u[args.pop:]
        grad = (np.asarray(eps).T @ (u_plus - u_minus)) / (2 * args.sigma * args.pop)
        theta = theta + args.lr * jp.asarray(grad)

        rec = {"iter": it, "env_steps": int(env_steps_total),
               "fit_mean": round(float(f.mean()), 4), "fit_max": round(float(f.max()), 4)}

        if it % args.eval_every == 0 or it == 1 or it == args.iters:
            key, kv = jax.random.split(key)
            ev_keys = jax.random.split(kv, args.eval_envs)
            ev = eval_mean_j(theta, ev_keys)
            last_eval = ev
            rec["eval_success"] = round(ev["success"], 4)
            rec["eval_reached"] = round(ev["reached"], 4)
            rec["eval_grasp"] = round(ev["grasp"], 4)
            rec["eval_lift_ok"] = round(ev["lift_ok"], 4)
            rec["eval_bt_max"] = round(ev["bt_max"], 4)
            rec["eval_ph_place"] = round(ev["ph_place"], 4)
            if ev["success"] > best_success:
                best_success = ev["success"]
                best_theta = theta
                np.save(out_dir / "best_theta.npy", np.asarray(jax.device_get(theta)))
            print(f"[it {it:4d}] steps={env_steps_total:>10,} fit={f.mean():.3f}/{f.max():.3f} "
                  f"EVAL success={ev['success']:.4f} reached={ev['reached']:.4f} "
                  f"grasp={ev['grasp']:.4f} lift={ev['lift_ok']:.4f} bt={ev['bt_max']:.4f} "
                  f"place={ev['ph_place']:.3f} (peak={best_success:.4f})", flush=True)
            hist.append(rec)
            write_progress(it, ev)
            write_results(status="running", last_eval=ev)
        else:
            hist.append(rec)
            if it % 10 == 0:
                print(f"[it {it:4d}] steps={env_steps_total:>10,} fit={f.mean():.3f}/{f.max():.3f}",
                      flush=True)
                write_progress(it, last_eval)

    # final honest eval on a fresh, larger held-out set with BEST theta
    key, kf = jax.random.split(key)
    fin_keys = jax.random.split(kf, max(args.eval_envs, 512))
    final_eval = eval_mean_j(best_theta, fin_keys)
    if final_eval["success"] > best_success:
        best_success = final_eval["success"]
    print(f"\n==== FINAL (best_theta, {len(fin_keys)} held-out envs) ====", flush=True)
    for k, v in final_eval.items():
        print(f"  {k}: {v:.4f}", flush=True)
    print(f"  PEAK success: {best_success:.4f}", flush=True)
    print(f"  env_steps_used: {env_steps_total:,}", flush=True)
    np.save(out_dir / "best_theta.npy", np.asarray(jax.device_get(best_theta)))
    write_progress(args.iters, final_eval, final_eval, done=True)
    write_results(final_eval=final_eval, status="done")
    print(f"\nwrote {out_dir}/RESULTS.json and PROGRESS.md", flush=True)


if __name__ == "__main__":
    main()
