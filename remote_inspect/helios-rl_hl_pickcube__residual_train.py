#!/usr/bin/env python3
"""Experiment #1: HL-controller + learned RESIDUAL policy on PandaPickCube.

APPROACH (OPTION A): EVOLUTION STRATEGIES on a small MLP residual.

  executed_action = clip( a_HL  +  alpha * tanh(pi_res(obs)) , -1, 1 )

where a_HL is the heuristic controller's per-step delta (controller.py, BEST=v9
knobs) and pi_res is a 2x256 MLP on the 66-dim env observation, output 8-dim.
alpha is a knob (default 0.3). The residual fixes only the hard grasp/lift/place
part the heuristic ceilings on (~6-10% success).

Why ES: gradient-free, robust to contact non-smoothness, and reuses the
vmapped-rollout harness pattern. The ES population is folded into the vmap batch
together with per-member eval envs (batch = POP * ENVS_PER_MEMBER). The phase
carry `cs` is threaded through a lax.scan exactly like harness.py.

SCORING: fitness is a shaped proxy (correlates with true success) used only to
RANK perturbations; we SELECT and REPORT on TRUE success (box_target>=0.9 at any
step) evaluated on a separate held-out env set with the current mean params.
Reports PEAK and FINAL. Telemetry (reached/grasp/lift) mirrors harness.analyze.

Nothing is fabricated: all reported numbers come from real rollouts and are
written to RESULTS.json / PROGRESS.md.

Usage (smoke):
  python residual_train.py --pop 64 --envs_per 1 --iters 40 --eval_envs 128 \
      --alpha 0.3 --tag smoke --out_dir /path
Usage (full):
  python residual_train.py --pop 256 --envs_per 2 --iters 400 --eval_envs 512 \
      --alpha 0.3 --sigma 0.05 --lr 0.03 --tag full --out_dir /path
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MJPG_IMPL", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import argparse, json, time, sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jp
from mujoco_playground import registry

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controller as C

# ----- v9 BEST knobs (success ceiling 0.0625-0.0977) -----
V9_KNOBS = dict(
    hover_height=0.1, grasp_z_offset=0.005, xy_tol=0.006, descend_z_tol=0.01,
    close_threshold=0.03, grasp_width=0.0, grasp_hold_steps=50,
    grasp_close_rate=0.04, lift_clear=0.12, lift_done_z=0.1,
    transport_xy_tol=0.03, place_tol=0.02, ik_gain=1.0, ik_iters=12,
    ori_w=0.0, aik_from=2, use_aik=1, cart_step=0.06, place_cart_step=0.06,
    place_gain=1.0, up_bias=0.02, approach_max=25, descend_max=25, grasp_max=55,
)

# ---------------- tiny MLP residual (functional params) ----------------
LAYERS = [66, 256, 256, 8]


def init_params(key):
    params = []
    ks = jax.random.split(key, len(LAYERS) - 1)
    for i, (din, dout) in enumerate(zip(LAYERS[:-1], LAYERS[1:])):
        # small init so the residual starts near zero (HL prior dominates).
        scale = (1.0 / np.sqrt(din)) * (0.01 if i == len(LAYERS) - 2 else 1.0)
        W = jax.random.normal(ks[i], (din, dout)) * scale
        b = jp.zeros((dout,))
        params.append((W, b))
    return params


def mlp(params, x):
    h = x
    for W, b in params[:-1]:
        h = jp.tanh(h @ W + b)
    W, b = params[-1]
    return jp.tanh(h @ W + b)  # in [-1,1]


def flatten(params):
    flat, tree = jax.tree_util.tree_flatten(params)
    shapes = [a.shape for a in flat]
    vec = jp.concatenate([a.ravel() for a in flat])
    return vec, (tree, shapes)


def unflatten(vec, aux):
    tree, shapes = aux
    arrs, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        arrs.append(vec[i:i + n].reshape(s))
        i += n
    return jax.tree_util.tree_unflatten(tree, arrs)


# ---------------- rollout (HL + residual), vmappable ----------------
def build_rollout(env, knobs, alpha, n_steps):
    ctrl = C.make_controller(env, knobs)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.vmap(env.step)
    act_hl = jax.vmap(ctrl.act)
    init_cs_single = ctrl.init_state()

    def rollout(theta_vec, aux, keys):
        """theta_vec: (D,) one residual param vector (same for all `keys` envs).
        keys: (M,) PRNG keys -> M envs run in parallel. Returns per-env telemetry."""
        params = unflatten(theta_vec, aux)
        n = keys.shape[0]
        state = reset(keys)
        state = step(state, jp.zeros((n, 8)))
        cs = jax.tree_util.tree_map(
            lambda x: jp.broadcast_to(x, (n,) + x.shape), init_cs_single)
        box0 = state.data.xpos[:, env._obj_body, 2]

        def res_for_obs(obs):
            return mlp(params, obs)
        res_vmap = jax.vmap(res_for_obs)

        def body(carry, _):
            state, cs = carry
            a_hl, cs = act_hl(state, cs)
            r = res_vmap(state.obs)                     # (n,8) in [-1,1]
            action = jp.clip(a_hl + alpha * r, -1.0, 1.0)
            state = step(state, action)
            box = state.data.xpos[:, env._obj_body]
            site = state.data.site_xpos[:, env._gripper_site]
            fw = state.data.qpos[:, 7]
            d = jp.linalg.norm(box - site, axis=-1)
            tele = {
                "box_target": state.metrics["box_target"],     # (n,)
                "reached": state.info["reached_box"],
                "box_z": box[:, 2],
                "fw": fw,
                "d": d,
            }
            return (state, cs), tele

        (state, cs), tele = jax.lax.scan(body, (state, cs), None, length=n_steps)
        # collapse over time to per-env summary (max over T)
        bt_max = tele["box_target"].max(axis=0)               # (n,)
        success = (tele["box_target"] >= 0.9).any(axis=0).astype(jp.float32)
        reached = (tele["reached"] > 0.5).any(axis=0).astype(jp.float32)
        lift = tele["box_z"] - box0[None, :]                  # (T,n)
        grasp = ((tele["fw"] < knobs.close_threshold) & (tele["fw"] > 0.005)
                 & (tele["d"] < 0.04)).any(axis=0).astype(jp.float32)
        lift_ok = ((lift > 0.06) & (tele["d"] < 0.05)).any(axis=0).astype(jp.float32)
        max_lift = lift.max(axis=0)                           # (n,)
        return {
            "bt_max": bt_max, "success": success, "reached": reached,
            "grasp": grasp, "lift_ok": lift_ok, "max_lift": max_lift,
        }

    return rollout, ctrl


def fitness_from(summ):
    """Shaped proxy that CORRELATES with true success (used only to RANK).
    success dominates; box_target + lift give dense gradient below the ceiling."""
    return (4.0 * summ["success"] + summ["bt_max"]
            + 0.5 * jp.clip(summ["max_lift"], 0.0, 0.2) / 0.2
            + 0.25 * summ["lift_ok"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--envs_per", type=int, default=2)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--eval_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_every", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    knobs = C.Knobs()
    for k, v in V9_KNOBS.items():
        setattr(knobs, k, type(getattr(knobs, k))(v))

    env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
    print(f"env obs={env.observation_size} act={env.action_size} dt={env.dt}", flush=True)
    rollout, ctrl = build_rollout(env, knobs, args.alpha, args.steps)

    key = jax.random.PRNGKey(args.seed)
    key, kp = jax.random.split(key)
    theta0, aux = flatten(init_params(kp))
    D = theta0.shape[0]
    print(f"residual params D={D}, pop={args.pop} (antithetic), envs_per={args.envs_per}, "
          f"eval_envs={args.eval_envs}, steps={args.steps}, alpha={args.alpha}", flush=True)

    # ----- vmapped population eval -----
    # batch dims: (pop_full, envs_per). antithetic doubles pop.
    pop_full = args.pop * 2

    def eval_population(theta, eps, keys_pe):
        # theta (D,), eps (pop, D) base noises -> antithetic (+/-)
        thetas = jp.concatenate([theta[None] + args.sigma * eps,
                                 theta[None] - args.sigma * eps], axis=0)  # (pop_full,D)
        # keys_pe: (pop_full, envs_per)
        per_member = jax.vmap(lambda th, ks: rollout(th, aux, ks))(thetas, keys_pe)
        fit = jax.vmap(fitness_from)(per_member)            # (pop_full, envs_per)
        return fit.mean(axis=1), per_member                 # mean over eval envs

    eval_population_j = jax.jit(eval_population)

    def eval_mean(theta, keys):
        summ = rollout(theta, aux, keys)
        return {k: jp.mean(v) for k, v in summ.items()}
    eval_mean_jit = jax.jit(eval_mean)

    def eval_mean_j(theta, keys):
        d = eval_mean_jit(theta, keys)
        return {k: float(v) for k, v in d.items()}

    # ES loop -----------------------------------------------------------------
    theta = theta0
    hist = []
    best_success = 0.0
    best_theta = theta0
    env_steps_total = 0
    t_start = time.time()

    def write_progress(it, last_eval, final_eval=None, done=False):
        peak = best_success
        final = (final_eval or last_eval or {}).get("success")
        lines = []
        lines.append("# Experiment #1: HL controller + learned RESIDUAL (ES) on PandaPickCube")
        lines.append("")
        lines.append(f"approach: HL(v9 knobs) + alpha*tanh(MLP2x256(obs)), alpha={args.alpha}; trainer=Evolution Strategies (antithetic, OpenAI-ES)")
        lines.append(f"pop={args.pop}(x2 antithetic), envs_per={args.envs_per}, eval_envs={args.eval_envs}, steps={args.steps}, sigma={args.sigma}, lr={args.lr}")
        lines.append("")
        lines.append(f"iter {it}/{args.iters}  env_steps={env_steps_total:,}  elapsed={time.time()-t_start:.0f}s  {'DONE' if done else 'RUNNING'}")
        lines.append("")
        lines.append("## TRUE SUCCESS (held-out eval, box_target>=0.9)")
        lines.append(f"- PEAK success : {peak:.4f}")
        lines.append(f"- FINAL success: {final if final is not None else 'n/a'}")
        if last_eval:
            lines.append(f"- last eval: success={last_eval.get('success'):.4f} "
                         f"reached={last_eval.get('reached'):.4f} grasp={last_eval.get('grasp'):.4f} "
                         f"lift_ok={last_eval.get('lift_ok'):.4f} bt_max={last_eval.get('bt_max'):.4f} "
                         f"max_lift={last_eval.get('max_lift'):.4f}")
        lines.append("")
        lines.append("## CEILINGS / COMPARISON")
        lines.append("- HL heuristic (v9_rhold)      : 0.0625 success (curve) / 0.0977 (harness)")
        lines.append("- PPO (official brax)          : 0.66 success @ ~33M env-steps")
        lines.append("- TD-MPC2 / jumpy              : 0.00 success (reward-hack hover)")
        lines.append(f"- HL+residual (THIS, peak)     : {peak:.4f}  @ {env_steps_total:,} env-steps")
        (out_dir / "PROGRESS.md").write_text("\n".join(lines) + "\n")

    def write_results(final_eval=None, status="running"):
        res = {
            "experiment": "HL_controller_plus_learned_residual_ES",
            "approach": "executed = clip(a_HL + alpha*tanh(MLP2x256(obs)), -1, 1); HL=v9 BEST knobs",
            "trainer": "EvolutionStrategies (antithetic OpenAI-ES, rank-normalized)",
            "alpha": args.alpha, "sigma": args.sigma, "lr": args.lr,
            "pop": args.pop, "pop_antithetic": pop_full, "envs_per_member": args.envs_per,
            "eval_envs": args.eval_envs, "steps_per_ep": args.steps,
            "residual_arch": LAYERS, "residual_param_count": int(D),
            "status": status,
            "env_steps_used": int(env_steps_total),
            "iters_done": len(hist),
            "success_peak": round(best_success, 4),
            "success_final": (round(final_eval["success"], 4) if final_eval else None),
            "final_eval": ({k: round(v, 4) for k, v in final_eval.items()} if final_eval else None),
            "history": hist[-200:],
            "comparison": {
                "HL_heuristic_v9_curve": 0.0625,
                "HL_heuristic_v9_harness": 0.0977,
                "PPO_official": 0.66,
                "PPO_env_steps": 33_000_000,
                "TDMPC2": 0.0,
                "HL_residual_peak": round(best_success, 4),
                "HL_residual_env_steps": int(env_steps_total),
            },
            "elapsed_sec": round(time.time() - t_start, 1),
            "wrote_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # validate JSON before writing
        txt = json.dumps(res, indent=2)
        json.loads(txt)
        (out_dir / "RESULTS.json").write_text(txt)

    for it in range(1, args.iters + 1):
        key, ke, kk = jax.random.split(key, 3)
        eps = jax.random.normal(ke, (args.pop, D))
        # distinct env keys per member+env (resample each iter for fresh layouts)
        keys_flat = jax.random.split(kk, pop_full * args.envs_per)
        keys_pe = keys_flat.reshape(pop_full, args.envs_per, 2)
        fit, _ = eval_population_j(theta, eps, keys_pe)
        fit = jax.device_get(fit)
        env_steps_total += pop_full * args.envs_per * args.steps

        # OpenAI-ES update with rank normalization (centered ranks in [-0.5,0.5])
        f = np.asarray(fit)
        ranks = np.empty_like(f)
        ranks[np.argsort(f)] = np.arange(len(f))
        u = ranks / (len(f) - 1) - 0.5
        u_plus = u[:args.pop]
        u_minus = u[args.pop:]
        # antithetic gradient estimate
        grad = (np.asarray(eps).T @ (u_plus - u_minus)) / (2 * args.sigma * args.pop)
        theta = theta + args.lr * jp.asarray(grad)

        rec = {"iter": it, "env_steps": int(env_steps_total),
               "fit_mean": round(float(f.mean()), 4), "fit_max": round(float(f.max()), 4)}

        if it % args.eval_every == 0 or it == 1 or it == args.iters:
            key, kv = jax.random.split(key)
            ev_keys = jax.random.split(kv, args.eval_envs)
            ev = eval_mean_j(theta, ev_keys)
            rec["eval_success"] = round(ev["success"], 4)
            rec["eval_reached"] = round(ev["reached"], 4)
            rec["eval_grasp"] = round(ev["grasp"], 4)
            rec["eval_lift_ok"] = round(ev["lift_ok"], 4)
            rec["eval_bt_max"] = round(ev["bt_max"], 4)
            if ev["success"] > best_success:
                best_success = ev["success"]
                best_theta = theta
                np.save(out_dir / "best_theta.npy", np.asarray(jax.device_get(theta)))
            print(f"[it {it:4d}] steps={env_steps_total:>10,} fit={f.mean():.3f}/{f.max():.3f} "
                  f"EVAL success={ev['success']:.4f} reached={ev['reached']:.4f} "
                  f"grasp={ev['grasp']:.4f} lift={ev['lift_ok']:.4f} bt={ev['bt_max']:.4f} "
                  f"(peak={best_success:.4f})", flush=True)
            hist.append(rec)
            write_progress(it, ev)
            write_results(status="running")
        else:
            hist.append(rec)
            if it % 10 == 0:
                print(f"[it {it:4d}] steps={env_steps_total:>10,} fit={f.mean():.3f}/{f.max():.3f}", flush=True)
                write_progress(it, None)

    # ----- final honest eval on a fresh, larger held-out set with BEST theta -----
    key, kf = jax.random.split(key)
    fin_keys = jax.random.split(kf, max(args.eval_envs, 512))
    final_eval = eval_mean_j(best_theta, fin_keys)
    final_eval = {k: float(v) for k, v in final_eval.items()}
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
