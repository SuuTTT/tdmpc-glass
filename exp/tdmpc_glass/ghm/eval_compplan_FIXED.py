"""CompPlan OGBench rollout-eval loop (Stage 3) — base-policy ALONE vs base + CompPlan.

This is the eval harness that turns the CompPlan planner (planning/compplan_planner.py) into an
actual OGBench success measurement on antmaze-medium-navigate-singletask-task1-v0, comparing:

  * BASE-ALONE: a pre-trained goal-conditioned base policy (OGBench GC-BC, Stage 1) rolled out
    directly toward the *final* env goal each step.
  * BASE + COMPPLAN: at each step, use the trained Geometric Horizon Model (GHM, Stage 2) to JUMP
    to predicted future states under the base policy, pick the best subgoal sequence toward the
    final goal (beam search, scored by xy-distance), and feed the *first chosen subgoal* to the
    goal-conditioned base policy. Receding-horizon / MPC re-planning every `replan_every` steps.

Both arms are measured by OGBench `info['success']` over `--episodes` episodes (deterministic,
temperature 0). Numbers are read from the real env; nothing is fabricated.

NOTE on faithfulness (documented gap): the paper's CompPlan plans over a *library* of distinct
pre-trained policies. Here the "library" is ONE goal-conditioned GC-BC policy whose behavior is
steered by the planner-chosen subgoal — i.e. we instantiate plan-over-policies as plan-over-subgoals
for a single GC base policy (the standard subgoal-planning reduction). The GHM checkpoint used is
horizon-conditioned (gamma_min<gamma_max) but conditioned on its OWN actor's actions, not the GC-BC
policy's actions (the GHM's policy-conditioning hook `policy_embeds` is unused in this checkpoint).
These two gaps vs. the paper are stated in the final report.

Usage (on b3060, pin a GPU):
    cd /root/ghm/infom
    CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
      /root/ghm/.venv/bin/python planning/eval_compplan.py \
        --env_name antmaze-medium-navigate-singletask-task1-v0 \
        --ghm_dir '/root/ghm/exp/ghm_amz_t1/sd000_20260619_095353' --ghm_epoch 1500000 \
        --gcbc_dir '/root/ghm/gcrl_exp/OGBench/Debug/sd000_*' --gcbc_epoch 1000000 \
        --episodes 20 --arms both
"""
import argparse
import glob
import os
import sys
import traceback

import numpy as np

INFOM_DIR = '/root/ghm/infom'
IMPLS_DIR = '/root/ghm/ogbench_impls'


def _clear_pkgs(*names):
    for m in list(sys.modules):
        for n in names:
            if m == n or m.startswith(n + '.'):
                del sys.modules[m]
                break


def load_ghm_agent(ghm_dir, ghm_epoch, obs_dim, action_dim, agent_mod='ghm'):
    """Import infom's GHM agent, build it, restore the checkpoint. Returns a bound agent.

    MERGES the checkpoint's own flags.json agent config into the default get_config()
    (copied from probe2.py) so fourier/dim-changing variants (e.g. fourier=64 -> 643-dim
    field) build with the SAME shapes they were trained with — otherwise restore shape-
    mismatches. Also lets us point at alternate agent modules (e.g. ghm_polcond).
    """
    import importlib, json
    _clear_pkgs('utils', 'agents')
    sys.path.insert(0, INFOM_DIR)
    import jax.numpy as jnp
    mod = importlib.import_module('agents.' + agent_mod)
    from utils.flax_utils import restore_agent
    config = mod.get_config()
    fj = os.path.join(ghm_dir, 'flags.json')
    if os.path.exists(fj):
        saved = json.load(open(fj)).get('agent', {})
        merged = []
        for k, v in saved.items():
            try:
                if k in config:
                    config[k] = v
                    merged.append(f'{k}={v}')
            except Exception:
                pass
        print(f'[eval] merged GHM config from flags.json: {", ".join(merged)}')
    ex_obs = np.zeros((4, obs_dim), np.float32)
    ex_act = np.zeros((4, action_dim), np.float32)
    agent = mod.GHMAgent.create(0, jnp.asarray(ex_obs), jnp.asarray(ex_act), config)
    agent = restore_agent(agent, ghm_dir, ghm_epoch)
    print(f'[eval] GHM ({agent_mod}) restored from {ghm_dir} @ {ghm_epoch}')
    return agent


def load_gcbc_agent(gcbc_dir, gcbc_epoch, env, ex_obs, ex_act):
    """Import impls' GC-BC agent, build it, restore the checkpoint. Returns (agent, config)."""
    _clear_pkgs('utils', 'agents')
    sys.path.insert(0, IMPLS_DIR)
    from agents.gcbc import GCBCAgent, get_config
    from utils.flax_utils import restore_agent
    config = get_config()
    agent = GCBCAgent.create(0, ex_obs, ex_act, config)
    agent = restore_agent(agent, gcbc_dir, gcbc_epoch)
    print(f'[eval] GC-BC restored from {gcbc_dir} @ {gcbc_epoch}')
    return agent, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_name', default='antmaze-medium-navigate-singletask-task1-v0')
    ap.add_argument('--ghm_dir', required=True)
    ap.add_argument('--ghm_epoch', type=int, default=1500000)
    ap.add_argument('--gcbc_dir', required=True)
    ap.add_argument('--gcbc_epoch', type=int, default=1000000)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--arms', choices=['base', 'compplan', 'both'], default='both')
    # planner knobs
    ap.add_argument('--plan_depth', type=int, default=3)
    ap.add_argument('--beam_width', type=int, default=8)
    ap.add_argument('--num_subgoal_samples', type=int, default=64)
    ap.add_argument('--replan_every', type=int, default=10)
    ap.add_argument('--xy_dims', type=int, default=2, help='# leading obs dims used as the goal-scoring (xy) space')
    ap.add_argument('--ghm_agent_mod', default='ghm', help='agent module to build the GHM from (ghm | ghm_polcond | ...)')
    ap.add_argument('--ghm_norm_type', default=None,
                    help="override GHM obs_norm_type (none|normal|bounded); default = read from GHM flags.json")
    ap.add_argument('--plan_gamma', type=float, default=None,
                    help="horizon gamma_h to query the GHM at for subgoals (default = GHM gamma_max). "
                         "Smaller => NEARER, more-reachable subgoals.")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # ---- env (impls env_utils) -------------------------------------------------------------
    _clear_pkgs('utils', 'agents')
    sys.path.insert(0, IMPLS_DIR)
    from utils.env_utils import make_env_and_datasets
    env, train_ds, _ = make_env_and_datasets(args.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f'[eval] env={args.env_name} obs_dim={obs_dim} action_dim={action_dim}')

    # observation bounds (for GHM flow-goal clipping): use dataset min/max
    obs_all = np.asarray(train_ds['observations'])
    obs_min = obs_all.min(axis=0)
    obs_max = obs_all.max(axis=0)

    ex_obs = obs_all[:1]
    ex_act = np.asarray(train_ds['actions'])[:1]

    # ---- load base GC-BC policy ------------------------------------------------------------
    gcbc_dir = args.gcbc_dir
    gcbc_agent, gcbc_config = load_gcbc_agent(gcbc_dir, args.gcbc_epoch, env, ex_obs, ex_act)

    import jax  # after agents imported

    results = {}

    # ----------------------------------------------------------------- BASE-ALONE arm ------
    if args.arms in ('base', 'both'):
        # Seed the env so both arms see the SAME episode distribution (paired comparison).
        try:
            env.reset(seed=args.seed)
        except Exception:
            pass
        np.random.seed(args.seed)
        _clear_pkgs('utils', 'agents')
        sys.path.insert(0, IMPLS_DIR)
        from utils.evaluation import evaluate
        stats, _, _ = evaluate(
            agent=gcbc_agent, env=env, task_id=None, config=gcbc_config,
            num_eval_episodes=args.episodes, num_video_episodes=0, eval_temperature=0,
        )
        base_succ = float(stats.get('success', np.nan))
        results['base'] = dict(success=base_succ, all=stats)
        print(f'[eval] BASE-ALONE GC-BC success = {base_succ:.3f} over {args.episodes} eps')

    # ----------------------------------------------------------------- BASE+COMPPLAN arm ---
    if args.arms in ('compplan', 'both'):
        ghm_agent = load_ghm_agent(args.ghm_dir, args.ghm_epoch, obs_dim, action_dim,
                                   agent_mod=args.ghm_agent_mod)
        import jax.numpy as jnp

        # ---- NORMALIZATION AWARENESS (root-cause fix) -------------------------------------
        # The GHM is trained in its dataset's obs_norm_type frame (antmaze default 'normal' =>
        # xy ~ N(0,1)); GC-BC + the env live in RAW obs space. We must (1) normalize the env
        # observation/goal into the GHM frame before jumping, and (2) de-normalize the GHM's
        # subgoal back to raw before handing it to GC-BC / scoring vs the raw goal.
        import json as _json
        ghm_flags = {}
        _fj = os.path.join(args.ghm_dir, 'flags.json')
        if os.path.exists(_fj):
            ghm_flags = _json.load(open(_fj))
        ghm_norm_type = args.ghm_norm_type or ghm_flags.get('obs_norm_type', 'none')
        obs_mean = obs_all.mean(axis=0).astype(np.float32)
        obs_var = obs_all.var(axis=0).astype(np.float32)
        eps = 1e-8

        def to_ghm(x):
            x = np.asarray(x, np.float32)
            if ghm_norm_type == 'normal':
                return (x - obs_mean) / np.sqrt(obs_var + eps)
            if ghm_norm_type == 'bounded':
                return 2 * (x - obs_min) / (obs_max - obs_min) - 1.0
            return x

        def from_ghm(x):
            x = np.asarray(x, np.float32)
            if ghm_norm_type == 'normal':
                return x * np.sqrt(obs_var + eps) + obs_mean
            if ghm_norm_type == 'bounded':
                return (x + 1.0) / 2.0 * (obs_max - obs_min) + obs_min
            return x

        print(f'[eval] GHM obs_norm_type={ghm_norm_type} -> normalization-aware planning '
              f'(obs_mean[:2]={obs_mean[:2]}, sqrt(var)[:2]={np.sqrt(obs_var[:2]+eps)})')

        gamma_h = float(args.plan_gamma) if args.plan_gamma is not None else float(ghm_agent.config['gamma_max'])
        print(f'[eval] planning gamma_h={gamma_h} (GHM gamma_max={float(ghm_agent.config["gamma_max"])})')
        latent_dim = int(ghm_agent.config['latent_dim'])
        clip = bool(ghm_agent.config['clip_flow_goals'])
        # GHM was clip-trained with NORMALIZED bounds; build them in the GHM frame.
        obs_min_g = to_ghm(obs_min)
        obs_max_g = to_ghm(obs_max)
        obs_min_j = jnp.asarray(np.minimum(obs_min_g, obs_max_g))
        obs_max_j = jnp.asarray(np.maximum(obs_min_g, obs_max_g))
        M = int(args.num_subgoal_samples)

        # jit the GC actor (deterministic) and the GHM jump for speed (host-loop was the bottleneck).
        @jax.jit
        def gc_action_j(observation, goal, seed):
            a = gcbc_agent.sample_actions(observations=observation, goals=goal, seed=seed, temperature=0)
            return jnp.clip(a, -1, 1)

        def gc_action(observation, goal, seed):
            # observation/goal are RAW (env frame), as GC-BC was trained.
            return np.asarray(gc_action_j(jnp.asarray(observation), jnp.asarray(goal), seed))

        @jax.jit
        def ghm_jump_j(state_g, action, rng):
            # state_g is in the GHM (normalized) frame; returns flow goals in the GHM frame.
            noise_rng, lat_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (M, obs_dim), jnp.float32)
            obs_rep = jnp.broadcast_to(state_g[None], (M, obs_dim))
            act_rep = jnp.broadcast_to(action[None], (M, action_dim))
            latents = jax.random.normal(lat_rng, (M, latent_dim), jnp.float32)
            horizons = jnp.full((M,), gamma_h, jnp.float32)
            kw = {}
            if clip:
                kw['observation_min'] = obs_min_j
                kw['observation_max'] = obs_max_j
            return ghm_agent.compute_fwd_flow_goals(
                noises, obs_rep, act_rep, latents, horizons=horizons,
                use_target_network=False, **kw)

        def ghm_jump(state_raw, action, rng, M_unused=None):
            # Take a RAW state, normalize into GHM frame, jump, de-normalize back to RAW.
            state_g = to_ghm(state_raw)
            fut_g = np.asarray(ghm_jump_j(jnp.asarray(state_g, jnp.float32),
                                          jnp.asarray(action, jnp.float32), rng))
            return from_ghm(fut_g)

        xy = args.xy_dims

        def plan_subgoal(state, final_goal, rng):
            """Beam search over GHM jumps under the GC policy steered toward sampled subgoals.

            At each depth we expand each beam state by: (1) computing the base policy's action when
            aimed at the FINAL goal, (2) GHM-jumping to M future states, (3) scoring jumped states by
            xy-distance to the final goal. The first-depth jumped state that leads the best terminal
            sequence is returned as the SUBGOAL to feed the GC policy this control step.
            """
            beam = [(np.asarray(state), None, 0.0)]  # (state, first_subgoal, score)
            for _ in range(args.plan_depth):
                cands = []
                for (s, first_sg, _sc) in beam:
                    rng, ar, jr = jax.random.split(rng, 3)
                    a = gc_action(s, final_goal, ar)
                    a = np.clip(a, -1, 1)
                    fut = ghm_jump(s, a, jr, args.num_subgoal_samples)  # (M, obs)
                    d = np.linalg.norm(fut[:, :xy] - final_goal[:xy][None], axis=-1)
                    order = np.argsort(d)[: args.beam_width]
                    for m in order:
                        sg = first_sg if first_sg is not None else fut[m]
                        cands.append((fut[m], sg, -float(d[m])))
                cands.sort(key=lambda c: c[2], reverse=True)
                beam = cands[: args.beam_width]
            return beam[0][1]  # best first subgoal

        # --- SUBGOAL SANITY accumulators (TASK 4) ---
        # For each plan we record:
        #   closer_frac: subgoal xy is closer to the final goal than the current state
        #   d_cur, d_sg: xy-dist(state, goal) and xy-dist(subgoal, goal)
        #   jump_mag: xy displacement of subgoal from current state (how far the GHM jumps)
        sg_stats = dict(n=0, n_closer=0, sum_d_cur=0.0, sum_d_sg=0.0, sum_jump=0.0,
                        sum_progress=0.0)

        def record_subgoal(state, subgoal, final_goal):
            d_cur = float(np.linalg.norm(state[:xy] - final_goal[:xy]))
            d_sg = float(np.linalg.norm(subgoal[:xy] - final_goal[:xy]))
            jump = float(np.linalg.norm(subgoal[:xy] - state[:xy]))
            sg_stats['n'] += 1
            sg_stats['n_closer'] += int(d_sg < d_cur)
            sg_stats['sum_d_cur'] += d_cur
            sg_stats['sum_d_sg'] += d_sg
            sg_stats['sum_jump'] += jump
            sg_stats['sum_progress'] += (d_cur - d_sg)  # +ve = subgoal makes progress

        _clear_pkgs('utils', 'agents')
        sys.path.insert(0, IMPLS_DIR)
        successes = []
        rng = jax.random.PRNGKey(args.seed)
        # Re-seed env identically to the base arm for a paired comparison.
        try:
            env.reset(seed=args.seed)
        except Exception:
            pass
        np.random.seed(args.seed)
        from tqdm import trange
        for ep in trange(args.episodes, desc='compplan'):
            obs, info = env.reset(options=dict(task_id=None, render_goal=False))
            final_goal = np.asarray(info['goal'], np.float32)
            done = False
            step = 0
            subgoal = None
            ep_success = 0
            while not done:
                if step % args.replan_every == 0 or subgoal is None:
                    rng, pr = jax.random.split(rng)
                    subgoal = plan_subgoal(obs, final_goal, pr)
                    record_subgoal(np.asarray(obs, np.float32), np.asarray(subgoal, np.float32), final_goal)
                rng, ar = jax.random.split(rng)
                action = gc_action(obs, subgoal.astype(np.float32), ar)
                action = np.clip(np.asarray(action), -1, 1)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
                if float(info.get('success', 0.0)) > 0:
                    ep_success = 1
            successes.append(ep_success)
        cp_succ = float(np.mean(successes))
        n = max(sg_stats['n'], 1)
        subgoal_sanity = dict(
            n_plans=sg_stats['n'],
            frac_subgoal_closer=sg_stats['n_closer'] / n,
            mean_dist_cur_to_goal=sg_stats['sum_d_cur'] / n,
            mean_dist_subgoal_to_goal=sg_stats['sum_d_sg'] / n,
            mean_jump_magnitude=sg_stats['sum_jump'] / n,
            mean_progress_toward_goal=sg_stats['sum_progress'] / n,
        )
        results['compplan'] = dict(success=cp_succ, per_episode=successes,
                                   subgoal_sanity=subgoal_sanity)
        print(f'[eval] BASE+COMPPLAN success = {cp_succ:.3f} over {args.episodes} eps  '
              f'(per-ep {successes})')
        print('[eval] SUBGOAL SANITY: '
              f"frac_closer={subgoal_sanity['frac_subgoal_closer']:.2f} "
              f"d(cur->goal)={subgoal_sanity['mean_dist_cur_to_goal']:.2f} "
              f"d(sg->goal)={subgoal_sanity['mean_dist_subgoal_to_goal']:.2f} "
              f"jump={subgoal_sanity['mean_jump_magnitude']:.2f} "
              f"progress={subgoal_sanity['mean_progress_toward_goal']:.2f} "
              f"(n={subgoal_sanity['n_plans']})")

    # ---- report --------------------------------------------------------------------------
    print('\n==================== CompPlan eval summary ====================')
    print(f'env: {args.env_name}   episodes: {args.episodes}')
    if 'base' in results:
        print(f"  base-alone (GC-BC):    success = {results['base']['success']:.3f}")
    if 'compplan' in results:
        print(f"  base + CompPlan:       success = {results['compplan']['success']:.3f}")
    print('===============================================================')

    if args.out:
        import json
        with open(args.out, 'w') as f:
            json.dump({k: {kk: (vv if not isinstance(vv, dict) else
                                {x: float(y) for x, y in vv.items()})
                           for kk, vv in v.items()
                           if kk in ('success', 'per_episode', 'subgoal_sanity')}
                       for k, v in results.items()},
                      f, indent=2, default=float)
        print(f'[eval] wrote {args.out}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('[eval] FAILED with exception:')
        traceback.print_exc()
        sys.exit(1)
