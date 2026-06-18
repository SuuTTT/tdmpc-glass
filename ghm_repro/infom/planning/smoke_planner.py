"""Smoke test for the CompPlan plan-over-policies planner.

Builds a GHM agent (restored from a checkpoint if one is found, else random-init), constructs the
CompPlanPlanner with the agent's own actor as the first base policy, and runs ONE planning step on a
random observation. Reports ran-clean or the exact error. CPU is fine (no env required).

Usage (on a box, sharing a GPU with a mem cap, or CPU):
    cd /root/ghm/infom && . /root/ghm/.venv/bin/activate
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
        python planning/smoke_planner.py --obs_dim 29 --action_dim 8 \
        [--restore_dir /root/ghm/exp/ghm_a8_t4_s1/sd001_20260618_132415 --restore_epoch 1500000]
"""
import argparse
import glob
import os
import sys
import traceback

import numpy as np

# Allow running as `python planning/smoke_planner.py` from infom/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from agents.ghm import GHMAgent, get_config  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402
from planning.compplan_planner import CompPlanPlanner, actor_base_policy  # noqa: E402


def find_checkpoint():
    """Return (restore_dir, epoch) for any params_*.pkl under /root/ghm/exp, or (None, None)."""
    for root in ['/root/ghm/exp', '/root/ghm/exp_test']:
        hits = sorted(glob.glob(os.path.join(root, '*', '*', 'params_*.pkl')))
        if hits:
            path = hits[-1]
            d = os.path.dirname(path)
            epoch = int(os.path.basename(path).split('_')[1].split('.')[0])
            return d, epoch
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obs_dim', type=int, default=29)
    ap.add_argument('--action_dim', type=int, default=8)
    ap.add_argument('--restore_dir', type=str, default=None)
    ap.add_argument('--restore_epoch', type=int, default=None)
    ap.add_argument('--auto_find_ckpt', type=int, default=1)
    args = ap.parse_args()

    config = get_config()

    rng = np.random.default_rng(0)
    ex_obs = rng.standard_normal((4, args.obs_dim)).astype(np.float32)
    ex_act = rng.standard_normal((4, args.action_dim)).astype(np.float32)

    print(f'[smoke] building GHM agent: obs_dim={args.obs_dim} action_dim={args.action_dim}')
    agent = GHMAgent.create(0, jnp.asarray(ex_obs), jnp.asarray(ex_act), config)

    restore_dir, restore_epoch = args.restore_dir, args.restore_epoch
    if restore_dir is None and args.auto_find_ckpt:
        restore_dir, restore_epoch = find_checkpoint()
    if restore_dir is not None:
        try:
            agent = restore_agent(agent, restore_dir, restore_epoch)
            print(f'[smoke] restored checkpoint from {restore_dir} @ {restore_epoch}')
        except Exception as e:  # noqa: BLE001
            print(f'[smoke] WARN: restore failed ({e!r}); continuing with random-init agent')
    else:
        print('[smoke] no checkpoint found; using random-init agent')

    # Build planner: agent's own actor is the first (and only) base policy.
    base_policies = [actor_base_policy(agent, temperature=0.0)]
    planner = CompPlanPlanner(
        agent=agent,
        base_policies=base_policies,
        gamma_h=None,            # -> agent's gamma_max
        plan_depth=3,
        beam_width=8,
        num_subgoal_samples=64,
        scorer='l2',
        latent_type='prior',
    )

    obs = rng.standard_normal((args.obs_dim,)).astype(np.float32)
    goal = rng.standard_normal((args.obs_dim,)).astype(np.float32)

    print('[smoke] running ONE planning step (plan_depth=3, beam=8, M=64) ...')
    out = planner.plan(obs, goal, seed=0)

    # Validate shapes.
    assert out['action'].shape == (args.action_dim,), f"action shape {out['action'].shape}"
    assert out['predicted_terminal_state'].shape == (args.obs_dim,)
    assert len(out['policy_sequence']) == planner.plan_depth
    print('[smoke] OK ran-clean.')
    print(f"[smoke]   action            = {np.round(out['action'], 3)}")
    print(f"[smoke]   policy_index      = {out['policy_index']}")
    print(f"[smoke]   policy_sequence   = {out['policy_sequence']}")
    print(f"[smoke]   plan score        = {out['score']:.4f}")
    print(f"[smoke]   pred terminal[:5] = {np.round(out['predicted_terminal_state'][:5], 3)}")


if __name__ == '__main__':
    try:
        main()
    except Exception:  # noqa: BLE001
        print('[smoke] FAILED with exception:')
        traceback.print_exc()
        sys.exit(1)
