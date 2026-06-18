"""Minimal smoke test for the GHM agent — no mujoco env, no OGBench, just shapes.

Constructs agents/ghm.py on a tiny random batch and runs ONE flow-matching pretrain update.
Verifies the agent imports, builds, and the update runs without shape errors.

Run on a GPU box (or any box with jax installed) from infom/:
    python smoke_ghm.py
Expected: prints "GHM SMOKE OK" plus a couple of loss scalars.

NOTE: This intentionally does NOT call eval / make_env (those need mujoco + a GPU).
"""
import numpy as np

from agents.ghm import GHMAgent, get_config


def main():
    rng = np.random.default_rng(0)
    B, obs_dim, act_dim = 8, 11, 3

    config = get_config()
    config['batch_size'] = B
    # Enable real horizon-conditioning + discount-consistency subset so the new code paths run.
    config['gamma_min'] = 0.95
    config['gamma_max'] = 0.999
    config['discount_consistency_frac'] = 0.5
    config['horizon_fourier_dim'] = 16  # exercise the Fourier embed path too
    config['encoder'] = None

    ex_observations = rng.standard_normal((B, obs_dim)).astype(np.float32)
    ex_actions = rng.standard_normal((B, act_dim)).astype(np.float32)

    agent = GHMAgent.create(0, ex_observations, ex_actions, config)
    print('agent constructed')

    def make_batch():
        return {
            'observations': rng.standard_normal((B, obs_dim)).astype(np.float32),
            'actions': np.clip(rng.standard_normal((B, act_dim)), -1, 1).astype(np.float32),
            'next_observations': rng.standard_normal((B, obs_dim)).astype(np.float32),
            'next_actions': np.clip(rng.standard_normal((B, act_dim)), -1, 1).astype(np.float32),
            'rewards': rng.standard_normal((B,)).astype(np.float32),
            'observation_min': np.full((obs_dim,), -10.0, np.float32),
            'observation_max': np.full((obs_dim,), 10.0, np.float32),
        }

    # One pretrain (flow-matching + BC) update.
    agent, info = agent.pretrain(make_batch())
    fm = float(info['flow_occupancy/flow_matching_loss'])
    gh = float(info['flow_occupancy/gamma_h_mean'])
    print(f'pretrain update ok: flow_matching_loss={fm:.4f} gamma_h_mean={gh:.4f}')

    # Policy-conditioning hook: supply cond_actions/cond_next_actions (e.g. a base policy's actions).
    pbatch = make_batch()
    pbatch['cond_actions'] = np.clip(rng.standard_normal((B, act_dim)), -1, 1).astype(np.float32)
    pbatch['cond_next_actions'] = np.clip(rng.standard_normal((B, act_dim)), -1, 1).astype(np.float32)
    agent, info = agent.pretrain(pbatch)
    print('policy-conditioning hook update ok')

    # One finetune update (reward + critic + flow + actor) — exercises critic_loss horizon path.
    agent, info = agent.finetune(make_batch(), full_update=True)
    print(f"finetune update ok: critic_loss={float(info['critic/critic_loss']):.4f}")

    print('GHM SMOKE OK')


if __name__ == '__main__':
    main()
