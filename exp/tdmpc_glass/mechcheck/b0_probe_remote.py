"""
B0 OOD value-decodability probe.

Loads a trained PPO policy (ManiSkill PPO checkpoint, state obs), rolls out
episodes at the TRAIN object count and a HELD-OUT object count, then fits a
linear ridge probe predicting per-step return-to-go (discounted) from the flat
state observation. Reports R^2 (train-count vs held-out-count) under
train/test split. The pre-registered headroom signal is: R^2 COLLAPSES at the
held-out object count relative to the train count.

Usage:
  python b0_probe_remote.py --env STACK_ENV --train_n 2 --ood_n 4 \
      --ckpt /root/coods_b0/ppo_ckpt.pt --num_envs 64 --episodes 64
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs  # noqa: register envs
import b0_pickcube_multi  # noqa: register PickCubeMulti-v1
import b0_pushcube_multi  # noqa: register PushCubeMulti-v1
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


# ---- PPO agent definition must match the training script's Agent ----
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_action(self, x, deterministic=True):
        mean = self.actor_mean(x)
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd)
        return mean + torch.randn_like(mean) * std


def rollout(env_id, num_obj_kw, n_obj, ckpt_path, num_envs, episodes, gamma, device, random_policy):
    base = gym.make(env_id, num_envs=num_envs, num_cubes=n_obj, obs_mode="state",
                    control_mode="pd_joint_delta_pos", sim_backend="gpu",
                    reward_mode="normalized_dense")
    env = ManiSkillVectorEnv(base, num_envs, ignore_terminations=True, auto_reset=True)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    max_steps = base.spec.max_episode_steps or 50

    agent = None
    if not random_policy:
        agent = Agent(obs_dim, act_dim).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(sd)
        agent.eval()

    all_obs, all_rtg, ep_returns = [], [], []
    n_iters = max(1, episodes // num_envs)
    for _ in range(n_iters):
        obs, _ = env.reset(seed=999)
        ep_obs, ep_rew = [], []
        for t in range(max_steps):
            ep_obs.append(obs.cpu().numpy())
            with torch.no_grad():
                if random_policy:
                    act = torch.rand((num_envs, act_dim), device=device) * 2 - 1
                else:
                    act = agent.get_action(obs.to(device), deterministic=True)
            obs, rew, term, trunc, info = env.step(act)
            ep_rew.append(rew.cpu().numpy())
        ep_obs = np.stack(ep_obs, axis=0)   # (T, E, D)
        ep_rew = np.stack(ep_rew, axis=0)   # (T, E)
        T, E = ep_rew.shape
        rtg = np.zeros_like(ep_rew)
        run = np.zeros(E)
        for t in reversed(range(T)):
            run = ep_rew[t] + gamma * run
            rtg[t] = run
        all_obs.append(ep_obs.reshape(T * E, -1))
        all_rtg.append(rtg.reshape(T * E))
        ep_returns.append(ep_rew.sum(axis=0))
    env.close()
    X = np.concatenate(all_obs, axis=0)
    y = np.concatenate(all_rtg, axis=0)
    return X, y, np.concatenate(ep_returns), obs_dim


def ridge_r2(X, y, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    ntr = int(0.8 * n)
    tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    ym = y[tr].mean()
    ytr = y[tr] - ym
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w + ym
    ss_res = ((y[te] - pred) ** 2).sum()
    ss_tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--num_obj_kw", default=None)
    ap.add_argument("--train_n", type=int, required=True)
    ap.add_argument("--ood_n", type=int, required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--out", default="/root/coods_b0/probe_result.json")
    args = ap.parse_args()
    device = "cuda"
    random_policy = args.ckpt is None

    res = {}
    for tag, n in [("train_count", args.train_n), ("ood_count", args.ood_n)]:
        X, y, ep_ret, obs_dim = rollout(
            args.env, args.num_obj_kw, n, args.ckpt,
            args.num_envs, args.episodes, args.gamma, device, random_policy)
        r2 = float(np.mean([ridge_r2(X, y, seed=s) for s in range(3)]))
        res[tag] = {
            "n_obj": n, "obs_dim": int(obs_dim), "n_samples": int(X.shape[0]),
            "probe_r2": r2, "mean_ep_return": float(ep_ret.mean()),
        }
        print(tag, "n=", n, "obs_dim=", obs_dim, "r2=", round(r2, 4),
              "mean_ret=", round(float(ep_ret.mean()), 3), flush=True)

    res["r2_collapse"] = res["train_count"]["probe_r2"] - res["ood_count"]["probe_r2"]
    res["random_policy"] = random_policy
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
