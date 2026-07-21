"""
B0 OOD value-decodability probe (uses official ManiSkill PPO Agent).

For each object count N we generate rollouts and fit a linear ridge probe that
predicts discounted return-to-go (the value target) from the FLAT state obs.
We report cross-validated R^2 at the TRAIN count and a HELD-OUT count.

Pre-registered headroom signal: linear value-decodability R^2 COLLAPSES at the
held-out object count relative to the train count.

Policy for rollouts: the N=2-trained official Agent is applied on the shared
leading obs dims (the target-cube + tcp + goal dims are first; distractor dims
are appended). For object counts whose obs_dim differs from the policy's input
dim, we feed the policy the first `policy_obs_dim` dims (target-relevant) and act;
distractor dims still vary the *state* the probe must decode from. This keeps the
behavior comparable across N while letting the obs composition change.
We ALSO report the random-policy version as a control.
"""
import argparse, json
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mani_skill.envs  # noqa
import b0_pushcube_multi  # noqa
import b0_pickcube_multi  # noqa
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Matches official ManiSkill ppo.py Agent (3 hidden layers, 256)."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def act(self, x):
        return self.actor_mean(x)

    def value(self, x):
        return self.critic(x).flatten()


def rollout(env_id, n_obj, agent, policy_obs_dim, num_envs, episodes, gamma, device,
            random_policy):
    base = gym.make(env_id, num_envs=num_envs, num_cubes=n_obj, obs_mode="state",
                    control_mode="pd_joint_delta_pos", sim_backend="physx_cuda",
                    reward_mode="normalized_dense")
    env = ManiSkillVectorEnv(base, num_envs, ignore_terminations=True, auto_reset=True)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    max_steps = 50
    all_obs, all_rtg, ep_ret, succ = [], [], [], []
    n_iters = max(1, episodes // num_envs)
    for _ in range(n_iters):
        obs, _ = env.reset(seed=np.random.randint(1 << 30))
        ep_obs, ep_rew = [], []
        ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for t in range(max_steps):
            ep_obs.append(obs.cpu().numpy())
            with torch.no_grad():
                if random_policy or agent is None:
                    act = torch.rand((num_envs, act_dim), device=device) * 2 - 1
                else:
                    # feed the policy its trained obs dims (leading, target-relevant)
                    pin = obs[:, :policy_obs_dim]
                    if policy_obs_dim > obs.shape[1]:
                        pad = torch.zeros((num_envs, policy_obs_dim - obs.shape[1]),
                                          device=device)
                        pin = torch.cat([obs, pad], dim=1)
                    act = torch.clamp(agent.act(pin), -1, 1)
            obs, rew, term, trunc, info = env.step(act)
            ep_rew.append(rew.cpu().numpy())
            if "success" in info:
                ever = ever | info["success"]
        ep_obs = np.stack(ep_obs, 0)   # (T,E,D)
        ep_rew = np.stack(ep_rew, 0)   # (T,E)
        T, E = ep_rew.shape
        rtg = np.zeros_like(ep_rew)
        run = np.zeros(E)
        for t in reversed(range(T)):
            run = ep_rew[t] + gamma * run
            rtg[t] = run
        all_obs.append(ep_obs.reshape(T * E, -1))
        all_rtg.append(rtg.reshape(T * E))
        ep_ret.append(ep_rew.sum(0))
        succ.append(ever.float().cpu().numpy())
    env.close()
    return (np.concatenate(all_obs, 0), np.concatenate(all_rtg, 0),
            np.concatenate(ep_ret), np.concatenate(succ), obs_dim, act_dim)


def ridge_r2(X, y, alpha=10.0, seeds=(0, 1, 2)):
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        idx = rng.permutation(n)
        ntr = int(0.8 * n)
        tr, te = idx[:ntr], idx[ntr:]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        ym = y[tr].mean(); ytr = y[tr] - ym
        d = Xtr.shape[1]
        A = Xtr.T @ Xtr + alpha * np.eye(d)
        w = np.linalg.solve(A, Xtr.T @ ytr)
        pred = Xte @ w + ym
        ss_res = ((y[te] - pred) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
        out.append(1.0 - ss_res / ss_tot)
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushCubeMulti-v1")
    ap.add_argument("--train_n", type=int, default=2)
    ap.add_argument("--ood_n", type=int, default=4)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--policy_obs_dim", type=int, default=52)  # N=2 PushCubeMulti
    ap.add_argument("--policy_act_dim", type=int, default=8)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--out", default="/root/coods_b0/probe_result.json")
    args = ap.parse_args()
    device = "cuda"
    np.random.seed(0); torch.manual_seed(0)

    agent = None
    if args.ckpt and not args.random:
        agent = Agent(args.policy_obs_dim, args.policy_act_dim).to(device)
        sd = torch.load(args.ckpt, map_location=device)
        agent.load_state_dict(sd)
        agent.eval()

    res = {"env": args.env, "gamma": args.gamma, "random_policy": bool(args.random or agent is None)}
    for tag, n in [("train_count", args.train_n), ("ood_count", args.ood_n)]:
        X, y, ep_ret, succ, obs_dim, act_dim = rollout(
            args.env, n, agent, args.policy_obs_dim, args.num_envs,
            args.episodes, args.gamma, device, args.random)
        r2_mean, r2_std = ridge_r2(X, y)
        res[tag] = {
            "n_obj": int(n), "obs_dim": int(obs_dim), "n_samples": int(X.shape[0]),
            "value_decode_r2": r2_mean, "value_decode_r2_std": r2_std,
            "mean_ep_return": float(ep_ret.mean()),
            "success_once": float(succ.mean()),
        }
        print(f"{tag}: N={n} obs_dim={obs_dim} r2={r2_mean:.4f}+/-{r2_std:.4f} "
              f"ret={ep_ret.mean():.2f} succ={succ.mean():.3f}", flush=True)

    res["r2_collapse"] = res["train_count"]["value_decode_r2"] - res["ood_count"]["value_decode_r2"]
    res["r2_collapse_frac"] = (res["r2_collapse"] /
                               (abs(res["train_count"]["value_decode_r2"]) + 1e-9))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
