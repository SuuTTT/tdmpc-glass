"""
Compact CleanRL-style PPO for ManiSkill GPU-vectorized state envs.
Self-contained (no examples/baselines dependency). Trains on PickCubeMulti-v1
at a given num_cubes, evaluates vs a random baseline, saves an Agent checkpoint
whose architecture matches b0_probe_remote.py's Agent.
"""
import argparse, json, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import mani_skill.envs  # noqa
import b0_pickcube_multi  # noqa register PickCubeMulti-v1
import b0_pushcube_multi  # noqa register PushCubeMulti-v1
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper  # noqa
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(1)
        ent = dist.entropy().sum(1)
        return action, logp, ent, self.critic(x)

    def get_action(self, x, deterministic=True):
        mean = self.actor_mean(x)
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd)
        return mean + torch.randn_like(mean) * std


def make_env(env_id, num_envs, num_cubes):
    extra = {"num_cubes": num_cubes} if "Multi" in env_id else {}
    env = gym.make(env_id, num_envs=num_envs,
                   obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="gpu", reward_mode="normalized_dense", **extra)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=False, auto_reset=True)
    return env


@torch.no_grad()
def evaluate(agent, env, device, n_steps, random_policy=False):
    # Episode is a success if `success` is true at ANY step before it ends.
    obs, _ = env.reset(seed=12345)
    N = env.num_envs
    adim = env.single_action_space.shape[0]
    succ_results, ret_results = [], []
    ever = torch.zeros(N, dtype=torch.bool, device=device)
    rsum = torch.zeros(N, device=device)
    for t in range(n_steps):
        if random_policy:
            act = (torch.rand((N, adim), device=device) * 2 - 1)
        else:
            act = agent.get_action(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(torch.clamp(act, -1.0, 1.0))
        rsum += rew
        if "success" in info:
            ever = ever | info["success"]
        done = torch.logical_or(term, trunc)
        if done.any():
            succ_results.append(ever[done].float())
            ret_results.append(rsum[done])
            ever = ever.clone(); ever[done] = False
            rsum = rsum.clone(); rsum[done] = 0
    succ = torch.cat(succ_results).mean().item() if succ_results else 0.0
    ret = torch.cat(ret_results).mean().item() if ret_results else float(rsum.mean())
    return succ, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PickCubeMulti-v1")
    ap.add_argument("--num_cubes", type=int, default=1)
    ap.add_argument("--num_envs", type=int, default=512)
    ap.add_argument("--total_steps", type=int, default=2_000_000)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--num_minibatches", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--gae_lambda", type=float, default=0.9)
    ap.add_argument("--target_kl", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--ckpt", default="/root/coods_b0/ppo_ckpt.pt")
    ap.add_argument("--log", default="/root/coods_b0/ppo_train.log")
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)

    env = make_env(args.env, args.num_envs, args.num_cubes)
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    agent = Agent(obs_dim, act_dim).to(device)
    opt = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # random baseline first
    rand_succ, rand_ret = evaluate(agent, env, device, 200, random_policy=True)

    N = args.num_envs
    T = args.num_steps
    batch = N * T
    mb = batch // args.num_minibatches
    obs_buf = torch.zeros((T, N, obs_dim), device=device)
    act_buf = torch.zeros((T, N, act_dim), device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    term_buf = torch.zeros((T, N), device=device)   # real termination (success)
    done_buf = torch.zeros((T, N), device=device)   # term OR trunc (episode boundary)
    val_buf = torch.zeros((T, N), device=device)
    nextval_buf = torch.zeros((T, N), device=device)  # bootstrap value of s_{t+1}

    obs, _ = env.reset(seed=0)
    n_updates = args.total_steps // batch
    t0 = time.time()
    logf = open(args.log, "w")
    best_succ = 0.0
    for upd in range(1, n_updates + 1):
        # linear LR anneal (matches reference ppo.py)
        frac = 1.0 - (upd - 1) / n_updates
        for g in opt.param_groups:
            g["lr"] = frac * args.lr
        for step in range(T):
            obs_buf[step] = obs
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(obs)
            val_buf[step] = value.flatten()
            act_buf[step] = action
            logp_buf[step] = logp
            obs, rew, term, trunc, info = env.step(torch.clamp(action, -1.0, 1.0))
            rew_buf[step] = rew
            term_buf[step] = term.float()
            done = torch.logical_or(term, trunc)
            done_buf[step] = done.float()
            # bootstrap value of the next state. On truncation, auto_reset has
            # replaced obs with the reset obs, so use final_observation for the
            # truncated envs. Zero the bootstrap only on REAL termination.
            with torch.no_grad():
                nv = agent.get_value(obs).flatten()
                if done.any() and "final_observation" in info:
                    fo = info["final_observation"]
                    nv_final = agent.get_value(fo).flatten()
                    trunc_only = (trunc & ~term)
                    nv = torch.where(trunc_only, nv_final, nv)
            nv = nv * (1.0 - term.float())
            nextval_buf[step] = nv
        with torch.no_grad():
            adv = torch.zeros_like(rew_buf)
            lastgae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                delta = rew_buf[t] + args.gamma * nextval_buf[t] - val_buf[t]
                # reset GAE accumulation at any episode boundary (term or trunc)
                cont = 1.0 - done_buf[t]
                lastgae = delta + args.gamma * args.gae_lambda * cont * lastgae
                adv[t] = lastgae
            ret = adv + val_buf
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)
        idx = np.arange(batch)
        stop = False
        for _ in range(args.update_epochs):
            if stop:
                break
            np.random.shuffle(idx)
            for start in range(0, batch, mb):
                mbi = idx[start:start + mb]
                _, newlogp, ent, newval = agent.get_action_and_value(b_obs[mbi], b_act[mbi])
                logratio = newlogp - b_logp[mbi]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                a = b_adv[mbi]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg1 = -a * ratio
                pg2 = -a * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                pg_loss = torch.max(pg1, pg2).mean()
                newval = newval.flatten()
                v_loss = 0.5 * ((newval - b_ret[mbi]) ** 2).mean()
                ent_loss = ent.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop = True
                    break
        if upd % 5 == 0 or upd == n_updates:
            succ, eret = evaluate(agent, env, device, 100)
            gstep = upd * batch
            sps = gstep / (time.time() - t0)
            msg = f"upd={upd} step={gstep} eval_succ={succ:.3f} eval_ret={eret:.2f} SPS={sps:.0f}"
            print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
            if succ >= best_succ:
                best_succ = succ
                torch.save(agent.state_dict(), args.ckpt)
    # final eval
    succ, eret = evaluate(agent, env, device, 200)
    torch.save(agent.state_dict(), args.ckpt)
    result = {
        "env": args.env, "num_cubes": args.num_cubes, "obs_dim": obs_dim,
        "act_dim": act_dim, "total_steps": args.total_steps,
        "random_eval_success": rand_succ, "random_eval_return": rand_ret,
        "ppo_final_eval_success": succ, "ppo_final_eval_return": eret,
        "ppo_best_eval_success": best_succ,
    }
    with open("/root/coods_b0/ppo_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("PPO_RESULT", json.dumps(result), flush=True)
    logf.close()
    env.close()


if __name__ == "__main__":
    main()
