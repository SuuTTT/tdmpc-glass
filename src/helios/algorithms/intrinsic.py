"""iter-21 — abstraction-grounded exploration for TD-MPC2 on SPARSE tasks.

The deep-research consensus (docs/research/dr-synthesis-iter21.md): abstraction/skills beat
flat model-based RL only in the SPARSE / long-horizon / exploration-limited regime (where the
flat baseline scores ~0), not on dense control. So iter-21 tests EXPLORATION bonuses on sparse
MJX tasks, as host-side intrinsic rewards added to the training reward (eval reward untouched):

- RND (Random Network Distillation): the simple, proven baseline. Novelty = predictor error
  against a fixed random target. Lower-risk control — does ANY exploration rescue sparse TD-MPC2?
- LAPLACIAN eigenpurpose (DCEO-style): the abstraction bet. Learn the graph-Laplacian
  representation phi(s) (eigenfunctions: smooth over transitions, decorrelated); reward
  ||phi(s')-phi(s)|| = movement along slow manifold directions / crossing bottlenecks. This is
  the RIGHT use of the transition-graph abstraction (eigen-directions as exploration), not the
  iter-19 reach-centroid subgoals that failed. No continuous-control or world-model precedent
  for Laplacian options -> genuine gap.

Both: framework = a small flax net + optax, updated online on collection transitions; intrinsic
reward normalized by running std so the coefficient is task-agnostic. Operate on RAW obs
(encoder-independent). jax-only (runs on the worker, not the control plane).
"""
from __future__ import annotations
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np


class _MLP(nn.Module):
    hidden: tuple
    out: int

    @nn.compact
    def __call__(self, x):
        for d in self.hidden:
            x = nn.relu(nn.Dense(d)(x))
        return nn.Dense(self.out)(x)


class RunningNorm:
    """Host-side running mean/std for obs (RND is scale-sensitive) and for the
    intrinsic reward (so coef is task-agnostic)."""
    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):  # x: (N, ...)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + d * d * self.count * bc / tot) / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


def make_rnd(obs_dim, D=64, hidden=(256, 256), lr=1e-4, seed=0):
    tgt = _MLP(hidden, D)
    prd = _MLP(hidden, D)
    kt, kp = jax.random.split(jax.random.PRNGKey(seed))
    tparams = tgt.init(kt, jnp.zeros((1, obs_dim)))
    pparams = prd.init(kp, jnp.zeros((1, obs_dim)))
    tx = optax.adam(lr)
    opt = tx.init(pparams)

    @jax.jit
    def reward(pp, obs):  # obs (N, obs_dim) normalized; returns (N,) raw novelty
        t = jax.lax.stop_gradient(tgt.apply(tparams, obs))
        p = prd.apply(pp, obs)
        return jnp.mean((p - t) ** 2, axis=-1)

    @jax.jit
    def update(pp, opt, obs):
        def loss(pp):
            t = jax.lax.stop_gradient(tgt.apply(tparams, obs))
            return jnp.mean((prd.apply(pp, obs) - t) ** 2)
        g = jax.grad(loss)(pp)
        u, opt = tx.update(g, opt, pp)
        return optax.apply_updates(pp, u), opt

    return {"pp": pparams, "opt": opt, "reward": reward, "update": update,
            "onorm": RunningNorm((obs_dim,)), "rnorm": RunningNorm(())}


def make_laplacian(obs_dim, D=10, hidden=(256, 256), lr=1e-4, ortho_coef=1.0, seed=0):
    net = _MLP(hidden, D)
    pparams = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, obs_dim)))
    tx = optax.adam(lr)
    opt = tx.init(pparams)

    @jax.jit
    def phi(pp, obs):
        return net.apply(pp, obs)  # (N, D)

    @jax.jit
    def reward(pp, obs, obs_next):  # ||phi(s')-phi(s)||_1 = manifold movement (N,)
        f = net.apply(pp, obs); fn = net.apply(pp, obs_next)
        return jnp.sum(jnp.abs(fn - f), axis=-1)

    @jax.jit
    def update(pp, opt, obs, obs_next):
        def loss(pp):
            f = net.apply(pp, obs); fn = net.apply(pp, obs_next)
            smooth = jnp.mean(jnp.sum((f - fn) ** 2, axis=-1))   # transitions -> close
            B = f.shape[0]
            G = (f.T @ f) / B
            ortho = jnp.mean((G - jnp.eye(D)) ** 2)              # E[ff^T] -> I (decorrelate)
            return smooth + ortho_coef * ortho
        g = jax.grad(loss)(pp)
        u, opt = tx.update(g, opt, pp)
        return optax.apply_updates(pp, u), opt

    return {"pp": pparams, "opt": opt, "reward": reward, "update": update,
            "onorm": RunningNorm((obs_dim,)), "rnorm": RunningNorm(())}
