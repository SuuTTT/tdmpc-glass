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


# ── iter-24 — SI2E-style SE-driven exploration ────────────────────────────────
# Pointing the VALIDATED structural-entropy structure (pre-check 1: 53% SE gap in TD-MPC2 latents)
# at the task it actually fits: COVERAGE/exploration, not adaptive-k (which died: jumpy error uniform).
# After SI2E (Zeng et al., NeurIPS 2024, arXiv:2410.06621): a dynamics-relevant embedding, an
# SE-OPTIMAL community partition of its transition graph, and a value-conditional coverage bonus that
# rewards visiting under-covered communities (anti-redundancy). v1 = SE-community count-coverage *
# value-difference weight. Pre-registered gate: must BEAT RND on sparse tasks (the iter-21 G2 bar the
# geometric Laplacian failed). networkx-optional: SE-optimal Louvain if present, else numpy kmeans.

def _np_kmeans(X, n, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), min(n, len(X)), replace=False)].copy()
    for _ in range(iters):
        d2 = (X * X).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (X @ C.T)
        lab = d2.argmin(1)
        for c in range(len(C)):
            m = lab == c
            if m.any():
                C[c] = X[m].mean(0)
    return C


def _se_communities(emb, n_comm, resolution=0.5, knn=8, seed=0):
    """SE-aligned community centroids over a kNN graph of embeddings. Louvain (modularity, which a
    kNN graph makes SE-aligned) when networkx is available AND it yields a sensible granularity
    (3..64 communities); otherwise kmeans(n_comm). NOTE: do NOT extra-sparsify a kNN graph — it is
    already sparse, and keep_frac fragmentation produced ~all-singletons (useless coverage codebook)."""
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        Xn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        S = Xn @ Xn.T
        np.fill_diagonal(S, 0.0)
        A = np.zeros_like(S)
        for i in range(S.shape[0]):
            nn_i = np.argsort(-S[i])[:knn]
            A[i, nn_i] = np.clip(S[i, nn_i], 0.0, None)
        A = np.maximum(A, A.T)                         # symmetric kNN (union), stays sparse
        G = nx.from_numpy_array(A)
        comms = [c for c in louvain_communities(G, weight="weight", resolution=resolution, seed=seed) if c]
        if 3 <= len(comms) <= 64:
            return np.stack([emb[list(c)].mean(0) for c in comms])
    except Exception:
        pass
    return _np_kmeans(emb, n_comm, seed=seed)


def make_random_encoder(obs_dim, D=32, seed=0):
    """Fixed random projection obs -> feature (the SI2E/VCSE 'random_encoder'; no training).
    numpy, encoder-independent. Returns a callable (N,obs_dim)->(N,D)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((obs_dim, D)).astype(np.float32) / np.sqrt(obs_dim)
    b = rng.standard_normal((D,)).astype(np.float32) * 0.1
    def enc(obs):
        return np.tanh(np.asarray(obs, np.float32) @ W + b)
    return enc


def make_se_explore(feat_dim, use_cluster=True, k_nn=12, value_beta=0.5,
                    buf_cap=2048, n_comm=24, rebuild_every=2000, seed=0):
    """FAITHFUL VCSE / SI2E (Seo VCSE + Zeng SI2E, NeurIPS'24, fast kmeans variant), FEATURE-AGNOSTIC:
    the caller supplies features phi(s) (random-encoder for vcse/si2e, or the WORLD-MODEL latent for the
    novel wmsi2e) and the critic value V(s).

      joint x = [ normalize(phi(s)) ; value_beta * normalize(V(s)) ]      (value-conditioning = concat V)
      r0 = 0.5 * log( ||x - x_kNN||^2 + eps )           # value-conditional kNN state entropy (VCSE)
      r1 = leaf r0 evaluated at the state's CLUSTER CENTROID (kmeans, fast variant)   # SI2E group term
      bonus = r0                  (vcse)
            = r0 - r1             (si2e / wmsi2e)        # H(V0) - H(V1), the encoding-tree differential

    All host-side numpy (kNN over a small ring buffer); rewards normalized downstream by RunningNorm.
    Ablation ladder: rnd < vcse (value) < si2e (value+cluster, random feat) < wmsi2e (value+cluster, WM latent)."""
    Dj = feat_dim + 1
    st = {
        "rnorm": RunningNorm(()), "fnorm": RunningNorm((feat_dim,)), "vnorm": RunningNorm(()),
        "bufF": np.zeros((buf_cap, feat_dim), np.float32), "bufV": np.zeros((buf_cap,), np.float32),
        "n": 0, "i": 0, "since": 0, "centroids": None,
        "use_cluster": use_cluster, "k": k_nn, "vb": value_beta,
        "cap": buf_cap, "n_comm": n_comm, "rebuild_every": rebuild_every, "seed": seed,
    }

    def _joint(F, V):
        return np.concatenate([st["fnorm"].norm(F), st["vb"] * st["vnorm"].norm(V)[:, None]], 1)

    def reward(F, V):                       # F (N,feat_dim), V (N,) -> bonus (N,)
        F = np.asarray(F, np.float32); V = np.asarray(V, np.float32)
        if st["n"] < max(64, st["k"] + 1):
            return np.zeros(F.shape[0], np.float32)
        Bj = _joint(st["bufF"][: st["n"]], st["bufV"][: st["n"]])
        x = _joint(F, V)
        d2 = (x * x).sum(1)[:, None] + (Bj * Bj).sum(1)[None, :] - 2.0 * (x @ Bj.T)
        d2 = np.maximum(d2, 0.0)
        kth = np.partition(d2, st["k"], axis=1)[:, st["k"]]      # k-th NN squared dist
        r0 = 0.5 * np.log(kth + 1e-6)
        if st["use_cluster"] and st["centroids"] is not None:
            C = st["centroids"]                                  # (K, Dj) joint-space centroids
            cc = (C * C).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (C @ C.T)
            np.fill_diagonal(cc, np.inf)
            kc = min(st["k"], C.shape[0] - 1)
            ckth = np.partition(cc, kc, axis=1)[:, kc]
            r1c = 0.5 * np.log(ckth + 1e-6)                      # centroid-level entropy
            cid = (((x * x).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (x @ C.T))).argmin(1)
            return (r0 - r1c[cid]).astype(np.float32)
        return r0.astype(np.float32)

    def push(F, V):
        F = np.asarray(F, np.float32); V = np.asarray(V, np.float32)
        st["fnorm"].update(F); st["vnorm"].update(V)
        for j in range(F.shape[0]):
            st["bufF"][st["i"]] = F[j]; st["bufV"][st["i"]] = V[j]
            st["i"] = (st["i"] + 1) % st["cap"]; st["n"] = min(st["n"] + 1, st["cap"])
        st["since"] += F.shape[0]

    def maybe_rebuild():
        if st["use_cluster"] and st["since"] >= st["rebuild_every"] and st["n"] >= max(256, st["n_comm"] * 4):
            Bj = _joint(st["bufF"][: st["n"]], st["bufV"][: st["n"]])
            st["centroids"] = _np_kmeans(Bj, st["n_comm"], seed=st["seed"])
            st["since"] = 0
            return True
        return False

    st["reward"] = reward; st["push"] = push; st["maybe_rebuild"] = maybe_rebuild
    return st


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
