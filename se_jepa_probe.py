"""SE-vs-VICReg JEPA latent MECHANISM PROBE on PandaPickCube.

Question (from the H-JEPA NULL): does a Structural-Entropy-regularized JEPA latent
preserve task-relevant end-effector->cube geometry BETTER than a matched VICReg
latent?  Cheap + decisive: train the ENCODER only (SimNorm enc + 1-step latent
predictor + EMA target), freeze it, then linear-probe the latent for the geometry
quantities obs[46:49] (box-gripper, the ee->cube vector) and obs[49:52]
(target-box).  Everything is held identical across conditions except the
structure / anti-collapse term.

Conditions (matched: SAME cached data, SAME arch/lr/steps; seed varies init+order):
  vicreg            VICReg variance-hinge + covariance (the H-JEPA baseline term)
  se                differentiable 2D structural entropy on the batch's kNN latent
                    graph, minimized (in place of VICReg).
  se_randgraph      CONTROL: identical SE loss but on a RANDOM graph (adjacency
                    independent of the latents). If se ~ se_randgraph, the SE
                    *structure* is not load-bearing (USING_SE.md random-graph trap).

The differentiable 2D-SE is validated numerically against selib.metrics
.structural_entropy_2d at startup (faithfulness check; USING_SE.md repro trap).

USAGE
  python se_jepa_probe.py collect            # build the cached transition dataset
  python se_jepa_probe.py train --cond se --seed 0
  python se_jepa_probe.py validate_se        # diff-SE vs selib check only
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HELIOS = Path("/root/tdmpc_glass/helios-rl")
sys.path.insert(0, str(HELIOS / "src"))
sys.path.insert(0, "/root/tdmpc_glass/mujoco_playground_repo")
sys.path.insert(0, "/root/tdmpc_glass/selib")

OUT = Path("/root/tdmpc_glass/exp/se_jepa_panda")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "data.npz"

LD = 256          # latent dim
V = 8             # SimNorm groups
HID = (512, 512)
BATCH = 256
STEPS = 15000
LR = 3e-4
EMA = 0.99
KNN = 15          # kNN graph degree for SE
KC = 16           # number of soft clusters for SE assignment head
TAU_S = 0.5       # cluster-assignment softmax temperature
LAM_SE = 1.0      # SE loss weight
VICREG_W = 3.0    # var-hinge weight (matches run_hjepa)
VICCOV_W = 1.0
VIC_GAMMA = 0.05

# geometry probe targets (held-out R^2)
GEO_SLICES = {"ee_to_cube_46_49": (46, 49), "box_to_target_49_52": (49, 52),
              "geom_all_46_58": (46, 58)}


# ---------------------------------------------------------------------------
# 1. Data collection (random policy; cached so all conditions are matched)
# ---------------------------------------------------------------------------
def collect(n_steps_per_env=3000, num_envs=16, seed=0):
    import jax, jax.numpy as jnp
    from mujoco_playground import registry, wrapper
    env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
    env = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1)
    obs_dim, act_dim = int(env.observation_size), int(env.action_size)
    _reset = jax.jit(env.reset); _step = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    st = _reset(jax.random.split(rk, num_envs))
    O, A, O2, D = [], [], [], []
    t0 = time.time()
    for t in range(n_steps_per_env):
        key, ak = jax.random.split(key)
        a = jax.random.uniform(ak, (num_envs, act_dim), minval=-1, maxval=1)
        o = np.asarray(st.obs)
        nst = _step(st, a)
        O.append(o); A.append(np.asarray(a)); O2.append(np.asarray(nst.obs))
        D.append(np.asarray(nst.done).astype(np.float32))
        st = nst
        if t % 500 == 0:
            print(f"[collect] {t}/{n_steps_per_env} {time.time()-t0:.0f}s", flush=True)
    O = np.concatenate(O); A = np.concatenate(A); O2 = np.concatenate(O2); D = np.concatenate(D)
    keep = D < 0.5                      # drop transitions that crossed episode boundary
    O, A, O2 = O[keep], A[keep], O2[keep]
    print(f"[collect] obs_dim={obs_dim} act_dim={act_dim} N={O.shape[0]}", flush=True)
    np.savez_compressed(CACHE, O=O.astype(np.float32), A=A.astype(np.float32),
                        O2=O2.astype(np.float32), obs_dim=obs_dim, act_dim=act_dim)
    print(f"[collect] saved {CACHE} size={CACHE.stat().st_size/1e6:.1f}MB", flush=True)


# ---------------------------------------------------------------------------
# 2. Differentiable 2D structural entropy  (validated vs selib)
# ---------------------------------------------------------------------------
def build_se_fns():
    import jax, jax.numpy as jnp

    def knn_adj(z):
        """Symmetric kNN cosine graph on the batch latents (differentiable values,
        stop-grad topk mask)."""
        zc = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        sim = zc @ zc.T
        B = z.shape[0]
        sim = sim - jnp.eye(B) * 10.0                 # drop self-loops
        thr = jax.lax.top_k(sim, KNN)[0][:, -1:]      # kth-largest per row
        mask = jax.lax.stop_gradient((sim >= thr).astype(z.dtype))
        A = mask * jax.nn.relu(sim)
        A = jnp.maximum(A, A.T)                        # symmetrize
        return A

    def se2d_soft(A, S):
        """Differentiable 2D-SE (bits) of soft partition S over weighted adj A.
        H^2 = -sum_v (d_v/2m) sum_c S_vc log2(d_v/V_c)  -  sum_c (g_c/2m) log2(V_c/2m).
        """
        deg = A.sum(1)                                 # (B,)
        two_m = deg.sum() + 1e-12
        Vol = S.T @ deg                                # (k,)
        intra = jnp.diag(S.T @ A @ S)                  # within-cluster volume
        cut = jnp.clip(Vol - intra, 0.0, None)         # g_c
        logVol = jnp.log2(Vol + 1e-12)
        leaf = -jnp.sum((deg / two_m) * (jnp.log2(deg + 1e-12) - S @ logVol))
        cutt = -jnp.sum((cut / two_m) * jnp.log2((Vol + 1e-12) / two_m))
        return leaf + cutt

    return knn_adj, se2d_soft


def validate_se():
    """Numeric faithfulness check: diff-SE (one-hot S) == selib.structural_entropy_2d."""
    import jax, jax.numpy as jnp
    import networkx as nx
    from selib.metrics import structural_entropy_2d
    _, se2d_soft = build_se_fns()
    rng = np.random.default_rng(0)
    n = 60
    G = nx.gnm_random_graph(n, 240, seed=1)
    labels = rng.integers(0, 5, n)
    A = nx.to_numpy_array(G, nodelist=list(range(n)))   # symmetric 0/1
    S = np.eye(5)[labels]
    mine = float(se2d_soft(jnp.asarray(A), jnp.asarray(S.astype(np.float32))))
    ref = float(structural_entropy_2d(G, list(labels)))
    err = abs(mine - ref)
    print(f"[validate_se] diff-SE={mine:.6f} selib={ref:.6f} abs_err={err:.2e}", flush=True)
    res = {"diff_se": mine, "selib_se": ref, "abs_err": err, "match": err < 1e-4}
    (OUT / "se_validation.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# 3. numpy helpers (avoid sklearn/scipy: tight disk)
# ---------------------------------------------------------------------------
def ridge_r2(Ztr, Ytr, Zte, Yte, alpha=1.0):
    """Closed-form ridge; returns held-out R^2 (averaged over target dims)."""
    n, d = Ztr.shape
    Zc = np.concatenate([Ztr, np.ones((n, 1))], 1)
    Zce = np.concatenate([Zte, np.ones((Zte.shape[0], 1))], 1)
    A = Zc.T @ Zc
    reg = alpha * np.eye(A.shape[0]); reg[-1, -1] = 0.0   # don't penalize bias
    W = np.linalg.solve(A + reg, Zc.T @ Ytr)
    pred = Zce @ W
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(np.mean(r2)), [float(x) for x in r2]


def latent_health(Z):
    Z = np.asarray(Z, np.float64)
    n, D = Z.shape
    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None); s = ev.sum()
    eff_rank = float((s * s) / (np.square(ev).sum() + 1e-12)) if s > 0 else 0.0
    grp = Z.reshape(n, V, D // V); codes = grp.argmax(-1)
    ent = []
    for g in range(V):
        cnt = np.bincount(codes[:, g], minlength=D // V).astype(np.float64)
        p = cnt / cnt.sum(); p = p[p > 0]; ent.append(float(-(p * np.log(p)).sum()))
    max_ent = float(np.log(D // V))
    return {"z_eff_rank": eff_rank, "z_std_mean": float(Z.std(0).mean()),
            "code_entropy_frac": float(np.mean(ent) / max_ent)}


# ---------------------------------------------------------------------------
# 4. Train one encoder under one condition, then probe
# ---------------------------------------------------------------------------
def train(cond, seed, lam_se=LAM_SE):
    import jax, jax.numpy as jnp, flax.linen as nn, optax
    from helios.algorithms.tdmpc2 import simnorm, NormMLP

    d = np.load(CACHE)
    O, A_, O2 = d["O"], d["A"], d["O2"]
    obs_dim, act_dim = int(d["obs_dim"]), int(d["act_dim"])
    N = O.shape[0]
    rng = np.random.default_rng(seed)
    # train/test split for the FROZEN probe (held-out)
    perm = rng.permutation(N); ntr = int(0.85 * N)
    tr, te = perm[:ntr], perm[ntr:]

    class Encoder(nn.Module):
        @nn.compact
        def __call__(self, o):
            return simnorm(NormMLP(HID, LD)(o), V)

    class Pred(nn.Module):
        @nn.compact
        def __call__(self, z, a):
            x = jnp.concatenate([z, a], -1)
            return simnorm(NormMLP(HID, LD)(x), V)

    enc, pred = Encoder(), Pred()
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)
    p_enc = enc.init(k1, jnp.zeros((1, obs_dim)))
    p_pred = pred.init(k2, jnp.zeros((1, LD)), jnp.zeros((1, act_dim)))
    # SE soft-cluster centroids (learned head; only used by SE conditions)
    key, k3 = jax.random.split(key)
    C0 = jax.random.normal(k3, (KC, LD)) * 0.1
    params = {"enc": p_enc, "pred": p_pred, "C": C0}
    target = {"enc": jax.tree.map(lambda x: x, p_enc)}
    tx = optax.chain(optax.clip_by_global_norm(20.0), optax.adam(LR))
    opt = tx.init(params)

    knn_adj, se2d_soft = build_se_fns()

    def vicreg(z):
        B, D = z.shape
        std = jnp.sqrt(z.var(0) + 1e-4)
        v = jnp.mean(jax.nn.relu(VIC_GAMMA - std))
        zc = z - z.mean(0, keepdims=True); cov = (zc.T @ zc) / (B - 1)
        off = cov - jnp.diag(jnp.diag(cov)); c = jnp.sum(off ** 2) / D
        return VICREG_W * v + VICCOV_W * c

    def struct_loss(params, z_t, z_tk, zhat, rand_adj):
        if cond == "vicreg":
            return vicreg(z_t) + vicreg(z_tk) + vicreg(zhat)
        # SE conditions: minimize 2D-SE on the latent graph
        if cond == "se":
            A = knn_adj(z_t)
        elif cond == "se_randgraph":
            A = rand_adj                                  # graph independent of z
        else:
            raise ValueError(cond)
        S = jax.nn.softmax((z_t @ params["C"].T) / TAU_S, axis=1)
        return lam_se * se2d_soft(A, S)

    def loss_fn(params, target, o, a, o2, rand_adj):
        z_t = enc.apply(params["enc"], o)
        zhat = pred.apply(params["pred"], z_t, a)
        z_tgt = jax.lax.stop_gradient(enc.apply(target["enc"], o2))
        l_pred = jnp.mean(jnp.sum((zhat - z_tgt) ** 2, -1))
        z_tk = enc.apply(params["enc"], o2)
        l_struct = struct_loss(params, z_t, z_tk, zhat, rand_adj)
        return l_pred + l_struct, (l_pred, l_struct)

    @jax.jit
    def step(params, target, opt, o, a, o2, rand_adj):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
            params, target, o, a, o2, rand_adj)
        upd, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, upd)
        target = {"enc": jax.tree.map(lambda t, o_: EMA * t + (1 - EMA) * o_,
                                      target["enc"], params["enc"])}
        return params, target, opt, loss, aux

    t0 = time.time()
    Otr, Atr, O2tr = O[tr], A_[tr], O2[tr]
    log = []
    for it in range(STEPS):
        idx = rng.integers(0, Otr.shape[0], BATCH)
        o = jnp.asarray(Otr[idx]); a = jnp.asarray(Atr[idx]); o2 = jnp.asarray(O2tr[idx])
        # random-graph control: a fixed-structure random symmetric kNN-ish graph
        if cond == "se_randgraph":
            rk = rng.integers(0, 1 << 30)
            rg = np.random.default_rng(rk)
            M = np.zeros((BATCH, BATCH), np.float32)
            for i in range(BATCH):
                js = rg.choice(BATCH, KNN, replace=False)
                M[i, js] = rg.random(KNN).astype(np.float32)
            M = np.maximum(M, M.T); np.fill_diagonal(M, 0.0)
            rand_adj = jnp.asarray(M)
        else:
            rand_adj = jnp.zeros((BATCH, BATCH))
        params, target, opt, loss, aux = step(params, target, opt, o, a, o2, rand_adj)
        if it % 1000 == 0 or it == STEPS - 1:
            lp, ls = float(aux[0]), float(aux[1])
            log.append({"it": it, "loss": float(loss), "l_pred": lp, "l_struct": ls})
            print(f"[{cond} s{seed}] it={it} loss={float(loss):.4f} "
                  f"l_pred={lp:.4f} l_struct={ls:.4f} {time.time()-t0:.0f}s", flush=True)

    # ---- freeze + probe ----
    enc_j = jax.jit(lambda p, o: enc.apply(p["enc"], o))
    def encode_all(obs):
        out = []
        for i in range(0, obs.shape[0], 4096):
            out.append(np.asarray(enc_j(params, jnp.asarray(obs[i:i + 4096]))))
        return np.concatenate(out)
    Ztr = encode_all(O[tr]); Zte = encode_all(O[te])
    probe = {}
    for name, (lo, hi) in GEO_SLICES.items():
        r2, per = ridge_r2(Ztr, O[tr][:, lo:hi], Zte, O[te][:, lo:hi])
        probe[name] = {"r2": r2, "r2_per_dim": per}
    health = latent_health(Zte[:4000])

    res = {"cond": cond, "seed": seed, "steps": STEPS, "n_train": int(len(tr)),
           "n_test": int(len(te)), "latent_dim": LD, "knn": KNN, "kc": KC,
           "lam_se": lam_se, "wall_s": round(time.time() - t0, 1),
           "train_log": log, "probe": probe, "health": health}
    tag = f"_lam{lam_se}" if (cond == "se" and lam_se != LAM_SE) else ""
    fn = OUT / f"run_{cond}{tag}_seed{seed}.json"
    fn.write_text(json.dumps(res, indent=2))
    print(f"[{cond} s{seed}] DONE probe ee_to_cube R2="
          f"{probe['ee_to_cube_46_49']['r2']:.4f} eff_rank={health['z_eff_rank']:.1f} "
          f"-> {fn}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "validate_se", "train"])
    ap.add_argument("--cond", default="vicreg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam_se", type=float, default=LAM_SE)
    args = ap.parse_args()
    if args.mode == "collect":
        collect()
    elif args.mode == "validate_se":
        validate_se()
    elif args.mode == "train":
        validate_se()        # cheap faithfulness check before every run
        train(args.cond, args.seed, lam_se=args.lam_se)
