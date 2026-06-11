"""Numerical cross-check of helios.se_jax against the official selib.

Run inside /tmp/sevenv (has jax + networkx + editable selib install):

    /tmp/sevenv/bin/python -m pytest tests/test_se_jax.py -v
    # or just:
    /tmp/sevenv/bin/python tests/test_se_jax.py

Proves:
  * se_1d matches selib.calc.one_dimensional
  * se_2d(hard, symmetrize=True) matches selib.metrics.structural_entropy_2d
    on karate / les_miserables / planted-SBM to < 1e-4
  * soft S with large softmax temperature -> hard limit
  * quantifies the OLD Glass directed-vs-undirected discrepancy
"""
import os
import sys

import numpy as np
import networkx as nx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # match selib's float64 precision

# Import se_jax directly by file path to avoid pulling in helios/__init__
# (which imports flax/optax — not installed in this lean validation venv).
import importlib.util  # noqa: E402

_se_path = os.path.join(os.path.dirname(__file__), "..", "src", "helios", "se_jax.py")
_spec = importlib.util.spec_from_file_location("helios_se_jax", _se_path)
_se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_se)
se_1d, se_2d, se_2d_gap = _se.se_1d, _se.se_2d, _se.se_2d_gap

from selib.calc import one_dimensional as selib_1d  # noqa: E402
from selib.metrics import structural_entropy_2d as selib_2d  # noqa: E402

TOL = 1e-4


# ----------------------------------------------------------------------------
# graph fixtures
# ----------------------------------------------------------------------------
def _graphs():
    g_k = nx.convert_node_labels_to_integers(nx.karate_club_graph())
    g_lm = nx.convert_node_labels_to_integers(nx.les_miserables_graph())
    # planted 3-block SBM
    sizes = [25, 30, 20]
    p_in, p_out = 0.35, 0.03
    probs = [[p_in if i == j else p_out for j in range(3)] for i in range(3)]
    g_sbm = nx.convert_node_labels_to_integers(
        nx.stochastic_block_model(sizes, probs, seed=7)
    )
    return {"karate": g_k, "les_miserables": g_lm, "sbm3": g_sbm}


def _adj(G):
    return np.asarray(nx.to_numpy_array(G, weight="weight"), dtype=np.float64)


def _labels_for(G, name):
    """A non-trivial partition to test the SE *value* (not just optimality)."""
    if name == "sbm3":
        # ground-truth blocks live in the 'block' node attr
        return np.array([G.nodes[n]["block"] for n in range(G.number_of_nodes())])
    # greedy modularity communities -> labels aligned to node index
    comms = list(nx.community.greedy_modularity_communities(G))
    lab = np.zeros(G.number_of_nodes(), dtype=int)
    for c, nodes in enumerate(comms):
        for n in nodes:
            lab[n] = c
    return lab


def _one_hot(labels, K=None):
    labels = np.asarray(labels)
    K = K or int(labels.max() + 1)
    oh = np.zeros((labels.size, K), dtype=np.float64)
    oh[np.arange(labels.size), labels] = 1.0
    return oh


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_se_1d_matches_selib():
    for name, G in _graphs().items():
        A = jnp.asarray(_adj(G))
        mine = float(se_1d(A, symmetrize=True))
        ref = float(selib_1d(G))
        assert abs(mine - ref) < TOL, f"1D {name}: mine={mine} selib={ref}"
        print(f"[1D] {name:14s} mine={mine:.6f} selib={ref:.6f} diff={abs(mine-ref):.2e}")


def test_se_2d_matches_selib_hard():
    for name, G in _graphs().items():
        A = jnp.asarray(_adj(G))
        labels = _labels_for(G, name)
        S = jnp.asarray(_one_hot(labels))
        mine = float(se_2d(A, S, symmetrize=True, hard=True))
        ref = float(selib_2d(G, list(labels)))
        assert abs(mine - ref) < TOL, f"2D {name}: mine={mine} selib={ref}"
        print(f"[2D] {name:14s} mine={mine:.6f} selib={ref:.6f} diff={abs(mine-ref):.2e}")


def test_se_2d_soft_no_hard_equals_onehot():
    """With a genuine one-hot S, hard=False must equal hard=True (and selib)."""
    for name, G in _graphs().items():
        A = jnp.asarray(_adj(G))
        labels = _labels_for(G, name)
        S = jnp.asarray(_one_hot(labels))
        soft_val = float(se_2d(A, S, symmetrize=True, hard=False))
        ref = float(selib_2d(G, list(labels)))
        assert abs(soft_val - ref) < TOL, f"soft-onehot {name}: {soft_val} vs {ref}"


def test_soft_relaxation_converges_to_hard():
    """softmax(temp * logits) -> one-hot as temp grows; SE -> hard SE."""
    G = _graphs()["sbm3"]
    A = jnp.asarray(_adj(G))
    labels = _labels_for(G, "sbm3")
    K = int(labels.max() + 1)
    rng = np.random.default_rng(0)
    # logits peaked at the true label, plus noise
    logits = rng.normal(size=(labels.size, K)) * 0.3
    logits[np.arange(labels.size), labels] += 1.0
    logits = jnp.asarray(logits)

    hard_ref = float(selib_2d(G, list(np.argmax(np.asarray(logits), axis=1))))
    prev = None
    for temp in [1.0, 5.0, 20.0, 100.0]:
        S = jax.nn.softmax(temp * logits, axis=-1)
        val = float(se_2d(A, S, symmetrize=True, hard=False))
        print(f"[soft->hard] temp={temp:6.1f} se={val:.6f} hard_ref={hard_ref:.6f}")
        prev = val
    assert abs(prev - hard_ref) < 1e-3, f"temp=100 {prev} vs hard {hard_ref}"


def test_differentiable():
    """Gradients flow through both A and S."""
    G = _graphs()["karate"]
    A = jnp.asarray(_adj(G))
    labels = _labels_for(G, "karate")
    logits = jnp.asarray(_one_hot(labels) * 2.0)

    def loss_S(logits):
        S = jax.nn.softmax(logits, axis=-1)
        return se_2d(A, S, symmetrize=True)

    def loss_A(A):
        S = jax.nn.softmax(logits, axis=-1)
        return se_2d(A, S, symmetrize=True)

    gS = jax.grad(loss_S)(logits)
    gA = jax.grad(loss_A)(A)
    assert np.isfinite(np.asarray(gS)).all() and np.abs(np.asarray(gS)).sum() > 0
    assert np.isfinite(np.asarray(gA)).all() and np.abs(np.asarray(gA)).sum() > 0
    print(f"[grad] |dSE/dlogits|={float(jnp.abs(gS).sum()):.4f}  "
          f"|dSE/dA|={float(jnp.abs(gA).sum()):.4f}")


def test_gap_metric():
    for name, G in _graphs().items():
        A = jnp.asarray(_adj(G))
        labels = _labels_for(G, name)
        S = jnp.asarray(_one_hot(labels))
        gap = float(se_2d_gap(A, S, symmetrize=True, hard=True))
        h1 = float(selib_1d(G))
        h2 = float(selib_2d(G, list(labels)))
        ref = (h1 - h2) / h1
        assert abs(gap - ref) < TOL, f"gap {name}: {gap} vs {ref}"
        print(f"[gap] {name:14s} gap={gap:.4f}")


def test_old_glass_directed_discrepancy():
    """Quantify the OLD Glass directed-vs-undirected discrepancy.

    Build an ASYMMETRIC transition-like A. Compare:
      (a) old-glass two_dimensional_structural_entropy(A, one_hot)  [directed]
      (b) new se_2d(A, one_hot, symmetrize=True)                    [undirected, selib-matched]
      (c) new se_2d(A, one_hot, symmetrize=False)                   [directed, should == (a)]
    """
    rng = np.random.default_rng(3)
    N, K = 24, 3
    # asymmetric, row-stochastic-ish transition weights
    A_np = rng.uniform(0.0, 1.0, size=(N, N))
    A_np[rng.uniform(size=(N, N)) < 0.6] = 0.0  # sparsify
    np.fill_diagonal(A_np, 0.0)
    A = jnp.asarray(A_np, dtype=jnp.float64)
    assert not np.allclose(A_np, A_np.T), "test graph must be asymmetric"

    labels = rng.integers(0, K, size=N)
    S = jnp.asarray(_one_hot(labels, K))

    # (b) undirected selib-matched: build the undirected weighted graph from A_sym
    A_sym = 0.5 * (A_np + A_np.T)
    G_sym = nx.from_numpy_array(A_sym)  # undirected, uses 'weight'
    new_undirected = float(se_2d(A, S, symmetrize=True))
    selib_undirected = float(selib_2d(G_sym, list(labels)))
    assert abs(new_undirected - selib_undirected) < TOL, (
        f"new undirected {new_undirected} vs selib {selib_undirected}")

    # (c) new directed
    new_directed = float(se_2d(A, S, symmetrize=False))

    # (a) old glass: VERBATIM copy of the two SE functions from
    #     src/helios/algorithms/tdmpc_glass.py (lines ~207-247). We copy rather
    #     than import because the module imports flax/optax (not in this lean
    #     venv). This copy is byte-for-byte the old algebra under test.
    def _og_1d(A, mask=None, eps=1e-8):
        if mask is not None:
            A = A * mask[:, None] * mask[None, :]
        d = jnp.sum(A, axis=-1)
        two_m = jnp.sum(d)
        p = jnp.clip(d / (two_m + eps), eps, 1.0)
        if mask is not None:
            p = p * mask
        return -jnp.sum(p * jnp.log2(p))

    def _og_2d(A, S, mask=None, is_logits=True, eps=1e-8):
        if is_logits:
            S = jax.nn.softmax(S, axis=-1)
        if mask is not None:
            S = S * mask[:, None]
            A = A * mask[:, None] * mask[None, :]
        d = jnp.sum(A, axis=-1)
        two_m = jnp.sum(d)
        V = jnp.dot(d, S)
        AS = jnp.dot(A, S)
        g = jnp.sum(S * (d[:, None] - AS), axis=0)
        p_vol = V / (two_m + eps)
        p_cut = g / (two_m + eps)
        term1 = -jnp.sum(p_cut * jnp.log2(jnp.clip(p_vol, eps, 1.0)))
        h1 = _og_1d(A, mask=mask, eps=eps)
        term2 = h1 + jnp.sum(p_vol * jnp.log2(jnp.clip(p_vol, eps, 1.0)))
        return term1 + term2

    old_glass = float(_og_2d(A, S, is_logits=False))

    print("\n==== OLD GLASS DIRECTED-vs-UNDIRECTED DISCREPANCY ====")
    print(f"  (a) old glass two_dimensional_structural_entropy : {old_glass:.6f}")
    print(f"  (c) new se_2d(symmetrize=False)  [directed]      : {new_directed:.6f}")
    print(f"  (b) new se_2d(symmetrize=True)   [undirected]    : {new_undirected:.6f}")
    print(f"      selib structural_entropy_2d(A_sym)           : {selib_undirected:.6f}")
    print(f"  |old_glass - new_directed|   = {abs(old_glass - new_directed):.2e}")
    print(f"  |old_glass - selib_undir|    = {abs(old_glass - selib_undirected):.6f}")
    print("======================================================\n")

    # VERDICT: old glass == the directed quantity (not buggy), but != selib undirected.
    assert abs(old_glass - new_directed) < TOL, (
        "old glass should equal the directed se_2d (proves it's a correct "
        "directed SE, just not the undirected canonical one)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\nALL SE_JAX TESTS PASSED")
