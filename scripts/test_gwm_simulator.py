#!/usr/bin/env python3
"""Tiny CPU-runnable selftest for the GWM-as-simulator mechanism-check infra.

Runs WITHOUT a GPU; needs only jax + flax + optax on CPU. Does NOT touch the
queue/daemons or run real training.

Checks:
  1. contact env builds at N=5; obs / per-entity shapes correct; reset/step jit;
     contact_graph in info with right shape, symmetric, zero diagonal, 0/1.
  2. PHYSICS: disks do NOT overlap after a step (separation enforced); the box
     keeps disks inside the walls; momentum of a free 2-disk head-on collision is
     approximately conserved (no actuation/damping); the contact graph is nonzero
     exactly when two disks are placed adjacent + approaching.
  3. reward sparsity: perturbing a value-IRRELEVANT disk leaves reward unchanged;
     perturbing the value target (or agent) CHANGES it.
  4. N-monotonic value role: value_target / value_relevant stable across N.
  5. BOTH WMs (EntityWM graph + MonolithicWM) build, forward to the shared head
     contract, and run 2 train steps with finite loss; param counts reported and
     matched_hidden returns a sane width.

Usage:
    PYTHONPATH=<repo>/src python3 <repo>/scripts/test_gwm_simulator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from helios.dynamics.entity_wm import EntityWM, entity_wm_loss
from helios.dynamics.monolithic_wm import (
    MonolithicWM,
    matched_hidden,
    monolithic_wm_loss,
    _count_params,
)
from helios.envs import contact_entities as ce


def check_env_shapes():
    env = ce.make_env(n_entities=5, seed=0)
    state = env.reset(jax.random.PRNGKey(0))
    obs = env.observe(state)
    ent = env.observe_entities(state)
    assert obs.shape == (5 * ce.ENTITY_DIM,), obs.shape
    assert ent.shape == (5, ce.ENTITY_DIM), ent.shape

    jreset = jax.jit(lambda k: env.reset(k))
    jstep = jax.jit(lambda s, a: env.step(s, a))
    s = jreset(jax.random.PRNGKey(1))
    s2, o2, r, d, info = jstep(s, jnp.array([0.5, -0.5]))
    assert o2.shape == (5 * ce.ENTITY_DIM,)
    assert jnp.isfinite(r)
    cg = info["contact_graph"]
    assert cg.shape == (5, 5), cg.shape
    assert jnp.allclose(cg, cg.T), "contact graph must be symmetric"
    assert jnp.allclose(jnp.diag(cg), 0.0), "contact graph diagonal must be zero"
    assert jnp.all((cg == 0) | (cg == 1)), "contact graph must be 0/1"
    assert "any_contact" in info and "wall_contact" in info
    print("[ok] env shapes + jit reset/step + contact_graph contract")
    return env


def check_no_overlap_and_walls():
    """After a step, movable disks must not overlap and must stay in the box."""
    env = ce.make_env(n_entities=6, seed=2)
    jstep = jax.jit(lambda s, a: env.step(s, a))
    key = jax.random.PRNGKey(11)
    state = env.reset(key)
    for t in range(40):
        key, ak = jax.random.split(key)
        a = jax.random.uniform(ak, (2,), minval=-1.0, maxval=1.0)
        state, _, _, _, _ = jstep(state, a)
    pos = np.asarray(state.pos)
    # walls: every disk center within [-(box-r), box-r] (goal pinned inside too)
    lim = env.box - env.radius
    assert np.all(pos >= -lim - 1e-4) and np.all(pos <= lim + 1e-4), (
        f"disk left the box: pos range [{pos.min()}, {pos.max()}], lim={lim}"
    )
    # no movable-movable overlap (allow tiny numerical slack)
    move = [i for i in range(env.n_entities) if i != ce.GOAL_IDX]
    min_sep = 2.0 * env.radius - 1e-2
    for a_i in range(len(move)):
        for b_i in range(a_i + 1, len(move)):
            i, j = move[a_i], move[b_i]
            dij = np.linalg.norm(pos[i] - pos[j])
            assert dij >= min_sep, (
                f"disks {i},{j} overlap after step: dist={dij:.4f} < {min_sep:.4f}"
            )
    print(f"[ok] no disk overlap + disks stay in walls (40 steps, N={env.n_entities})")


def check_momentum_and_contact_signal():
    """Free 2-disk head-on collision (no actuation/damping/walls): total momentum
    ~conserved and the contact graph fires when the disks meet."""
    env = ce.make_env(
        n_entities=4, seed=0, damping=0.0, box=100.0, init_speed=0.0, dt=0.05
    )
    # Place disks 0 and 2 (both movable; idx1 is the static goal) head-on, the
    # rest far away & at rest so only the 0-2 pair interacts.
    pos = jnp.array([
        [-0.5, 0.0],   # disk 0 (agent) moving +x
        env.goal_pos,  # disk 1 goal region (static)
        [0.5, 0.0],    # disk 2 moving -x
        [50.0, 50.0],  # disk 3 far away
    ])
    # adjacency: |0-2| = 1.0 > 2r=0.7, so not yet touching; give approaching vels
    vel = jnp.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    state = ce.EnvState(pos=pos, vel=vel, step=jnp.int32(0), key=jax.random.PRNGKey(0))

    move = np.array([i for i in range(env.n_entities) if i != ce.GOAL_IDX])
    p_before = np.sum(np.asarray(vel)[move], axis=0)

    jstep = jax.jit(lambda s, a: env.step(s, a))
    saw_contact = False
    for _ in range(40):
        state, _, _, _, info = jstep(state, jnp.zeros(2))
        if float(info["contact_graph"][0, 2]) > 0.5:
            saw_contact = True
    p_after = np.sum(np.asarray(state.vel)[move], axis=0)
    # Momentum of the free movable subsystem should be ~conserved (no damping,
    # no walls reached, no actuation). Elastic equal-mass impulse conserves it.
    assert np.allclose(p_before, p_after, atol=1e-3), (
        f"momentum not conserved: before={p_before} after={p_after}"
    )
    assert saw_contact, "contact graph never fired for an approaching disk pair"
    print(f"[ok] momentum ~conserved ({p_before}->{p_after}) + contact graph fires")

    # Adjacent-and-approaching => contact graph nonzero THIS step.
    pos2 = jnp.array([
        [-0.3, 0.0], env.goal_pos, [0.3, 0.0], [50.0, 50.0],
    ])  # |0-2|=0.6 < 2r=0.7 -> overlapping
    vel2 = jnp.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    st2 = ce.EnvState(pos=pos2, vel=vel2, step=jnp.int32(0), key=jax.random.PRNGKey(0))
    _, _, _, _, info2 = jstep(st2, jnp.zeros(2))
    assert float(info2["contact_graph"][0, 2]) == 1.0, (
        "adjacent+approaching disks did not register a contact"
    )
    assert float(jnp.sum(info2["contact_graph"])) > 0.0
    print("[ok] adjacent+approaching disks -> contact graph nonzero")


def check_reward_sparsity():
    env = ce.make_env(n_entities=6, seed=0)
    state = env.reset(jax.random.PRNGKey(3))
    base_r = ce._reward(env, state.pos)
    relevant = set(env.value_relevant_entities)
    irrelevant = sorted(set(range(env.n_entities)) - relevant)
    assert irrelevant, f"need an irrelevant disk; relevant={relevant}"
    irr = irrelevant[0]
    pos_irr = state.pos.at[irr].add(jnp.array([3.0, -2.0]))
    assert jnp.allclose(ce._reward(env, pos_irr), base_r), (
        f"reward changed for IRRELEVANT disk {irr}"
    )
    t = env.value_target
    pos_rel = state.pos.at[t].add(jnp.array([3.0, -2.0]))
    assert not jnp.allclose(ce._reward(env, pos_rel), base_r), (
        f"reward did NOT change for the value target {t}"
    )
    print(f"[ok] reward sparsity: relevant={sorted(relevant)} irrelevant={irrelevant}")


def check_n_monotonic():
    e5 = ce.make_env(n_entities=5, seed=0)
    e7 = ce.make_env(n_entities=7, seed=0)
    assert e5.value_target == e7.value_target == ce.TARGET_IDX
    assert e5.value_relevant_entities == e7.value_relevant_entities
    print(f"[ok] N-monotonic value role: target={e5.value_target} "
          f"relevant={e5.value_relevant_entities} stable across N")


def check_both_wms():
    N = 5
    rng = jax.random.PRNGKey(7)
    B = 8
    ent = jax.random.normal(rng, (B, N, ce.ENTITY_DIM))
    act = jax.random.normal(rng, (B, 2))

    graph = EntityWM(entity_dim=ce.ENTITY_DIM, action_dim=2, n_entities=N,
                     d_model=32, n_layers=2, n_heads=4, max_entities=16)
    g_params = graph.init(rng, ent[:2], act[:2])["params"]
    g_n = _count_params(g_params)
    g_out = graph.apply({"params": g_params}, ent, act)
    assert g_out["next_ent"].shape == (B, N, ce.ENTITY_DIM)
    assert g_out["reward"].shape == (B,) and g_out["q"].shape == (B,)

    hidden, target_n = matched_hidden(g_n, ce.ENTITY_DIM, 2, N, n_layers=2)
    assert 8 <= hidden <= 4096, hidden
    mono = MonolithicWM(entity_dim=ce.ENTITY_DIM, action_dim=2, n_entities=N,
                        hidden=hidden, n_layers=2)
    m_params = mono.init(rng, ent[:2], act[:2])["params"]
    m_n = _count_params(m_params)
    m_out = mono.apply({"params": m_params}, ent, act)
    assert m_out["next_ent"].shape == (B, N, ce.ENTITY_DIM)
    assert m_out["reward"].shape == (B,) and m_out["q"].shape == (B,)
    # monolithic OOD apply (N-agnostic via pooled encode)
    ent7 = jax.random.normal(rng, (B, 7, ce.ENTITY_DIM))
    m_out7 = mono.apply({"params": m_params}, ent7, act)
    assert m_out7["next_ent"].shape == (B, 7, ce.ENTITY_DIM)
    print(f"[ok] both WMs build; graph={g_n} params, mono(hidden={hidden})={m_n} "
          f"params (target {target_n}, ratio {m_n / g_n:.2f})")

    # 2 train steps each, finite loss.
    batch = {
        "ent": ent, "action": act, "next_ent": ent + 0.01,
        "reward": jnp.zeros((B,)), "q_target": jnp.zeros((B,)),
    }
    for name, model, params, loss_fn in [
        ("graph", graph, g_params, entity_wm_loss),
        ("mono", mono, m_params, monolithic_wm_loss),
    ]:
        opt = optax.adam(1e-3)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, model.apply, batch
            )
            upd, opt_state = opt.update(grads, opt_state)
            return optax.apply_updates(params, upd), opt_state, loss

        for i in range(2):
            params, opt_state, loss = step(params, opt_state)
            assert jnp.isfinite(loss), f"{name}: non-finite loss at step {i}"
        print(f"[ok] {name} WM 2 train steps; final loss = {float(loss):.4f}")


def main():
    print("=== gwm_simulator selftest ===")
    check_env_shapes()
    check_no_overlap_and_walls()
    check_momentum_and_contact_signal()
    check_reward_sparsity()
    check_n_monotonic()
    check_both_wms()
    print("=== ALL SELFTESTS PASSED ===")


if __name__ == "__main__":
    main()
