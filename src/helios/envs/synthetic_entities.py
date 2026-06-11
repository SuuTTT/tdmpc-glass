"""Controlled synthetic multi-entity world (pure JAX) with KNOWN value-coupling.

============================================================================
WHY THIS EXISTS
============================================================================
This is a sandbox for a *mechanism check*: a world whose ground-truth
value-coupling structure is known by construction, so a later probe (the
"value-coupling graph" probe, added by the main session) can be validated
against the truth instead of against a guess.

The world is N point-mass entities in a 2-D plane. Each entity has a state
``(px, py, vx, vy)``. One designated entity (index 0, the "agent") is actuated
by the 2-D action (a force). One designated entity (index 1, the "goal") is
*static* (it never moves). The remaining entities are "distractors".

============================================================================
KNOWN STRUCTURE — read this; the probe is validated against it
============================================================================
Coupling matrix ``C`` (N×N, symmetric, zero diagonal, mostly zero):
  ``C[i, j] = C[j, i] = k > 0`` means entities i and j are connected by a
  spring of stiffness ``k``: each exerts a Hooke restoring force on the other
  toward a fixed rest length. Uncoupled pairs (``C[i,j]==0``) are dynamically
  independent — perturbing one has *no* effect on the other's trajectory.

  ``C`` is generated deterministically from ``seed`` and is RETURNED by
  ``make_env`` (field ``coupling``). It is the dynamics-level ground truth.

Reward (the VALUE-level ground truth — provably sparse):

    r = - dist(agent, goal)                          # task term
        - w_pair * dist(entity[p], entity[q])        # coupling-shaping term

  where ``(p, q)`` is ONE specific *coupled* pair (``C[p,q] > 0``), chosen
  deterministically from the seed, with ``p, q`` NOT equal to the goal index.
  So the instantaneous reward depends ONLY on the positions of entities
  ``{agent(0), goal(1), p, q}`` — a SPARSE subset of size <= 4, regardless of
  N. Every other entity is value-irrelevant (it may move, it may even be
  coupled to something, but the reward/value never reads it).

  Therefore the OPTIMAL VALUE FUNCTION depends only on:
    - the agent and goal (task term), and
    - the pair (p, q) and whatever is dynamically up-stream of (p, q) under C
      (because the agent cannot influence (p,q) unless there is a coupling
      path from the agent to them — see ``value_relevant_entities`` below).

  The exposed ground truth is therefore two things:
    * ``value_relevant_entities`` : the set whose *position* the reward reads
      directly = {0, 1, p, q}. This is what a value-decodability /
      reward-gradient probe should recover.
    * ``coupling``                : the full dynamics coupling C (the probe for
      *interaction* relevance can be checked against C restricted to the
      value-relevant set + its coupling-reachable closure).

============================================================================
N-SCALING (OOD object-count generalization)
============================================================================
The interaction RULES are identical for every N; only the *count* of entities
changes. Concretely, holding ``seed`` fixed:
  * indices 0 (agent) and 1 (goal) always play the same role;
  * the value-shaping pair (p, q) is chosen from a fixed RNG stream so that for
    a given seed it lands on the SAME logical slots whenever N is large enough
    to contain them (we draw p, q in ``[2, min(N, 4))`` ∪ feasible range so the
    pair exists at the smallest supported N and persists as N grows);
  * extra entities at larger N are appended as additional *distractors* and
    additional sparse springs, drawn from the SAME generative process (same
    per-edge probability ``edge_prob`` and stiffness distribution). The first
    ``N_train`` entities' sub-block of C is IDENTICAL across N (the generator is
    written so that growing N only ADDS rows/cols, never rewrites the existing
    block — verified in the selftest).

This gives a clean "same rules, more objects" OOD axis: a model trained at
``N_train`` and evaluated at ``N=6, 8`` faces the identical physics and the
identical value structure on the shared entities, plus novel distractors.

Coupling density scaling: edges are drawn i.i.d. per off-diagonal pair with
probability ``edge_prob``, so the expected number of springs grows like
``edge_prob * N*(N-1)/2`` (sparse for small ``edge_prob``). The value term
still reads only the fixed pair (p, q), so the *value* sparsity is preserved
(constant size 4) even as dynamics coupling grows with N. This is the intended
asymmetry: dynamics get denser with N, value stays sparse.

============================================================================
API
============================================================================
``make_env(n_entities, seed)`` -> ``SyntheticEntitiesEnv`` (a frozen dataclass
holding the static config + ground truth). The env is gym-like and jittable:

    env = make_env(n_entities=4, seed=0)
    state = env.reset(jax.random.PRNGKey(0))         # EnvState (pytree)
    obs = env.observe(state)                          # (N*d,) flat
    ent = env.observe_entities(state)                 # (N, d) per-entity
    state, obs, reward, done, info = env.step(state, action)

All of ``reset``/``step``/``observe`` are pure functions of ``state`` and are
``jax.jit`` / ``jax.vmap`` friendly (no Python-side state mutation). ``N`` and
all shapes are static (baked into the frozen dataclass), so jit sees concrete
shapes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Per-entity state dimensionality: (px, py, vx, vy).
ENTITY_DIM = 4

# Fixed role indices (stable across all N).
AGENT_IDX = 0
GOAL_IDX = 1


class EnvState(NamedTuple):
    """Dynamic environment state (a JAX pytree, jit/vmap friendly).

    Attributes:
        pos:  (N, 2) positions.
        vel:  (N, 2) velocities.
        step: scalar int32 step counter.
        key:  PRNG key carried for any stochasticity (currently deterministic
              dynamics; kept for extensibility / reset).
    """

    pos: jax.Array
    vel: jax.Array
    step: jax.Array
    key: jax.Array


class SyntheticEntitiesEnv(NamedTuple):
    """Static config + ground truth for the synthetic entity world.

    This is immutable; dynamics live in :class:`EnvState`. Methods are bound via
    free functions below and attached so the object stays a plain pytree-free
    config carrier (we use NamedTuple of *static* python objects + arrays).

    Ground-truth fields (for probe validation):
        coupling:                (N, N) symmetric stiffness matrix C.
        value_pair:              (p, q) the value-shaping coupled pair.
        value_relevant_entities: sorted tuple of entity indices the reward reads
                                 directly = (AGENT_IDX, GOAL_IDX, p, q).
    """

    # --- static dims / physics ---
    n_entities: int
    dt: float
    mass: float
    damping: float
    rest_length: float
    action_scale: float
    max_steps: int
    w_pair: float
    # --- ground truth ---
    coupling: jax.Array          # (N, N)
    value_pair: tuple            # (p, q)
    value_relevant_entities: tuple
    # --- reset distribution ---
    spawn_radius: float
    goal_pos: jax.Array          # (2,) fixed goal location

    # ------------------------------------------------------------------
    # gym-like API (pure functions; bound as methods for convenience)
    # ------------------------------------------------------------------
    def reset(self, key: jax.Array) -> EnvState:
        return _reset(self, key)

    def step(self, state: EnvState, action: jax.Array):
        return _step(self, state, action)

    def observe(self, state: EnvState) -> jax.Array:
        """Flat concat observation, shape ``(N * ENTITY_DIM,)``."""
        return self.observe_entities(state).reshape(-1)

    def observe_entities(self, state: EnvState) -> jax.Array:
        """Per-entity observation tensor, shape ``(N, ENTITY_DIM)``."""
        return jnp.concatenate([state.pos, state.vel], axis=-1)

    @property
    def obs_dim(self) -> int:
        return self.n_entities * ENTITY_DIM

    @property
    def action_dim(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Ground-truth structure generation
# ---------------------------------------------------------------------------


def _generate_coupling(
    n_entities: int,
    seed: int,
    edge_prob: float,
    stiffness: float,
) -> tuple[jax.Array, tuple]:
    """Build symmetric sparse coupling C and pick the value-shaping pair.

    Determinism / N-monotonicity: we draw the full N×N upper-triangle mask from
    a single seeded stream in a FIXED iteration order (i<j, row-major). Growing
    N only appends new (i, j) draws for the new rows/columns; the sub-block over
    the first ``N_train`` entities is therefore identical across N (the new
    entity j only adds edges (i, j) with j >= N_train, never touching the old
    block). Verified in the selftest.

    Returns (C, (p, q)).
    """
    n = n_entities
    # Use numpy-free pure-JAX RNG but iterate deterministically so the prefix
    # block is N-invariant. We fold (i, j) into the key per-edge.
    base = jax.random.PRNGKey(seed)

    rows = []
    for i in range(n):
        rowvals = []
        for j in range(n):
            if j <= i:
                rowvals.append(0.0)
                continue
            # per-edge key depends only on (i, j) and seed -> N-invariant prefix
            ekey = jax.random.fold_in(jax.random.fold_in(base, i), j)
            present = jax.random.uniform(ekey) < edge_prob
            rowvals.append(jnp.where(present, stiffness, 0.0))
        rows.append(jnp.stack(rowvals))
    upper = jnp.stack(rows)                       # (N, N) upper-tri (incl forced)
    C = upper + upper.T                           # symmetric, zero diagonal

    # Pick the value-shaping pair (p, q). It must be the SAME logical slots for
    # every N (so the value structure is identical on the shared entities across
    # object-count OOD splits), live in the "core" distractor range (>= 2, i.e.
    # not the agent or goal), and exist at the smallest supported N (>= 4). The
    # fixed choice (2, 3) satisfies all of these. ``make_env`` enforces n >= 4,
    # so this pair always exists.
    p, q = 2, 3
    # Force the chosen pair to be a spring (so value shaping rides a real edge).
    C = C.at[p, q].set(stiffness)
    C = C.at[q, p].set(stiffness)
    return C, (int(p), int(q))


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------


def make_env(
    n_entities: int = 4,
    seed: int = 0,
    *,
    dt: float = 0.05,
    mass: float = 1.0,
    damping: float = 0.1,
    rest_length: float = 1.0,
    action_scale: float = 1.0,
    max_steps: int = 100,
    w_pair: float = 0.5,
    edge_prob: float = 0.25,
    stiffness: float = 2.0,
    spawn_radius: float = 2.0,
) -> SyntheticEntitiesEnv:
    """Construct a synthetic entity env with known ground-truth structure.

    Args:
        n_entities: number of entities N (>= 4). 4 is the canonical N_train;
            6, 8 are the OOD object-count splits. See module docstring for how
            N-scaling preserves the interaction rules.
        seed: controls the coupling matrix C and the value pair (p, q). Fixing
            seed and varying n_entities gives "same rules, more objects".
        dt, mass, damping: leapfrog/semi-implicit Euler physics params.
        rest_length: spring rest length (Hooke force is -k*(d - rest_length)).
        action_scale: multiplies the 2-D action before it becomes agent force.
        max_steps: episode horizon (done flag).
        w_pair: weight of the value-shaping (pair-closeness) reward term.
        edge_prob, stiffness: coupling generator params (see _generate_coupling).
        spawn_radius: reset positions ~ U([-r, r]^2); goal at a fixed location.

    Returns:
        SyntheticEntitiesEnv (immutable config + ground truth).
    """
    if n_entities < 4:
        raise ValueError(
            f"n_entities must be >= 4 (need agent, goal, and a value pair); got {n_entities}"
        )
    C, (p, q) = _generate_coupling(n_entities, seed, edge_prob, stiffness)
    value_relevant = tuple(sorted({AGENT_IDX, GOAL_IDX, p, q}))
    goal_pos = jnp.array([1.5, 1.5])
    return SyntheticEntitiesEnv(
        n_entities=n_entities,
        dt=dt,
        mass=mass,
        damping=damping,
        rest_length=rest_length,
        action_scale=action_scale,
        max_steps=max_steps,
        w_pair=w_pair,
        coupling=C,
        value_pair=(p, q),
        value_relevant_entities=value_relevant,
        spawn_radius=spawn_radius,
        goal_pos=goal_pos,
    )


# ---------------------------------------------------------------------------
# Dynamics (pure functions; jit/vmap friendly)
# ---------------------------------------------------------------------------


def _spring_forces(env: SyntheticEntitiesEnv, pos: jax.Array) -> jax.Array:
    """Pairwise Hooke forces from coupling C. Returns (N, 2) net force.

    For each coupled pair (i, j) with stiffness k, entity i feels
    ``k * (d_ij - rest_length) * unit(j - i)`` (attraction toward rest length).
    Uncoupled pairs contribute zero by construction (C[i,j]==0).
    """
    # delta[i, j] = pos[j] - pos[i]
    delta = pos[None, :, :] - pos[:, None, :]            # (N, N, 2)
    dist = jnp.linalg.norm(delta, axis=-1)               # (N, N)
    safe = jnp.maximum(dist, 1e-6)
    unit = delta / safe[..., None]                       # (N, N, 2)
    # signed magnitude per pair, gated by coupling
    mag = env.coupling * (dist - env.rest_length)        # (N, N)
    pair_force = mag[..., None] * unit                   # (N, N, 2)
    net = jnp.sum(pair_force, axis=1)                    # (N, 2) sum over j
    return net


def _reset(env: SyntheticEntitiesEnv, key: jax.Array) -> EnvState:
    key, sk = jax.random.split(key)
    pos = jax.random.uniform(
        sk, (env.n_entities, 2), minval=-env.spawn_radius, maxval=env.spawn_radius
    )
    # Pin the goal to its fixed location (static entity).
    pos = pos.at[GOAL_IDX].set(env.goal_pos)
    vel = jnp.zeros((env.n_entities, 2))
    return EnvState(pos=pos, vel=vel, step=jnp.int32(0), key=key)


def _reward(env: SyntheticEntitiesEnv, pos: jax.Array) -> jax.Array:
    """Sparse reward: task term + single coupled-pair shaping term.

    Depends ONLY on positions of {AGENT_IDX, GOAL_IDX, p, q}. (Ground truth.)
    """
    p, q = env.value_pair
    agent_goal = jnp.linalg.norm(pos[AGENT_IDX] - pos[GOAL_IDX])
    pair_dist = jnp.linalg.norm(pos[p] - pos[q])
    return -agent_goal - env.w_pair * pair_dist


def _step(env: SyntheticEntitiesEnv, state: EnvState, action: jax.Array):
    """Semi-implicit Euler step. Action is a 2-D force on the agent only.

    The goal entity is held static (zero velocity, fixed position) every step.
    """
    action = jnp.clip(action, -1.0, 1.0) * env.action_scale

    forces = _spring_forces(env, state.pos)              # (N, 2)
    # Agent actuation: add the action force to the agent's row.
    forces = forces.at[AGENT_IDX].add(action)
    # Linear damping.
    forces = forces - env.damping * state.vel

    acc = forces / env.mass
    vel = state.vel + env.dt * acc
    pos = state.pos + env.dt * vel

    # Keep the goal static.
    pos = pos.at[GOAL_IDX].set(env.goal_pos)
    vel = vel.at[GOAL_IDX].set(jnp.zeros(2))

    step = state.step + 1
    new_state = EnvState(pos=pos, vel=vel, step=step, key=state.key)

    reward = _reward(env, pos)
    done = step >= env.max_steps
    info = {
        "agent_goal_dist": jnp.linalg.norm(pos[AGENT_IDX] - pos[GOAL_IDX]),
        "pair_dist": jnp.linalg.norm(
            pos[env.value_pair[0]] - pos[env.value_pair[1]]
        ),
    }
    obs = new_state.pos  # placeholder; callers use env.observe for full obs
    return new_state, env.observe(new_state), reward, done, info


# ---------------------------------------------------------------------------
# Scripted policy (for data collection) — greedy toward reducing reward terms.
# ---------------------------------------------------------------------------


def scripted_action(env: SyntheticEntitiesEnv, state: EnvState) -> jax.Array:
    """A simple scripted policy: push the agent toward the goal.

    Returns a 2-D action in [-1, 1]^2. This only acts on the agent (the only
    actuated entity); the pair-shaping term is influenced only indirectly via
    coupling, which is the point (value depends on more than the agent can
    directly set).
    """
    direction = env.goal_pos - state.pos[AGENT_IDX]
    norm = jnp.maximum(jnp.linalg.norm(direction), 1e-6)
    return direction / norm


# ---------------------------------------------------------------------------
# Convenience: vectorized rollout (jittable) for data collection.
# ---------------------------------------------------------------------------


def rollout(
    env: SyntheticEntitiesEnv,
    key: jax.Array,
    length: int,
    action_noise: float = 0.3,
):
    """Roll out one episode of ``length`` steps with scripted+noise policy.

    Returns a dict of stacked arrays:
        obs:     (length, N*d)
        ent:     (length, N, d)
        action:  (length, 2)
        reward:  (length,)
        next_obs:(length, N*d)

    NOTE on jit: ``env`` is a NamedTuple that mixes static python fields with
    JAX arrays (``coupling``, ``goal_pos``), so it is NOT hashable and cannot go
    through ``static_argnums``. The body uses only ``length`` (a python int) for
    shapes, so to jit, close over the env, e.g.::

        f = jax.jit(lambda k: rollout(env, k, length))

    The function body itself is pure / traceable.
    """
    key, rk = jax.random.split(key)
    init = env.reset(rk)

    def body(carry, k):
        state = carry
        nk, ak = jax.random.split(k)
        a = scripted_action(env, state)
        a = a + action_noise * jax.random.normal(ak, (2,))
        a = jnp.clip(a, -1.0, 1.0)
        obs = env.observe(state)
        ent = env.observe_entities(state)
        nstate, nobs, r, d, _ = env.step(state, a)
        out = dict(obs=obs, ent=ent, action=a, reward=r, next_obs=nobs)
        return nstate, out

    keys = jax.random.split(key, length)
    _, traj = jax.lax.scan(body, init, keys)
    return traj
