"""Contact-rich multi-entity world (pure JAX, jittable) with a GROUND-TRUTH
CONTACT GRAPH per step.

============================================================================
WHY THIS EXISTS (and how it differs from synthetic_entities)
============================================================================
``helios.envs.synthetic_entities`` is a SPRING world: the interaction structure
is a *fixed, smooth, always-on* coupling matrix C. Springs are exactly the kind
of dense, continuous coupling that a monolithic MLP can fit just fine — there is
no event-like relational structure for a graph/entity-factored latent to exploit.

This module is the CONTACT world the "Graph World Model as Simulator" mechanism
check actually needs. The GWM survey (arXiv 2604.27895) argues a graph/entity
latent should help precisely where dynamics are *relational and event-like*:
hard contacts. So here the only interaction is ELASTIC COLLISION:

  * N movable disks of radius ``r`` live in a 2-D box with WALLS.
  * Disks fly ballistically (constant velocity + light global damping) UNTIL
    they touch — then an impulse resolves the collision (elastic, equal mass).
  * Wall contacts reflect the normal velocity component (elastic).
  * One disk (index 0) is the AGENT, actuated by the 2-D action (a force).
  * One disk (index 1) is the GOAL region (a static target location; it does NOT
    participate in collision — it is a region, not a body).
  * Reward is SPARSE: push a designated "target" disk into the goal region.

Contacts are SPARSE and EVENT-LIKE: on a typical step most disk pairs are not
touching, so the contact graph is mostly empty; collisions happen only at the
instants disks meet. This is the relational signal springs lacked, and the key
signal the mechanism-check probe conditions on.

============================================================================
COLLISION MODEL (read this; it defines the ground-truth contact graph)
============================================================================
One step = (1) integrate forces -> velocity, (2) move, (3) resolve contacts via
impulses, (4) clamp goal static. Concretely, per step:

  forces      = action on agent  -  damping * vel              (no springs!)
  vel         = vel + dt * forces / mass
  pos         = pos + dt * vel

  DISK-DISK elastic collision (equal mass, 1 impulse pass):
    For every ordered pair (i, j), i<j, both MOVABLE (not the goal region):
      overlap_ij = (dist_ij < 2*r)  AND  (approaching: (v_i - v_j)·n_ij < 0)
      where n_ij = unit(pos_i - pos_j).
      If in contact, apply the equal-mass elastic impulse along n_ij:
        v_i' = v_i - ((v_i - v_j)·n_ij) n_ij
        v_j' = v_j + ((v_i - v_j)·n_ij) n_ij
      (this is the standard 1-D-along-normal elastic swap of the normal
      component; tangential component is preserved). Impulses for all pairs are
      summed (a single Jacobi-style pass, vectorized over the N×N pair grid).
      Positions are then separated to remove residual overlap (push each disk
      out by half the penetration along n_ij) so disks DO NOT overlap after the
      step.

  WALL collision (elastic): if a disk center is within ``r`` of a box wall, its
    position is clamped to ``r`` inside the wall and the normal velocity
    component is flipped (reflected). Box is [-box, box]^2.

  GOAL region (index 1): held static every step (fixed position, zero velocity),
    and EXCLUDED from disk-disk collision (it is a target zone, not a body).

GROUND-TRUTH CONTACT GRAPH (the key exposed signal):
  ``info["contact_graph"]`` is an (N, N) symmetric 0/1 matrix (float32), zero
  diagonal, where ``contact_graph[i, j] == 1`` iff movable disks i and j were in
  CONTACT this step — i.e. ``dist_ij < 2*r`` AND they were approaching
  (``(v_i - v_j)·n_ij < 0``) at the moment of resolution (so an impulse was
  applied between them this step). The goal-region index 1 is never in contact
  (its row/col is always zero). ``info["any_contact"]`` is a scalar 0/1 = whether
  ANY disk-disk contact occurred this step. ``info["wall_contact"]`` is an (N,)
  0/1 vector of per-disk wall contacts. The mechanism-check splits prediction
  error by ``any_contact`` (contact vs non-contact timesteps).

  This contact graph is the relational ground truth the survey says matters: it
  is sparse, event-like, and (for OOD) it gets DENSER as N grows while the
  reward stays sparse — the same asymmetry synthetic_entities was built around,
  but now carried by genuine contacts instead of always-on springs.

============================================================================
REWARD (value-level ground truth — sparse subset, same role indices as springs)
============================================================================
    r = - dist(target_disk, goal_region)          # task term (push target in)
        - w_agent * dist(agent, target_disk)       # shaping: agent near target
        + bonus * 1[dist(target_disk, goal) < goal_radius]   # sparse success

  The reward reads ONLY {agent=0, goal=1, target_disk=t}. ``t`` is chosen
  deterministically from the seed in the "core" range (index 2) so it exists at
  the smallest supported N (>=4 here, kept consistent with the spring world) and
  is the SAME logical slot for every N. Every other disk is value-irrelevant: it
  may collide, it may be in the contact graph, but the reward never reads it.
  So dynamics get denser with N (more contacts) while value stays sparse — the
  intended asymmetry, now carried by contacts.

============================================================================
N-SCALING (OOD object-count generalization)
============================================================================
Holding ``seed`` fixed and growing N keeps the SAME physics (same radius r, same
box, same dt/mass/damping), the same roles for indices 0/1/2, and just APPENDS
extra movable distractor disks. The collision rule is N-agnostic, so a model
trained at ``N_train`` and evaluated at ``N=N_train+2, +4`` faces identical
physics on the shared disks plus novel distractors (and a denser contact graph).

============================================================================
API (mirrors synthetic_entities)
============================================================================
    env = make_env(n_entities=5, seed=0)
    state = env.reset(jax.random.PRNGKey(0))          # EnvState (pytree)
    obs = env.observe(state)                            # (N*d,) flat
    ent = env.observe_entities(state)                  # (N, d) per-entity
    state, obs, reward, done, info = env.step(state, action)
    info["contact_graph"]  # (N, N) 0/1 ground-truth contact graph this step

All of reset/step/observe are pure functions, jax.jit / jax.vmap friendly; N and
all shapes are static (baked into the frozen NamedTuple config).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Per-entity state dimensionality: (px, py, vx, vy).
ENTITY_DIM = 4

# Fixed role indices (stable across all N).
AGENT_IDX = 0
GOAL_IDX = 1          # the goal REGION (static target; not a colliding body)
TARGET_IDX = 2        # the disk the reward wants pushed into the goal region


class EnvState(NamedTuple):
    """Dynamic environment state (a JAX pytree, jit/vmap friendly)."""

    pos: jax.Array      # (N, 2)
    vel: jax.Array      # (N, 2)
    step: jax.Array     # scalar int32
    key: jax.Array      # PRNG key


class ContactEntitiesEnv(NamedTuple):
    """Static config + ground truth for the contact-rich entity world.

    Ground-truth fields (for probe / selftest):
        value_target:            index t of the disk the reward reads (TARGET_IDX).
        value_relevant_entities: sorted tuple = (AGENT_IDX, GOAL_IDX, t).
    """

    # --- static dims / physics ---
    n_entities: int
    dt: float
    mass: float
    damping: float
    radius: float
    box: float
    action_scale: float
    max_steps: int
    # --- reward ---
    w_agent: float
    goal_radius: float
    success_bonus: float
    # --- ground truth ---
    value_target: int
    value_relevant_entities: tuple
    # --- reset distribution ---
    spawn_radius: float
    goal_pos: jax.Array          # (2,) fixed goal-region location
    init_speed: float            # disks spawn with random velocity of this speed

    # ------------------------------------------------------------------
    # gym-like API
    # ------------------------------------------------------------------
    def reset(self, key: jax.Array) -> EnvState:
        return _reset(self, key)

    def step(self, state: EnvState, action: jax.Array):
        return _step(self, state, action)

    def observe(self, state: EnvState) -> jax.Array:
        return self.observe_entities(state).reshape(-1)

    def observe_entities(self, state: EnvState) -> jax.Array:
        return jnp.concatenate([state.pos, state.vel], axis=-1)

    @property
    def obs_dim(self) -> int:
        return self.n_entities * ENTITY_DIM

    @property
    def action_dim(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------


def make_env(
    n_entities: int = 5,
    seed: int = 0,
    *,
    dt: float = 0.05,
    mass: float = 1.0,
    damping: float = 0.05,
    radius: float = 0.35,
    box: float = 2.5,
    action_scale: float = 2.0,
    max_steps: int = 100,
    w_agent: float = 0.5,
    goal_radius: float = 0.5,
    success_bonus: float = 1.0,
    spawn_radius: float = 2.0,
    init_speed: float = 1.0,
) -> ContactEntitiesEnv:
    """Construct a contact-rich entity env with a sparse, known reward.

    Args:
        n_entities: number of disks N (>= 4 for consistency with the spring
            world; >= 3 is the structural minimum: agent, goal, target). 5 is the
            canonical N_train; N_train+2, +4 are the OOD object-count splits.
        seed: controls the value target slot (kept in the core range) and the
            reset RNG default; physics are identical across seeds. ``seed`` does
            NOT change collision rules, so OOD = same rules, more objects.
        dt, mass, damping: semi-implicit Euler params (light global damping so the
            box does not run away; collisions are the dominant interaction).
        radius: disk radius r; disks collide at center distance < 2*r.
        box: half-width of the square box [-box, box]^2 (walls).
        action_scale: multiplies the clipped 2-D action -> agent force.
        max_steps: episode horizon (done flag).
        w_agent: weight of the agent->target shaping term.
        goal_radius: radius of the goal region for the success bonus.
        success_bonus: reward bonus when the target disk is inside the goal.
        spawn_radius: reset positions ~ U([-r,r]^2) (then de-overlapped/clamped).
        init_speed: magnitude of random initial velocity given to movable disks
            (drives contacts early; the goal region stays at rest).

    Returns:
        ContactEntitiesEnv (immutable config + ground truth).
    """
    if n_entities < 4:
        raise ValueError(
            "n_entities must be >= 4 (agent, goal region, target, + >=1 "
            f"distractor for a meaningful contact OOD axis); got {n_entities}"
        )
    # Value target is the SAME logical slot for every N (so value structure is
    # identical on the shared disks across object-count OOD splits) and lives in
    # the core range. TARGET_IDX (=2) satisfies this and exists for n>=4.
    t = TARGET_IDX
    value_relevant = tuple(sorted({AGENT_IDX, GOAL_IDX, t}))
    goal_pos = jnp.array([1.5, 1.5])
    return ContactEntitiesEnv(
        n_entities=n_entities,
        dt=dt,
        mass=mass,
        damping=damping,
        radius=radius,
        box=box,
        action_scale=action_scale,
        max_steps=max_steps,
        w_agent=w_agent,
        goal_radius=goal_radius,
        success_bonus=success_bonus,
        value_target=int(t),
        value_relevant_entities=value_relevant,
        spawn_radius=spawn_radius,
        goal_pos=goal_pos,
        init_speed=init_speed,
    )


# ---------------------------------------------------------------------------
# Helpers: movable mask (everything except the goal region)
# ---------------------------------------------------------------------------


def _movable_mask(env: ContactEntitiesEnv) -> jax.Array:
    """(N,) float mask: 1 for movable disks, 0 for the static goal region."""
    idx = jnp.arange(env.n_entities)
    return (idx != GOAL_IDX).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Dynamics (pure functions; jit/vmap friendly)
# ---------------------------------------------------------------------------


def _reset(env: ContactEntitiesEnv, key: jax.Array) -> EnvState:
    key, pk, vk = jax.random.split(key, 3)
    pos = jax.random.uniform(
        pk, (env.n_entities, 2), minval=-env.spawn_radius, maxval=env.spawn_radius
    )
    # Random initial velocities (drive contacts); goal region pinned & at rest.
    vdir = jax.random.normal(vk, (env.n_entities, 2))
    vnorm = jnp.maximum(jnp.linalg.norm(vdir, axis=-1, keepdims=True), 1e-6)
    vel = env.init_speed * vdir / vnorm
    pos = pos.at[GOAL_IDX].set(env.goal_pos)
    vel = vel.at[GOAL_IDX].set(jnp.zeros(2))
    # Clamp inside walls at reset so the first step starts feasible.
    lim = env.box - env.radius
    pos = jnp.clip(pos, -lim, lim)
    pos = pos.at[GOAL_IDX].set(env.goal_pos)
    return EnvState(pos=pos, vel=vel, step=jnp.int32(0), key=key)


def _contact_graph(env: ContactEntitiesEnv, pos: jax.Array, vel: jax.Array):
    """Compute the ground-truth disk-disk contact graph for the CURRENT pos/vel.

    Returns:
        contact: (N, N) symmetric 0/1 float, zero diagonal. contact[i,j]==1 iff
                 movable disks i,j overlap (dist < 2*r) AND are approaching.
        n_ij:    (N, N, 2) collision normals (unit(pos_i - pos_j)).
        approach:(N, N) approaching-speed-along-normal magnitude (>=0 where
                 contact, used by the impulse pass).
        dist:    (N, N) pairwise center distances.
    """
    N = env.n_entities
    move = _movable_mask(env)                                  # (N,)
    pair_movable = move[:, None] * move[None, :]               # (N,N) both movable

    delta = pos[:, None, :] - pos[None, :, :]                  # (N,N,2)=pos_i-pos_j
    dist = jnp.linalg.norm(delta, axis=-1)                     # (N,N)
    safe = jnp.maximum(dist, 1e-6)
    n_ij = delta / safe[..., None]                             # unit(pos_i-pos_j)

    rel_v = vel[:, None, :] - vel[None, :, :]                  # v_i - v_j (N,N,2)
    vn = jnp.sum(rel_v * n_ij, axis=-1)                        # (N,N) closing<0

    eye = jnp.eye(N)
    overlap = (dist < 2.0 * env.radius).astype(jnp.float32) * (1.0 - eye)
    approaching = (vn < 0.0).astype(jnp.float32)
    contact = overlap * approaching * pair_movable             # (N,N) 0/1
    contact = jnp.maximum(contact, contact.T)                  # symmetric
    return contact, n_ij, vn, dist


def _resolve_contacts(env: ContactEntitiesEnv, pos: jax.Array, vel: jax.Array):
    """One Jacobi-style elastic impulse pass + positional de-overlap.

    Returns (new_vel, new_pos, contact_graph).
    """
    contact, n_ij, vn, dist = _contact_graph(env, pos, vel)

    # Equal-mass elastic impulse along the normal: each contacting pair (i,j)
    # removes the closing normal component from v_i and adds it to v_j.
    #   dv_i = - (vn_ij) * n_ij      (only where contact)
    # Summed over all j (Jacobi pass). vn<0 for contacts so this slows closing.
    impulse = -(contact * vn)[..., None] * n_ij                # (N,N,2)
    dv = jnp.sum(impulse, axis=1)                              # (N,2)
    new_vel = vel + dv

    # Positional de-overlap: push disk i out of disk j by half the penetration
    # along n_ij, summed over contacts. Keeps disks from overlapping post-step.
    penetration = jnp.maximum(2.0 * env.radius - dist, 0.0)    # (N,N) >=0
    sep = (contact * 0.5 * penetration)[..., None] * n_ij      # (N,N,2)
    dpos = jnp.sum(sep, axis=1)                                # (N,2)
    new_pos = pos + dpos
    return new_vel, new_pos, contact


def _resolve_walls(env: ContactEntitiesEnv, pos: jax.Array, vel: jax.Array):
    """Elastic wall collisions: clamp center to r inside the box; reflect normal
    velocity component. Returns (new_pos, new_vel, wall_contact (N,))."""
    lim = env.box - env.radius
    below = pos < -lim
    above = pos > lim
    hit = jnp.logical_or(below, above)                         # (N,2) per axis
    new_pos = jnp.clip(pos, -lim, lim)
    # Flip velocity on axes that hit a wall (elastic reflection).
    new_vel = jnp.where(hit, -vel, vel)
    wall_contact = jnp.any(hit, axis=-1).astype(jnp.float32)   # (N,)
    return new_pos, new_vel, wall_contact


def _reward(env: ContactEntitiesEnv, pos: jax.Array) -> jax.Array:
    """Sparse reward reading ONLY {agent, goal region, target disk}."""
    t = env.value_target
    target_goal = jnp.linalg.norm(pos[t] - env.goal_pos)
    agent_target = jnp.linalg.norm(pos[AGENT_IDX] - pos[t])
    success = (target_goal < env.goal_radius).astype(jnp.float32)
    return -target_goal - env.w_agent * agent_target + env.success_bonus * success


def _step(env: ContactEntitiesEnv, state: EnvState, action: jax.Array):
    """One physics step: integrate, collide (disk-disk then walls), reward."""
    action = jnp.clip(action, -1.0, 1.0) * env.action_scale

    move = _movable_mask(env)[:, None]                         # (N,1)
    # Forces: only the agent is actuated; light global damping. No springs.
    forces = jnp.zeros_like(state.pos)
    forces = forces.at[AGENT_IDX].add(action)
    forces = forces - env.damping * state.vel
    forces = forces * move                                     # goal feels nothing

    acc = forces / env.mass
    vel = state.vel + env.dt * acc
    pos = state.pos + env.dt * vel * move                      # goal does not drift

    # Disk-disk elastic collisions (records the ground-truth contact graph).
    vel, pos, contact = _resolve_contacts(env, pos, vel)
    # Wall collisions (elastic reflection).
    pos, vel, wall_contact = _resolve_walls(env, pos, vel)

    # Keep the goal region static.
    pos = pos.at[GOAL_IDX].set(env.goal_pos)
    vel = vel.at[GOAL_IDX].set(jnp.zeros(2))

    step = state.step + 1
    new_state = EnvState(pos=pos, vel=vel, step=step, key=state.key)

    reward = _reward(env, pos)
    done = step >= env.max_steps
    info = {
        "contact_graph": contact,                              # (N,N) 0/1 GT graph
        "any_contact": (jnp.sum(contact) > 0.0).astype(jnp.float32),
        "wall_contact": wall_contact,                          # (N,) 0/1
        "n_contacts": jnp.sum(contact) / 2.0,                  # # of contacting pairs
        "target_goal_dist": jnp.linalg.norm(pos[env.value_target] - env.goal_pos),
        "agent_target_dist": jnp.linalg.norm(
            pos[AGENT_IDX] - pos[env.value_target]
        ),
    }
    return new_state, env.observe(new_state), reward, done, info


# ---------------------------------------------------------------------------
# Scripted policy (for data collection): push agent toward the target disk so it
# herds the target into the goal (generates contacts AND value-relevant motion).
# ---------------------------------------------------------------------------


def scripted_action(env: ContactEntitiesEnv, state: ContactEntitiesEnv) -> jax.Array:
    """Push the agent toward the target disk (so the agent collides with it and
    pushes it toward the goal). Returns a 2-D action in [-1, 1]^2."""
    t = env.value_target
    direction = state.pos[t] - state.pos[AGENT_IDX]
    norm = jnp.maximum(jnp.linalg.norm(direction), 1e-6)
    return direction / norm


# ---------------------------------------------------------------------------
# Vectorized rollout (jittable) for data collection.
# ---------------------------------------------------------------------------


def rollout(
    env: ContactEntitiesEnv,
    key: jax.Array,
    length: int,
    action_noise: float = 0.4,
):
    """Roll out one episode of ``length`` steps with scripted+noise policy.

    Returns a dict of stacked arrays:
        obs:           (length, N*d)
        ent:           (length, N, d)
        action:        (length, 2)
        reward:        (length,)
        next_obs:      (length, N*d)
        contact_graph: (length, N, N) ground-truth per-step contact graph
        any_contact:   (length,) 0/1 whether any disk-disk contact this step

    To jit, close over the env, e.g. ``jax.jit(lambda k: rollout(env, k, L))``.
    """
    key, rk = jax.random.split(key)
    init = env.reset(rk)

    def body(carry, k):
        state = carry
        ak = k
        a = scripted_action(env, state)
        a = a + action_noise * jax.random.normal(ak, (2,))
        a = jnp.clip(a, -1.0, 1.0)
        obs = env.observe(state)
        ent = env.observe_entities(state)
        nstate, nobs, r, d, info = env.step(state, a)
        out = dict(
            obs=obs,
            ent=ent,
            action=a,
            reward=r,
            next_obs=nobs,
            contact_graph=info["contact_graph"],
            any_contact=info["any_contact"],
        )
        return nstate, out

    keys = jax.random.split(key, length)
    _, traj = jax.lax.scan(body, init, keys)
    return traj
