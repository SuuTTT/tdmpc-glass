"""CompPlan plan-over-policies planner (eval-only, additive).

Implements gap (c) from docs/ghm_repro/EXTENSION_DESIGN.md: given a start observation and a goal,
search over short SEQUENCES of base policies using the trained Geometric Horizon Model (GHM) to
*jump* to predicted future states (samples of the discounted future-state / occupancy measure),
score candidates by predicted proximity to the goal, and return the first action toward the goal
(receding-horizon / MPC-style re-planning).

This module does NOT train and is NOT imported by main.py's train step, so it cannot affect the
training smoke. It only calls a frozen GHM agent's `compute_fwd_flow_goals` (the jump) and base
policies' action proposers.

Algorithm (beam / sampling over policy sequences), faithful to the design doc:

    inputs: s0, goal g, policy library {pi_1..pi_K}, jump discount gamma_h,
            plan depth L, beam width B, subgoal samples M, scorer d(s, g)
    beam <- [ (s0, [], 0.0) ]
    for step in 1..L:
        for (s, seq, sc) in beam:
            for k in 1..K:
                a_cond <- pi_k(s)
                jump M future states ~ m^{pi_k}_{gamma_h}( . | s, a_cond ) via the GHM
                push each jumped state as a candidate
        beam <- top-B candidates by -d(s_next, g)
    return best sequence; execute pi_{k_1}'s action at s0, then re-plan.

A "base policy" here is any callable `pi(obs_batch, *, seed) -> action_batch`. The GHM agent's own
DDPG+BC actor is wrapped as the first base policy by `actor_base_policy(agent)`.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def actor_base_policy(agent, temperature=0.0):
    """Wrap a GHM/InFOM agent's actor as a base policy callable.

    Returns a function `pi(observations, seed) -> actions` (deterministic if temperature==0).
    """
    def pi(observations, seed):
        return agent.sample_actions(observations=observations, seed=seed, temperature=temperature)
    return pi


def goal_conditioned_base_policy(agent, goal, temperature=0.0):
    """Wrap an agent as a goal-conditioned base policy by concatenating the goal to the obs.

    Many OGBench goal-conditioned policies take (obs, goal). The GHM actor here is unconditional, so
    by default we ignore the goal (the GHM's actor is the behavior-cloned policy). This helper is a
    placeholder hook for plugging a real GC policy library; for the smoke we use `actor_base_policy`.
    """
    def pi(observations, seed):
        return agent.sample_actions(observations=observations, seed=seed, temperature=temperature)
    return pi


class CompPlanPlanner:
    """Plan over a library of base policies using GHM jumps.

    Args:
        agent: a trained GHM agent (must expose `compute_fwd_flow_goals`, `config`).
        base_policies: list of callables `pi(observations, seed) -> actions`.
        gamma_h: jump discount (geometric horizon length of one policy segment). If None, uses the
            agent's `gamma_max` (longest horizon the model was trained on).
        plan_depth: number of policy segments to chain (L).
        beam_width: beam size (B).
        num_subgoal_samples: GHM future-state samples per (state, policy) expansion (M).
        scorer: 'l2' (negative L2 distance to goal in obs space) or 'reward' (learned reward head).
        latent_type: 'prior' (sample z~N(0,I)) or 'zeros'. Matches how the GHM was conditioned.
        observation_min / observation_max: bounds for flow-goal clipping (optional).
    """

    def __init__(self, agent, base_policies, gamma_h=None, plan_depth=3, beam_width=8,
                 num_subgoal_samples=64, scorer='l2', latent_type='prior',
                 observation_min=None, observation_max=None):
        self.agent = agent
        self.base_policies = list(base_policies)
        assert len(self.base_policies) >= 1, 'need at least one base policy'
        cfg = agent.config
        self.gamma_h = float(cfg['gamma_max']) if gamma_h is None else float(gamma_h)
        self.plan_depth = int(plan_depth)
        self.beam_width = int(beam_width)
        self.M = int(num_subgoal_samples)
        self.scorer = scorer
        self.latent_type = latent_type
        self.latent_dim = int(cfg['latent_dim'])
        self.obs_min = observation_min
        self.obs_max = observation_max

    # -------------------------------------------------------------------- GHM jump (M samples) --
    def _jump(self, state, action, rng):
        """Sample M future states ~ m^{pi}_{gamma_h}( . | state, action ) via the GHM.

        state, action: 1-D arrays (single state/action). Returns (M, obs_dim) jumped states.
        """
        obs_dim = state.shape[-1]
        noise_rng, latent_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, shape=(self.M, obs_dim), dtype=state.dtype)
        obs_rep = jnp.broadcast_to(state[None], (self.M, obs_dim))
        act_rep = jnp.broadcast_to(action[None], (self.M, action.shape[-1]))
        if self.latent_type == 'prior':
            latents = jax.random.normal(latent_rng, shape=(self.M, self.latent_dim), dtype=state.dtype)
        else:
            latents = jnp.zeros((self.M, self.latent_dim), dtype=state.dtype)
        horizons = jnp.full((self.M,), self.gamma_h, dtype=state.dtype)
        kw = {}
        # `compute_fwd_flow_goals` clips to [obs_min, obs_max] when the agent's `clip_flow_goals` is
        # True. Always supply bounds so it never hits a None; default to wide bounds (effectively no
        # clip) when the caller did not pass dataset bounds.
        if self.obs_min is not None and self.obs_max is not None:
            kw['observation_min'] = self.obs_min
            kw['observation_max'] = self.obs_max
        elif self.agent.config['clip_flow_goals']:
            kw['observation_min'] = jnp.full((obs_dim,), -1e6, dtype=state.dtype)
            kw['observation_max'] = jnp.full((obs_dim,), 1e6, dtype=state.dtype)
        future_states = self.agent.compute_fwd_flow_goals(
            noises, obs_rep, act_rep, latents, horizons=horizons,
            use_target_network=False, **kw)
        return future_states  # (M, obs_dim)

    # ------------------------------------------------------------------------------- scoring --
    def _score(self, states, goal):
        """Score candidate states by proximity to goal. Higher is better."""
        if self.scorer == 'reward':
            # Learned reward head as a goal-proximity proxy (reward was trained for the task goal).
            r = self.agent.network.select('reward')(states)
            return np.asarray(r).reshape(-1)
        # default: negative L2 distance in (normalized) observation space.
        d = jnp.linalg.norm(states - goal[None], axis=-1)
        return np.asarray(-d)

    # ------------------------------------------------------------------------------ planning --
    def plan(self, observation, goal, seed=0):
        """Run one receding-horizon plan from `observation` toward `goal`.

        Returns a dict with the chosen first action, the best policy sequence, and the predicted
        terminal state / score (for logging). The caller executes `action`, steps the env, and
        calls `plan` again at the next state (MPC).
        """
        observation = jnp.asarray(observation)
        goal = jnp.asarray(goal)
        rng = jax.random.PRNGKey(seed)

        # Beam entries: (predicted_state, policy_index_seq, cumulative_score, first_action)
        beam = [(observation, [], 0.0, None)]

        for depth in range(self.plan_depth):
            candidates = []
            for (state, seq, _sc, first_action) in beam:
                for k, pi in enumerate(self.base_policies):
                    rng, act_rng, jump_rng = jax.random.split(rng, 3)
                    a_cond = pi(state[None], seed=act_rng)[0]  # action at this state
                    a_cond = jnp.clip(jnp.asarray(a_cond), -1, 1)
                    fa = first_action if first_action is not None else (k, np.asarray(a_cond))
                    future_states = self._jump(state, a_cond, jump_rng)  # (M, obs_dim)
                    scores = self._score(future_states, goal)  # (M,)
                    best_m = int(np.argmax(scores))
                    s_next = future_states[best_m]
                    candidates.append((s_next, seq + [k], float(scores[best_m]), fa))
            # Keep top-B by score.
            candidates.sort(key=lambda c: c[2], reverse=True)
            beam = candidates[: self.beam_width]

        best = beam[0]
        best_state, best_seq, best_score, best_first = best
        first_k, first_action = best_first
        return dict(
            action=np.asarray(first_action),
            policy_index=first_k,
            policy_sequence=best_seq,
            predicted_terminal_state=np.asarray(best_state),
            score=best_score,
        )
