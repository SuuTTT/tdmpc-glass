"""GHM (Geometric Horizon Model) agent — CompPlan-style jumpy world model.

This is a copy of `agents/infom.py` (InFOM flow-occupancy agent) extended toward the
Geometric Horizon Model of "Compositional Planning with Jumpy World Models" (CompPlan,
arXiv 2602.19634), built on Temporal-Difference Flows (TD-Flow, arXiv 2503.09817).

Two extensions over InFOM (see docs/ghm_repro/EXTENSION_DESIGN.md):
  (a) HORIZON-conditioning: instead of one fixed `discount`, a per-example geometric-horizon
      discount `gamma_h` is sampled per batch and fed to the velocity field, and the discounted-TD
      flow-matching recursion is made horizon-consistent (same `gamma_h` on both sides). One model
      then answers every horizon. A `discount_consistency_frac` masks the bootstrap term on a subset
      of the batch (paper uses 25% antmaze / 12.5% cube); default 1.0 recovers plain InFOM.
  (b) POLICY-conditioning HOOK: `flow_occupancy_loss` accepts optional `cond_actions`/`cond_next_actions`
      so the future-state measure can be modeled UNDER A GIVEN base policy (relabel the conditioning
      action with pi_k(s)). If absent it falls back to dataset actions == plain InFOM. An optional
      policy-embedding channel is wired in `create` but defaults off.

All defaults are chosen so GHM reduces to InFOM behavior, so existing main.py runs unchanged via
`--agent=agents/ghm.py`. The flow-matching + TD-bootstrap + target-network structure is preserved.
"""
import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import HorizonVectorField, Actor, IntentionEncoder, Value


class GHMAgent(flax.struct.PyTreeNode):
    """Geometric Horizon Model (GHM) agent: horizon- and policy-conditioned flow-occupancy model."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    # ------------------------------------------------------------------ horizon sampling (NEW) --
    def sample_horizons(self, rng, shape, dtype):
        """Sample per-example geometric-horizon discounts gamma_h.

        Log-uniform over (1 - gamma) so horizons h ~ 1/(1-gamma_h) are log-uniform between
        ~1/(1-gamma_min) and ~1/(1-gamma_max). With gamma_min == gamma_max this returns the fixed
        InFOM discount (degenerate, recovers InFOM exactly).
        """
        gmin = self.config['gamma_min']
        gmax = self.config['gamma_max']
        if gmin >= gmax:
            return jnp.full(shape, gmax, dtype=dtype)
        # u in [log(1-gmax), log(1-gmin)] -> gamma_h = 1 - exp(u)  (smaller 1-gamma => larger horizon)
        lo = jnp.log(1.0 - gmax)
        hi = jnp.log(1.0 - gmin)
        u = jax.random.uniform(rng, shape=shape, dtype=dtype, minval=lo, maxval=hi)
        return 1.0 - jnp.exp(u)

    def reward_loss(self, batch, grad_params):
        observations = batch['observations']
        rewards = batch['rewards']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)
        reward_preds = self.network.select('reward')(
            observations, params=grad_params,
        )

        reward_loss = jnp.square(reward_preds - rewards).mean()

        return reward_loss, {
            'reward_loss': reward_loss
        }

    def critic_loss(self, batch, grad_params, rng):
        """Compute the value loss.

        Uses a FIXED evaluation horizon (config['discount']) for the Q estimate, matching InFOM's
        downstream value semantics; the horizon-conditioned field is queried at that gamma.
        """
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        next_actions = batch['next_actions']

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(observations)

        rng, noise_rng, latent_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(
            noise_rng,
            shape=(self.config['num_flow_goals'], *observations.shape),
            dtype=observations.dtype
        )
        if self.config['critic_latent_type'] == 'prior':
            latents = jax.random.normal(
                latent_rng,
                shape=(*actions.shape[:-1], self.config['latent_dim']),
                dtype=observations.dtype,
            )
            latents = jnp.broadcast_to(
                latents[None],
                (self.config['num_flow_goals'], *latents.shape)
            )
        elif self.config['critic_latent_type'] == 'encoding':
            latent_dist = self.network.select('intention_encoder')(next_observations, next_actions)
            latents = latent_dist.sample(seed=latent_rng)
            latents = jnp.broadcast_to(
                latents[None],
                (self.config['num_flow_goals'], *latents.shape)
            )
        # Evaluate at the fixed downstream horizon (config['discount']).
        eval_gamma = jnp.full(
            (self.config['num_flow_goals'], *observations.shape[:-1]),
            self.config['discount'], dtype=observations.dtype)
        flow_goals = self.compute_fwd_flow_goals(
            noises,
            jnp.broadcast_to(
                observations[None],
                (self.config['num_flow_goals'], *observations.shape)
            ),
            jnp.broadcast_to(
                actions[None],
                (self.config['num_flow_goals'], *actions.shape)
            ),
            latents,
            horizons=eval_gamma,
            observation_min=batch.get('observation_min', None),
            observation_max=batch.get('observation_max', None),
        )

        future_rewards = self.network.select('reward')(flow_goals)
        target_q = 1.0 / (1 - self.config['discount']) * future_rewards.mean(axis=0)
        qs = self.network.select('critic')(batch['observations'], actions, params=grad_params)
        critic_loss = self.expectile_loss(target_q - qs, target_q - qs, self.config['expectile']).mean()

        # For logging
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def flow_occupancy_loss(self, batch, grad_params, rng):
        """Compute the horizon-conditioned flow occupancy loss (GHM core).

        Differences vs InFOM:
          * `gamma_h` is sampled per example (sample_horizons) and fed to the velocity field.
          * The (1-gamma)/gamma mixture weights use this per-example gamma_h.
          * Bootstrap target field is conditioned on the SAME gamma_h (horizon-consistent TD-Flow).
          * The bootstrap (future) term is masked on a (1 - discount_consistency_frac) subset.
          * POLICY-conditioning hook: conditioning actions come from batch['cond_actions'] /
            batch['cond_next_actions'] when present (a base policy's actions), else dataset actions.
        """

        batch_size = batch['observations'].shape[0]
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        next_actions = batch['next_actions']

        # --- POLICY-conditioning hook (NEW): relabel conditioning actions with a base policy's. ---
        # The flow-matching DATA target is still the dataset transition; only the *conditioning*
        # action fed to the field is (optionally) replaced, so the model learns m^{pi}(.|s, pi(s)).
        cond_actions = batch.get('cond_actions', actions)
        cond_next_actions = batch.get('cond_next_actions', next_actions)

        if self.config['encoder'] is not None:
            observations = self.network.select('critic_vf_encoder')(
                batch['observations'], params=grad_params)
            next_observations = self.network.select('target_critic_vf_encoder')(
                batch['next_observations'])

        # infer z using (s', a')  (intention latent, as in InFOM)
        rng, latent_rng = jax.random.split(rng)
        latent_dist = self.network.select('intention_encoder')(
            batch['next_observations'], next_actions, params=grad_params)
        latents = latent_dist.sample(seed=latent_rng)

        means = latent_dist.mean()
        log_stds = jnp.log(latent_dist.stddev())
        kl_loss = -0.5 * (1 + 2 * log_stds - means ** 2 - jnp.exp(2 * log_stds)).mean()

        # --- HORIZON-conditioning (NEW): sample one gamma_h per example. ---
        rng, horizon_rng = jax.random.split(rng)
        gamma_h = self.sample_horizons(horizon_rng, (batch_size,), observations.dtype)  # (B,)

        # SARSA^2 / TD-Flow flow matching for the horizon-conditioned occupancy model.
        rng, time_rng, current_noise_rng, future_noise_rng = jax.random.split(rng, 4)
        times = jax.random.uniform(time_rng, shape=(batch_size,), dtype=observations.dtype)
        current_noises = jax.random.normal(
            current_noise_rng, shape=observations.shape, dtype=observations.dtype)
        current_vf_pred = self.network.select('critic_vf')(
            times[..., None] * observations + (1 - times[..., None]) * current_noises,
            times,
            jax.lax.stop_gradient(observations), cond_actions, latents,
            horizons=gamma_h,
            params=grad_params,
        )
        # stop gradient for the image encoder
        current_flow_matching_loss = jnp.square(
            jax.lax.stop_gradient(observations - current_noises) - current_vf_pred).mean(axis=-1)

        future_noises = jax.random.normal(
            future_noise_rng, shape=observations.shape, dtype=observations.dtype)
        flow_future_observations = self.compute_fwd_flow_goals(
            future_noises, next_observations, cond_next_actions, jax.lax.stop_gradient(latents),
            horizons=gamma_h,
            observation_min=batch.get('observation_min', None),
            observation_max=batch.get('observation_max', None),
            use_target_network=True,
        )
        future_vf_target = self.network.select('target_critic_vf')(
            times[..., None] * flow_future_observations + (1 - times[..., None]) * future_noises,
            times,
            next_observations, cond_next_actions, jax.lax.stop_gradient(latents),
            horizons=gamma_h,
        )
        future_vf_pred = self.network.select('critic_vf')(
            times[..., None] * flow_future_observations + (1 - times[..., None]) * future_noises,
            times,
            jax.lax.stop_gradient(observations), cond_actions, jax.lax.stop_gradient(latents),
            horizons=gamma_h,
            params=grad_params,
        )
        future_flow_matching_loss = jnp.square(future_vf_target - future_vf_pred).mean(axis=-1)

        # --- discount-consistency subset (NEW): mask the bootstrap term on part of the batch. ---
        if self.config['discount_consistency_frac'] < 1.0:
            rng, mask_rng = jax.random.split(rng)
            consistency_mask = (jax.random.uniform(mask_rng, shape=(batch_size,))
                                < self.config['discount_consistency_frac']).astype(observations.dtype)
        else:
            consistency_mask = jnp.ones((batch_size,), dtype=observations.dtype)

        # Per-example horizon-consistent mixture: (1-gamma_h)*current + mask*gamma_h*future.
        flow_matching_loss = ((1 - gamma_h) * current_flow_matching_loss
                              + consistency_mask * gamma_h * future_flow_matching_loss).mean()

        # negative ELBO loss
        neg_elbo_loss = flow_matching_loss + self.config['kl_weight'] * kl_loss

        return neg_elbo_loss, {
            'neg_elbo_loss': neg_elbo_loss,
            'flow_matching_loss': flow_matching_loss,
            'kl_loss': kl_loss,
            'gamma_h_mean': gamma_h.mean(),
            'flow_future_obs_max': flow_future_observations.max(),
            'flow_future_obs_min': flow_future_observations.min(),
            'current_flow_matching_loss': current_flow_matching_loss.mean(),
            'future_flow_matching_loss': future_flow_matching_loss.mean(),
        }

    def behavioral_cloning_loss(self, batch, grad_params):
        """Compute the behavioral cloning loss for pretraining."""
        observations = batch['observations']
        actions = batch['actions']

        dist = self.network.select('actor')(observations, params=grad_params)
        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        return bc_loss, {
            'bc_loss': bc_loss,
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the DDPG + BC actor loss."""

        observations = batch['observations']
        actions = batch['actions']

        # DDPG+BC loss.
        dist = self.network.select('actor')(observations, params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        qs = self.network.select('critic')(observations, actions=q_actions)
        if self.config['q_agg'] == 'mean':
            q = jnp.mean(qs, axis=0)
        else:
            q = jnp.min(qs, axis=0)
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        log_prob = dist.log_prob(actions)
        bc_loss = -(self.config['alpha'] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def compute_fwd_flow_goals(self, noises, observations, actions, latents,
                               horizons=None, policy_embeds=None,
                               observation_min=None, observation_max=None,
                               init_times=None, end_times=None,
                               use_target_network=False):
        """Euler-integrate the (horizon-conditioned) velocity field from noise (t=0) to t=1.

        `horizons` (gamma_h) and `policy_embeds` are passed through to the field at every step so the
        sampled future state respects the requested geometric horizon / base policy. This is also the
        method the planner calls to JUMP to predicted future states.
        """
        if use_target_network:
            module_name = 'target_critic_vf'
        else:
            module_name = 'critic_vf'

        noisy_goals = noises
        if init_times is None:
            init_times = jnp.zeros(noisy_goals.shape[:-1], dtype=noisy_goals.dtype)
        if end_times is None:
            end_times = jnp.ones(noisy_goals.shape[:-1], dtype=noisy_goals.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def body_fn(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_goals, ) = carry

            times = i * step_size + init_times
            vf = self.network.select(module_name)(
                noisy_goals, times, observations, actions, latents,
                horizons=horizons, policy_embeds=policy_embeds)
            new_noisy_goals = noisy_goals + vf * jnp.expand_dims(step_size, axis=-1)
            if self.config['clip_flow_goals']:
                new_noisy_goals = jnp.clip(new_noisy_goals, observation_min + 1e-5, observation_max - 1e-5)

            return (new_noisy_goals,), None

        # Use lax.scan to iterate over num_flow_steps
        (noisy_goals,), _ = jax.lax.scan(
            body_fn, (noisy_goals,), jnp.arange(self.config['num_flow_steps']))

        return noisy_goals

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, flow_occupancy_rng = jax.random.split(rng)

        flow_occupancy_loss, flow_occupancy_info = self.flow_occupancy_loss(
            batch, grad_params, flow_occupancy_rng)
        for k, v in flow_occupancy_info.items():
            info[f'flow_occupancy/{k}'] = v

        bc_loss, bc_info = self.behavioral_cloning_loss(
            batch, grad_params)
        for k, v in bc_info.items():
            info[f'bc/{k}'] = v

        loss = flow_occupancy_loss + bc_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetuning_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng, flow_occupancy_rng, actor_rng = jax.random.split(
            rng, 4)

        reward_loss, reward_info = self.reward_loss(batch, grad_params)
        for k, v in reward_info.items():
            info[f'reward/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        flow_occupancy_loss, flow_occupancy_info = self.flow_occupancy_loss(
            batch, grad_params, flow_occupancy_rng)
        for k, v in flow_occupancy_info.items():
            info[f'flow_occupancy/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = reward_loss + critic_loss + flow_occupancy_loss + actor_loss
        return loss, info

    def target_reset(self):
        params = self.network.params
        if self.config['encoder'] is not None:
            params['modules_target_critic_vf_encoder'] = params['modules_critic_vf_encoder']
        params['modules_target_critic_vf'] = params['modules_critic_vf']

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def pretrain(self, batch):
        """Pre-train the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if self.config['encoder'] is not None:
            self.target_update(new_network, 'critic_vf_encoder')
        self.target_update(new_network, 'critic_vf')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.finetuning_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if self.config['encoder'] is not None:
            self.target_update(new_network, 'critic_vf_encoder')
        self.target_update(new_network, 'critic_vf')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        ex_orig_observations = ex_observations
        ex_times = ex_actions[..., 0]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))
        # Example horizon (gamma_h) input for initialization: shape (B,).
        ex_horizons = jnp.full(ex_actions.shape[:-1], config['gamma_max'], dtype=ex_actions.dtype)
        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        action_dtype = ex_actions.dtype

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            if 'mlp_hidden_dims' in encoder_module.keywords:
                obs_dim = encoder_module.keywords['mlp_hidden_dims'][-1]
            else:
                obs_dim = encoder_modules['impala'].mlp_hidden_dims[-1]
            rng, obs_rng = jax.random.split(rng, 2)
            ex_observations = jax.random.normal(
                obs_rng, shape=(ex_observations.shape[0], obs_dim), dtype=action_dtype)

            encoders['critic'] = encoder_module()
            encoders['critic_vf'] = encoder_module()
            encoders['intention'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define value and actor networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        intention_encoder_def = IntentionEncoder(
            hidden_dims=config['intention_encoder_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['intention_encoder_layer_norm'],
            encoder=encoders.get('intention')
        )
        # GHM uses the horizon-conditioned vector field.
        critic_vf_def = HorizonVectorField(
            vector_dim=obs_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            horizon_fourier_dim=config['horizon_fourier_dim'],
        )

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )
        reward_def = Value(
            hidden_dims=config['reward_hidden_dims'],
            layer_norm=config['reward_layer_norm'],
        )

        # Init args now include ex_horizons for the horizon-conditioned field.
        network_info = dict(
            critic=(critic_def, (ex_orig_observations, ex_actions)),
            critic_vf=(critic_vf_def, (
                ex_observations, ex_times,
                ex_observations, ex_actions, ex_latents, ex_horizons)),
            target_critic_vf=(copy.deepcopy(critic_vf_def), (
                ex_observations, ex_times,
                ex_observations, ex_actions, ex_latents, ex_horizons)),
            intention_encoder=(intention_encoder_def, (
                ex_orig_observations, ex_actions)),
            actor=(actor_def, (ex_orig_observations, )),
            reward=(reward_def, (ex_observations,)),
        )
        if config['encoder'] is not None:
            network_info['critic_vf_encoder'] = (
                encoders.get('critic_vf'), (ex_orig_observations,))
            network_info['target_critic_vf_encoder'] = (
                copy.deepcopy(encoders.get('critic_vf')), (ex_orig_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        if config['encoder'] is not None:
            params['modules_target_critic_vf_encoder'] = params['modules_critic_vf_encoder']
        params['modules_target_critic_vf'] = params['modules_critic_vf']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='ghm',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            intention_encoder_hidden_dims=(512, 512, 512, 512),  # Intention encoder network hidden dimensions.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            reward_hidden_dims=(512, 512, 512, 512),  # Reward network hidden dimensions.
            intention_encoder_layer_norm=True,  # Whether to use layer normalization for the intention encoder.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            reward_layer_norm=True,  # Whether to use layer normalization for the reward.
            latent_dim=512,  # Latent dimension for intention latents.
            discount=0.99,  # Fixed discount for downstream value (Q) estimation.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL style expectile.
            kl_weight=0.01,  # Weight for the KL divergence loss.
            q_agg='min',  # Aggregation method for Q values.
            critic_latent_type='prior',  # Type of critic latents. ('prior', 'encoding')
            num_flow_goals=16,  # Number of future flow goals for computing the target q.
            clip_flow_goals=True,  # Whether to clip the flow goals.
            actor_freq=4,  # Actor update frequency.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name ('mlp', 'impala_small', etc.).
            # ----- GHM (CompPlan) additions -----
            gamma_min=0.95,  # Min horizon discount sampled during training (GHM horizon-conditioning ON).
            gamma_max=0.999,  # Max horizon discount. gamma_min<gamma_max enables horizon training.
                             # (gamma_min==gamma_max==discount reduces GHM exactly to InFOM.)
            horizon_fourier_dim=0,  # 0 = feed raw gamma_h scalar; >0 = Fourier-embed to this many dims.
            discount_consistency_frac=0.125,  # Fraction of batch that gets the TD bootstrap term.
                                            # Paper: 0.25 (antmaze) / 0.125 (cube). 1.0 == InFOM.
        )
    )
    return config
