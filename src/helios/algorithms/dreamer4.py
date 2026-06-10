"""DreamerV4 (small): block-causal Transformer world model + imagination AC.

A Dreamer-4-INSPIRED agent that swaps the RSSM sequence core of DreamerV3 for a
small block-causal Transformer (``helios.dynamics.transformer_wm.TransformerWM``)
while reusing DreamerV3's actor / critic / reward & continue heads and its TD-λ
imagination actor-critic update (imported from ``helios.algorithms.dreamer``).

This is NOT faithful Dreamer 4 — no pixels, no VQ tokenizer, no shortcut-forcing
diffusion. It is a controllable transformer world model over STATE observations,
intended as a substrate for later structural-entropy analysis of the
attention / token graph (see ``transformer_wm`` module docstring, SE NORTH STAR).

World-model design vs RSSM
--------------------------
* The "latent" / feature is the transformer output ``z_t`` of width ``d_model``.
  There is no separate (h, z) split and no categorical KL term — the loss is a
  reconstruction (next-state in embedding space + raw obs) + reward + continue
  objective, which is enough to drive imagination and is the simplest correct
  thing for a feasibility probe.
* Training is teacher-forced: the transformer reads the whole real
  (embed, action) sequence at once (one causal forward pass) and predicts, at
  each position, the reward/continue for that transition and the NEXT-step
  observation embedding (a 1-step latent-consistency term).
* Imagination is autoregressive over a sliding window of <= context_len steps:
  starting from the encoded context of a real sub-sequence, the actor proposes
  actions, the transformer predicts the next latent, and reward/continue heads
  score it. The resulting (feat, reward, continue) tensors are fed verbatim to
  DreamerV3's existing TD-λ critic/actor updates.

Reference (inspiration only):
    Hafner et al., DreamerV3 (2023) https://arxiv.org/abs/2301.04104
    "Dreamer 4" block-causal transformer dynamics (state-based reduction here).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from helios.algorithms.base import BaseAgent
from helios.algorithms.dreamer import (
    Actor,
    Critic,
    RewardHead,
    ContinueHead,
    _critic_update,
    _actor_update,
)
from helios.core.distributions import TanhNormal
from helios.core.networks import MLP
from helios.dynamics.transformer_wm import TransformerWM


class DreamerV4Agent(BaseAgent):
    """Transformer-world-model imagination actor-critic agent.

    Args:
        config: SimpleNamespace/DictConfig with ``transformer`` sub-config plus
            actor/critic/optimizer hyperparameters (see scripts/run_dreamer4.py).
        observation_space: object exposing ``.shape`` (state envs only).
        action_space: object exposing ``.shape`` (Box) — continuous actions.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initial_state(self, key: jax.Array) -> dict[str, Any]:
        import gymnasium as gym

        cfg = self.config
        obs_shape = self.observation_space.shape
        if len(obs_shape) != 1:
            raise NotImplementedError(
                "DreamerV4Agent (transformer WM) supports STATE observations "
                f"only; got obs shape {obs_shape}."
            )
        obs_dim = int(obs_shape[0])

        if isinstance(self.action_space, gym.spaces.Box):
            action_dim = int(self.action_space.shape[0])
        else:
            action_dim = int(self.action_space.n)

        tcfg = cfg.transformer
        embed_dim = int(tcfg.embed_dim)
        d_model = int(tcfg.d_model)

        # -- Encoder: state -> embedding (MLP, mirrors dreamer.py state path) --
        encoder = MLP(hidden_dims=(embed_dim,), output_dim=embed_dim, activation="silu")

        # -- Transformer sequence core (replaces the RSSM) --
        wm = TransformerWM(
            embed_dim=embed_dim,
            action_dim=action_dim,
            d_model=d_model,
            n_layers=int(tcfg.n_layers),
            n_heads=int(tcfg.n_heads),
            context_len=int(tcfg.context_len),
            mlp_ratio=int(tcfg.mlp_ratio),
            pos_encoding=str(getattr(tcfg, "pos_encoding", "learned")),
        )

        feature_dim = d_model

        # -- Decoder back to raw obs (reconstruction target) --
        decoder = MLP(hidden_dims=(embed_dim,), output_dim=obs_dim, activation="silu")
        # -- 1-step next-embedding predictor (latent consistency) --
        embed_pred = MLP(hidden_dims=(embed_dim,), output_dim=embed_dim, activation="silu")

        reward_head = RewardHead(hidden_dims=tuple(cfg.actor_hidden_dims))
        continue_head = ContinueHead(hidden_dims=tuple(cfg.critic_hidden_dims))
        actor = Actor(action_dim=action_dim, hidden_dims=tuple(cfg.actor_hidden_dims))
        critic = Critic(hidden_dims=tuple(cfg.critic_hidden_dims))
        slow_critic = Critic(hidden_dims=tuple(cfg.critic_hidden_dims))

        key, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 9)

        # Dummy inputs (B=1, T=context_len) for init.
        ctx = int(tcfg.context_len)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_obs_seq = jnp.zeros((1 * ctx, obs_dim))
        dummy_embed_seq = jnp.zeros((1, ctx, embed_dim))
        dummy_action_seq = jnp.zeros((1, ctx, action_dim))
        dummy_feat = jnp.zeros((1, feature_dim))
        dummy_feat_flat = jnp.zeros((ctx, feature_dim))

        enc_params = encoder.init(k1, dummy_obs)
        wm_params_core = wm.init(k2, dummy_embed_seq, dummy_action_seq)
        dec_params = decoder.init(k3, dummy_feat_flat)
        embed_pred_params = embed_pred.init(k4, dummy_feat_flat)
        rew_params = reward_head.init(k5, dummy_feat)
        cont_params = continue_head.init(k6, dummy_feat)
        actor_params = actor.init(k7, dummy_feat)
        critic_params = critic.init(k8, dummy_feat)
        slow_critic_params = critic_params

        wm_params = {
            "encoder": enc_params,
            "wm": wm_params_core,
            "decoder": dec_params,
            "embed_pred": embed_pred_params,
            "reward": rew_params,
            "continue": cont_params,
        }

        model_tx = optax.chain(
            optax.clip_by_global_norm(float(cfg.max_grad_norm)),
            optax.adam(float(cfg.model_lr), eps=float(cfg.model_eps)),
        )
        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(cfg.max_grad_norm)),
            optax.adam(float(cfg.actor_lr), eps=float(cfg.actor_eps)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(cfg.max_grad_norm)),
            optax.adam(float(cfg.critic_lr), eps=float(cfg.critic_eps)),
        )

        wm_opt_state = model_tx.init(wm_params)
        actor_opt_state = actor_tx.init(actor_params)
        critic_opt_state = critic_tx.init(critic_params)

        return {
            # modules
            "encoder": encoder,
            "wm": wm,
            "decoder": decoder,
            "embed_pred": embed_pred,
            "reward_head": reward_head,
            "continue_head": continue_head,
            "actor": actor,
            "critic": critic,
            "slow_critic": slow_critic,
            # params
            "wm_params": wm_params,
            "actor_params": actor_params,
            "critic_params": critic_params,
            "slow_critic_params": slow_critic_params,
            # opt states
            "wm_opt_state": wm_opt_state,
            "actor_opt_state": actor_opt_state,
            "critic_opt_state": critic_opt_state,
            # optimizers
            "model_tx": model_tx,
            "actor_tx": actor_tx,
            "critic_tx": critic_tx,
            "step": 0,
        }

    # ------------------------------------------------------------------
    # Acting (used by the launcher's collection rollout via a jitted closure;
    # this method is a simple reference implementation)
    # ------------------------------------------------------------------

    def act(
        self,
        obs: jax.Array,
        state: dict[str, Any],
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Single-step action from a one-element history (no temporal context).

        The launcher (scripts/run_dreamer4.py) maintains a real sliding window
        for collection; this fallback encodes the current obs and runs the
        transformer over a length-1 sequence.
        """
        enc = state["encoder"]
        wm = state["wm"]
        actor = state["actor"]

        embed = enc.apply(state["wm_params"]["encoder"], obs)          # (B, e)
        a0 = jnp.zeros((obs.shape[0], actor.action_dim))
        out = wm.apply(
            state["wm_params"]["wm"], embed[:, None, :], a0[:, None, :]
        )
        feat = out["z"][:, -1]                                         # (B, d_model)

        actor_out = actor.apply(state["actor_params"], feat)
        dist = TanhNormal(actor_out["mean"], actor_out["log_std"])
        key, ak = jax.random.split(key)
        action = dist.mode() if deterministic else dist.sample(ak)[0]
        return action, state

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        batch,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        cfg = self.config

        # ---- 1. World-model (transformer) update; returns per-(B,T) latents ----
        wm_params, wm_opt_state, wm_metrics, feats_bt = _wm_update_transformer(
            wm_params=state["wm_params"],
            opt_state=state["wm_opt_state"],
            tx=state["model_tx"],
            encoder=state["encoder"],
            wm=state["wm"],
            decoder=state["decoder"],
            embed_pred=state["embed_pred"],
            reward_head=state["reward_head"],
            continue_head=state["continue_head"],
            batch=batch,
            recon_weight=float(cfg.reconstruction_loss_weight),
            embed_weight=float(getattr(cfg, "embed_loss_weight", 1.0)),
            reward_weight=float(cfg.reward_loss_weight),
            continue_weight=float(cfg.continue_loss_weight),
        )

        # ---- 2. Imagination rollout (autoregressive transformer) ----
        imag_feats, imag_rewards, imag_continues = _imagine_rollout_transformer(
            wm_params=wm_params,
            encoder=state["encoder"],
            wm=state["wm"],
            reward_head=state["reward_head"],
            continue_head=state["continue_head"],
            actor=state["actor"],
            actor_params=state["actor_params"],
            batch=batch,
            context_len=int(cfg.transformer.context_len),
            horizon=int(cfg.imagination_horizon),
        )

        # ---- 3. Critic update (reuse DreamerV3 TD-λ) ----
        critic_params, critic_opt_state, critic_metrics = _critic_update(
            critic_params=state["critic_params"],
            slow_critic_params=state["slow_critic_params"],
            opt_state=state["critic_opt_state"],
            tx=state["critic_tx"],
            critic=state["critic"],
            slow_critic=state["slow_critic"],
            imag_feats=imag_feats,
            imag_rewards=imag_rewards,
            imag_continues=imag_continues,
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
        )

        # ---- 4. Actor update (reuse DreamerV3) ----
        actor_params, actor_opt_state, actor_metrics = _actor_update(
            actor_params=state["actor_params"],
            opt_state=state["actor_opt_state"],
            tx=state["actor_tx"],
            actor=state["actor"],
            critic=state["critic"],
            critic_params=critic_params,
            imag_feats=imag_feats,
            imag_continues=imag_continues,
            entropy_scale=float(cfg.actor_entropy_scale),
        )

        # ---- 5. Slow critic EMA ----
        slow_critic_params = jax.tree_util.tree_map(
            lambda s, f: (1.0 - float(cfg.slow_target_fraction)) * s
            + float(cfg.slow_target_fraction) * f,
            state["slow_critic_params"],
            critic_params,
        )

        new_state = {
            **state,
            "wm_params": wm_params,
            "actor_params": actor_params,
            "critic_params": critic_params,
            "slow_critic_params": slow_critic_params,
            "wm_opt_state": wm_opt_state,
            "actor_opt_state": actor_opt_state,
            "critic_opt_state": critic_opt_state,
            "step": state["step"] + 1,
        }
        metrics = {**wm_metrics, **critic_metrics, **actor_metrics}
        return new_state, metrics


# ---------------------------------------------------------------------------
# World-model update (transformer, teacher-forced, one causal pass)
# ---------------------------------------------------------------------------


def _wm_update_transformer(
    wm_params,
    opt_state,
    tx,
    encoder,
    wm,
    decoder,
    embed_pred,
    reward_head,
    continue_head,
    batch,
    recon_weight: float,
    embed_weight: float,
    reward_weight: float,
    continue_weight: float,
):
    """Teacher-forced transformer WM loss. Returns (params, opt, metrics, feats)."""

    def loss_fn(params):
        B, T = batch.obs.shape[:2]
        obs_flat = batch.obs.reshape(B * T, -1)

        # Encode observations -> embeddings, then run ONE causal transformer pass.
        embed = encoder.apply(params["encoder"], obs_flat).reshape(B, T, -1)
        out = wm.apply(params["wm"], embed, batch.actions)
        feat = out["z"]                              # (B, T, d_model)
        feat_flat = feat.reshape(B * T, -1)

        # Reconstruction of the current observation from the latent.
        recon = decoder.apply(params["decoder"], feat_flat)
        recon_loss = jnp.mean((recon - obs_flat) ** 2)

        # 1-step latent consistency: predict embed_{t+1} from feat_t.
        pred_next_embed = embed_pred.apply(params["embed_pred"], feat_flat)
        pred_next_embed = pred_next_embed.reshape(B, T, -1)
        # target = embed shifted by one; last step has no target → masked out.
        tgt_next = jax.lax.stop_gradient(embed[:, 1:, :])
        embed_err = (pred_next_embed[:, :-1, :] - tgt_next) ** 2
        embed_loss = jnp.mean(embed_err)

        # Reward + continue for the transition at each step.
        pred_rew = reward_head.apply(params["reward"], feat_flat)
        rew_loss = jnp.mean((pred_rew - batch.rewards.reshape(B * T)) ** 2)

        pred_cont = continue_head.apply(params["continue"], feat_flat)
        cont_target = 1.0 - batch.dones.reshape(B * T)
        cont_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(pred_cont, cont_target)
        )

        total_loss = (
            recon_weight * recon_loss
            + embed_weight * embed_loss
            + reward_weight * rew_loss
            + continue_weight * cont_loss
        )
        metrics = {
            "wm/recon": recon_loss,
            "wm/embed": embed_loss,
            "wm/reward": rew_loss,
            "wm/continue": cont_loss,
            "wm/total": total_loss,
        }
        return total_loss, (metrics, feat)

    (_, (metrics, feat)), grads = jax.value_and_grad(loss_fn, has_aux=True)(wm_params)
    updates, new_opt_state = tx.update(grads, opt_state, wm_params)
    new_params = optax.apply_updates(wm_params, updates)
    return new_params, new_opt_state, metrics, feat


# ---------------------------------------------------------------------------
# Imagination rollout (autoregressive transformer over a sliding window)
# ---------------------------------------------------------------------------


def _imagine_rollout_transformer(
    wm_params,
    encoder,
    wm,
    reward_head,
    continue_head,
    actor,
    actor_params,
    batch,
    context_len: int,
    horizon: int,
):
    """Autoregressive imagination from real-trajectory contexts.

    We use each training sub-sequence as a real "prompt": encode its
    observations, keep the last ``ctx-1`` (embed, action) pairs as context, then
    roll the actor forward ``horizon`` steps. At every step the transformer reads
    the rolling window (context + imagined-so-far) and predicts the next latent;
    reward/continue heads score it. Imagined latents are mapped back to
    embeddings via the encoder's inverse role using the learned ``decoder`` is
    avoided — instead we feed the predicted latent's embedding via a light reuse:
    we approximate the next embedding with a zero embedding and rely on the
    action token (matching the RSSM "imagine" path which also drops the obs).

    Returns (imag_feats, imag_rewards, imag_continues) shaped like DreamerV3's
    imagination so the existing TD-λ updates apply unchanged:
        imag_feats:     (H, N, d_model)
        imag_rewards:   (H, N)
        imag_continues: (H, N)
    """
    B, T = batch.obs.shape[:2]
    obs_flat = batch.obs.reshape(B * T, -1)
    embed_all = encoder.apply(wm_params["encoder"], obs_flat).reshape(B, T, -1)
    embed_dim = embed_all.shape[-1]
    action_dim = batch.actions.shape[-1]

    # Use a window of the most recent (ctx-1) real steps as the prompt so that
    # after appending one imagined step the window length stays <= ctx.
    win = min(context_len - 1, T)
    win = max(win, 1)
    # Rolling histories, shape (N=B, win, *). Start from the tail of each seq.
    emb_hist = embed_all[:, T - win:, :]                 # (B, win, e)
    act_hist = batch.actions[:, T - win:, :]             # (B, win, a)
    N = B

    imag_feats = []
    imag_rewards = []
    imag_continues = []

    def trim(arr):
        # Keep at most context_len timesteps (drop oldest).
        if arr.shape[1] > context_len:
            return arr[:, -context_len:, :]
        return arr

    for t in range(horizon):
        out = wm.apply(wm_params["wm"], emb_hist, act_hist)
        feat = out["z"][:, -1]                           # (N, d_model)

        actor_out = actor.apply(actor_params, feat)
        dist = TanhNormal(actor_out["mean"], actor_out["log_std"])
        action, _ = dist.sample(jax.random.PRNGKey(t))   # (N, a)

        rew = reward_head.apply(wm_params["reward"], feat)
        cont = continue_head.apply(wm_params["continue"], feat)

        imag_feats.append(feat)
        imag_rewards.append(rew)
        imag_continues.append(cont)

        # Append an imagined step: no real observation, so push a zero embedding
        # (prior / imagine mode) with the actor's action — same information
        # content as the RSSM imagine path (embed=None).
        next_embed = jnp.zeros((N, 1, embed_dim))
        emb_hist = trim(jnp.concatenate([emb_hist, next_embed], axis=1))
        act_hist = trim(jnp.concatenate([act_hist, action[:, None, :]], axis=1))

    imag_feats = jnp.stack(imag_feats, axis=0)           # (H, N, d_model)
    imag_rewards = jnp.stack(imag_rewards, axis=0)       # (H, N)
    imag_continues = jnp.stack(imag_continues, axis=0)   # (H, N)
    return imag_feats, imag_rewards, imag_continues
