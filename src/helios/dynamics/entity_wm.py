"""Entity-factored world model (flax) — a transformer OVER ENTITIES.

============================================================================
WHAT THIS IS
============================================================================
A small world model whose tokens are ENTITIES (not timesteps). Given the
per-entity state tensor ``ent`` of shape ``(B, N, d_in)`` from the synthetic
world (see ``helios.envs.synthetic_entities``), it:

  1. embeds each entity to a ``d_model`` token (a per-entity token embedding),
  2. runs a few standard (NON-causal, fully-connected) transformer-encoder
     layers over the N entity tokens — every entity attends to every other,
     which is the substrate over which the later probe measures interaction
     relevance,
  3. produces three heads from the per-entity tokens / their pooled summary:
        * a self-predictive next-state head (per entity): predicts next
          per-entity state, giving a self-supervised consistency loss,
        * a reward head (scalar per transition),
        * a Q head (scalar action-value).

We use a *flat transformer over ground-truth entity states*, deliberately NOT
slot attention: the entities are given (one token per true entity), so there is
no slot-binding / slot-collapse failure mode to fight. That keeps the mechanism
check clean — any structure the probe finds is about value/interaction coupling,
not about whether slots bound correctly.

============================================================================
PROBE CONTRACT — read this; the value-coupling probe hangs off the Q head
============================================================================
``EntityWM.__call__(..., return_attn=True)`` returns, in the output dict:

  * ``attn``   : per-layer attention, stacked (n_layers, B, n_heads, N, N).
                 Each (N, N) slice is a directed weighted adjacency over the
                 ENTITY graph (token i attends to token j). Fully connected
                 (no mask), so it is the model's learned entity-interaction
                 graph — to be compared against ground-truth ``coupling`` and
                 ``value_relevant_entities``.
  * ``tokens`` : per-layer entity tokens (n_layers+1, B, N, d_model); index 0
                 is the input embedding, i>0 is layer i output. NODE features.
  * ``q``      : (B,) action-value. The probe will differentiate this.
  * ``reward`` : (B,) predicted reward.
  * ``next_ent``: (B, N, d_in) predicted next per-entity state.

Q-head design for the probe (IMPORTANT — flag for review):
  The Q head is built so that a later probe can compute the sensitivity of Q to
  *pairwise interactions*. Two affordances are provided:
    (a) Q is a function of the per-entity tokens AND the action, so
        ``∂Q/∂ent[i]`` (per-entity input gradient) is well defined and gives a
        per-entity value-relevance score; and
    (b) the attention tensor is returned, so a probe can also read
        ``∂Q/∂attn[layer, :, i, j]`` (interaction-edge sensitivity) by treating
        attention as an intermediate. To make (b) clean, the model exposes
        ``apply_with_attn_perturbation`` is NOT pre-baked here (left to the
        probe); instead we keep the forward fully differentiable and expose the
        attention as an output so the probe can use ``jax.grad`` of Q wrt a
        captured attention via a custom harness. See ``q_from_tokens`` which is
        the single choke-point the probe can target.

============================================================================
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Attention / transformer block over ENTITIES (no causal mask)
# ---------------------------------------------------------------------------


class EntitySelfAttention(nn.Module):
    """Full (unmasked) multi-head self-attention over N entity tokens.

    Returns ``(out, attn)`` with ``attn`` of shape (B, n_heads, N, N): the
    learned directed entity-interaction graph (post-softmax weights).
    """

    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        B, N, _ = x.shape
        head_dim = self.d_model // self.n_heads
        scale = 1.0 / jnp.sqrt(head_dim)

        qkv = nn.Dense(3 * self.d_model, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)   # (B,H,N,hd)
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale       # (B,H,N,N)
        attn = jax.nn.softmax(logits, axis=-1)                     # (B,H,N,N)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)               # (B,H,N,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.d_model)
        out = nn.Dense(self.d_model, name="proj")(out)
        return out, attn


class EntityBlock(nn.Module):
    """Pre-LN transformer-encoder block over entities. Returns (x, attn)."""

    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = nn.LayerNorm(name="ln1")(x)
        attn_out, attn = EntitySelfAttention(
            d_model=self.d_model, n_heads=self.n_heads, name="attn"
        )(h)
        x = x + attn_out
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(self.mlp_ratio * self.d_model, name="mlp_in")(h)
        h = getattr(nn, self.activation)(h)
        h = nn.Dense(self.d_model, name="mlp_out")(h)
        x = x + h
        return x, attn


# ---------------------------------------------------------------------------
# Entity-factored world model
# ---------------------------------------------------------------------------


class EntityWM(nn.Module):
    """Entity-factored world model: transformer over entity tokens + 3 heads.

    Args (Flax fields):
        n_entities:   N (number of entity tokens). Used only for the optional
                      learned entity-id embedding; the model also works for
                      OTHER N at apply time (attention/Dense are N-agnostic),
                      which is exactly what enables object-count OOD eval. When
                      ``use_id_embed`` is True the id embedding is sized to
                      ``max_entities`` so larger-N eval is supported.
        entity_dim:   per-entity input/output state dim (d_in).
        action_dim:   action dimensionality (fed to reward & Q heads).
        d_model:      token width.
        n_layers:     number of entity-transformer blocks (2-3).
        n_heads:      attention heads.
        mlp_ratio:    FFN expansion.
        use_id_embed: add a learned per-slot id embedding (helps the model use
                      the fixed agent/goal roles). Sized to ``max_entities``.
        max_entities: size of the id-embedding table (>= any eval N).
    """

    entity_dim: int
    action_dim: int
    n_entities: int = 4
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    use_id_embed: bool = True
    max_entities: int = 16

    # ------------------------------------------------------------------
    def encode(self, ent: jax.Array) -> tuple[jax.Array, list]:
        """Embed entities and run the entity-transformer.

        Args:
            ent: (B, N, entity_dim).
        Returns:
            tokens: (B, N, d_model) final per-entity tokens.
            collected: list with [input_embed, layer1_out, ...] tokens and the
                       per-layer attention. (Returned via __call__ packing.)
        """
        B, N, _ = ent.shape
        x = nn.Dense(self.d_model, name="entity_embed")(ent)       # (B,N,D)
        if self.use_id_embed:
            ids = jnp.arange(N)
            id_emb = nn.Embed(
                num_embeddings=self.max_entities,
                features=self.d_model,
                name="id_embed",
            )(ids)                                                 # (N, D)
            x = x + id_emb[None, :, :]

        token_layers = [x]
        attn_layers = []
        for i in range(self.n_layers):
            x, attn = EntityBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                name=f"block_{i}",
            )(x)
            token_layers.append(x)
            attn_layers.append(attn)
        return x, token_layers, attn_layers

    # ------------------------------------------------------------------
    def q_from_tokens(self, tokens: jax.Array, action: jax.Array) -> jax.Array:
        """SINGLE Q choke-point the probe targets.

        Q = MLP( [pooled_entity_summary , action] ). Pooling is mean over
        entity tokens, so Q is a smooth function of every per-entity token →
        ``∂Q/∂tokens[:, i, :]`` is a per-entity value-relevance signal, and
        because tokens are produced by attention over entities, the chain rule
        carries the signal back through the (returned) attention graph.

        Args:
            tokens: (B, N, d_model).
            action: (B, action_dim).
        Returns:
            q: (B,) scalar action-values.
        """
        pooled = jnp.mean(tokens, axis=1)                          # (B, D)
        h = jnp.concatenate([pooled, action], axis=-1)
        h = nn.Dense(self.d_model, name="q_in")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model, name="q_hidden")(h)
        h = nn.gelu(h)
        q = nn.Dense(1, name="q_out")(h)
        return q.squeeze(-1)

    # ------------------------------------------------------------------
    @nn.compact
    def __call__(
        self,
        ent: jax.Array,
        action: jax.Array,
        *,
        return_attn: bool = False,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            ent:    (B, N, entity_dim) current per-entity state.
            action: (B, action_dim) action.
            return_attn: also return per-layer attention + per-layer tokens.

        Returns dict with keys: next_ent (B,N,entity_dim), reward (B,), q (B,),
        and (if return_attn) attn (n_layers,B,n_heads,N,N), tokens
        (n_layers+1,B,N,d_model).
        """
        tokens, token_layers, attn_layers = self.encode(ent)

        # Self-predictive next-state head (per entity, residual prediction).
        delta = nn.Dense(self.entity_dim, name="next_head")(tokens)  # (B,N,d_in)
        next_ent = ent + delta

        # Reward head: pool entity tokens + action -> scalar.
        pooled = jnp.mean(tokens, axis=1)
        rh = jnp.concatenate([pooled, action], axis=-1)
        rh = nn.Dense(self.d_model, name="r_in")(rh)
        rh = nn.gelu(rh)
        reward = nn.Dense(1, name="r_out")(rh).squeeze(-1)          # (B,)

        # Q head (probe choke-point).
        q = self.q_from_tokens(tokens, action)                     # (B,)

        out = {"next_ent": next_ent, "reward": reward, "q": q}
        if return_attn:
            out["attn"] = jnp.stack(attn_layers, axis=0)
            out["tokens"] = jnp.stack(token_layers, axis=0)
        return out


# ---------------------------------------------------------------------------
# Loss (self-pred consistency + reward + Q)
# ---------------------------------------------------------------------------


def entity_wm_loss(
    params,
    apply_fn,
    batch: dict[str, jax.Array],
    *,
    w_self: float = 1.0,
    w_reward: float = 1.0,
    w_q: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Combined loss for one batch.

    Expected batch keys (all leading dim B):
        ent:       (B, N, d_in) current per-entity state.
        action:    (B, action_dim).
        next_ent:  (B, N, d_in) ground-truth next per-entity state.
        reward:    (B,) ground-truth reward.
        q_target:  (B,) bootstrap / Monte-Carlo return target for Q.

    Returns (scalar loss, metrics dict).
    """
    out = apply_fn({"params": params}, batch["ent"], batch["action"])
    self_loss = jnp.mean((out["next_ent"] - batch["next_ent"]) ** 2)
    reward_loss = jnp.mean((out["reward"] - batch["reward"]) ** 2)
    q_loss = jnp.mean((out["q"] - batch["q_target"]) ** 2)
    loss = w_self * self_loss + w_reward * reward_loss + w_q * q_loss
    metrics = {
        "loss": loss,
        "self_loss": self_loss,
        "reward_loss": reward_loss,
        "q_loss": q_loss,
    }
    return loss, metrics
