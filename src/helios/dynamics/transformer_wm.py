"""Block-causal Transformer dynamics — a Dreamer-4-inspired world model.

This is a deliberately SMALL, single-GPU (16 GB, RTX 5070 Ti) transformer
sequence model that REPLACES the RSSM sequence core
(``helios.dynamics.rssm.RSSMModule``) while keeping the rest of the DreamerV3
machinery (encoder / reward head / continue head / actor / critic / TD-λ
imagination) intact.

It is NOT a faithful Dreamer 4. Faithful Dreamer 4 uses pixel inputs, a VQ
tokenizer, and a shortcut-forcing diffusion objective, all of which need
8×24 GB GPUs. Here the goal is a *controllable* transformer world model over
STATE observations, whose PRIMARY purpose is to be a substrate for later
structural-entropy (SE) analysis over the attention / token graph.

Tokenization v1 (simplest)
--------------------------
Each timestep's (state-embedding, action) is mapped to ONE token via a linear
embed to ``d_model``. The sequence is therefore over TIME; "block-causal" here
reduces to a standard causal mask over timesteps (one block == one timestep).
A future v2 could split each timestep into multiple tokens (e.g. per-group
state tokens + an action token) and use a true block-causal mask where tokens
within the same timestep attend bidirectionally but only causally across
timesteps. The mask helper below is written to make that extension easy.

The transformer reads tokens ``x_0 .. x_{t}`` and, at position ``t``, predicts
the next latent ``z_{t+1}`` plus the reward and continue signal for that
transition — i.e. it is an autoregressive next-latent predictor, mirroring the
RSSM prior/posterior roles but with attention-over-history instead of a GRU.

============================================================================
SE NORTH STAR — attention / token graph substrate
============================================================================
``TransformerWM.__call__`` accepts ``return_attn=True`` and then returns, in the
output dict:

* ``attn``  : per-layer attention matrices, stacked as
              (n_layers, batch, n_heads, T, T). Each (T, T) slice is a directed,
              weighted adjacency over the TIME-token graph (rows attend to
              columns; causal so it is lower-triangular). This is the raw
              material for building the attention graph.
* ``tokens``: the per-layer token embeddings, stacked as
              (n_layers+1, batch, T, d_model) — index 0 is the input embedding,
              index i>0 is the output of layer i. These are the NODE features
              for the token graph.

TODO(SE): a later script (scripts/se_attention_graph.py, not yet written)
should: (1) roll out a trained TransformerWM with ``return_attn=True`` over a
batch of real trajectories; (2) build a graph whose nodes are tokens (time
steps, optionally × layers) and whose edge weights come from ``attn`` (e.g.
average over heads, threshold, or keep top-k per row); (3) run structural
entropy (2-D / multilevel SE, cf. the selib se_louvain encoder) on that graph
to measure how hierarchically the world model organises temporal structure.
Keep ``attn`` and ``tokens`` as the stable contract for that script.
============================================================================
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> jax.Array:
    """Standard fixed sinusoidal positional encoding, shape (seq_len, d_model)."""
    position = jnp.arange(seq_len)[:, None]              # (T, 1)
    div_term = jnp.exp(
        jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model)
    )                                                    # (d_model/2,)
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def causal_mask(seq_len: int) -> jax.Array:
    """Lower-triangular boolean mask, shape (T, T). True == attend allowed.

    With tokenization v1 (one token per timestep) this is a plain causal mask.
    For a future multi-token-per-timestep scheme, replace this with a
    block-causal mask: full attention within a timestep block, causal across
    blocks. The attention call below only assumes a boolean (T, T) mask, so
    swapping this helper is sufficient.
    """
    idx = jnp.arange(seq_len)
    return idx[None, :] <= idx[:, None]   # (T, T): query row i sees keys j<=i


# ---------------------------------------------------------------------------
# Transformer block (Pre-LN), returns attention weights for the SE substrate
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention that also returns the attention weights.

    Returns ``(out, attn)`` where ``attn`` has shape (batch, n_heads, T, T)
    (post-softmax weights). The weights are the directed edge weights of the
    per-layer token graph used by the SE analysis.
    """

    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        B, T, _ = x.shape
        head_dim = self.d_model // self.n_heads
        scale = 1.0 / jnp.sqrt(head_dim)

        qkv = nn.Dense(3 * self.d_model, name="qkv")(x)            # (B, T, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)   # (B, H, T, hd)

        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale       # (B, H, T, T)
        # mask: (T, T) True == allowed → set disallowed to -inf before softmax
        neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask[None, None, :, :], logits, neg)
        attn = jax.nn.softmax(logits, axis=-1)                     # (B, H, T, T)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)               # (B, H, T, hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        out = nn.Dense(self.d_model, name="proj")(out)
        return out, attn


class TransformerBlock(nn.Module):
    """Pre-LN transformer block. Returns (x, attn) so callers can collect attn."""

    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Pre-LN attention
        h = nn.LayerNorm(name="ln1")(x)
        attn_out, attn = MultiHeadSelfAttention(
            d_model=self.d_model, n_heads=self.n_heads, name="attn"
        )(h, mask)
        x = x + attn_out
        # Pre-LN MLP
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(self.mlp_ratio * self.d_model, name="mlp_in")(h)
        h = getattr(nn, self.activation)(h)
        h = nn.Dense(self.d_model, name="mlp_out")(h)
        x = x + h
        return x, attn


# ---------------------------------------------------------------------------
# Transformer world-model core
# ---------------------------------------------------------------------------


class TransformerWM(nn.Module):
    """Block-causal Transformer dynamics core.

    Consumes a sequence of (state-embedding, action) pairs and produces, at each
    timestep ``t``, a latent ``z_t`` (the "feature" the actor/critic/heads read)
    that summarises history ``0..t``. Because the model is causal, ``z_t`` is a
    valid one-step predictive state: feeding ``embed_t`` + ``action_t`` yields a
    representation from which reward/continue and the next state are predicted.

    Defaults are sized to fit 16 GB with the DreamerV3 batch (B*L tokens):
        d_model=256, n_layers=4, n_heads=4, context_len=32, mlp_ratio=4.

    Args (Flax fields):
        embed_dim:   dimensionality of the encoder output (state embedding).
        action_dim:  action dimensionality.
        d_model:     transformer width / latent (feature) dimensionality.
        n_layers:    number of transformer blocks.
        n_heads:     attention heads.
        context_len: maximum sequence length (positional table size).
        mlp_ratio:   FFN expansion factor.
        pos_encoding: "learned" or "sinusoidal".
    """

    embed_dim: int
    action_dim: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    context_len: int = 32
    mlp_ratio: int = 4
    activation: str = "gelu"
    pos_encoding: str = "learned"

    @property
    def feature_dim(self) -> int:
        """The actor/critic/heads consume the d_model-wide transformer output."""
        return self.d_model

    @nn.compact
    def __call__(
        self,
        embed: jax.Array,
        action: jax.Array,
        return_attn: bool = False,
    ) -> dict[str, Any]:
        """Forward the full sequence.

        Args:
            embed:  state embeddings, shape (B, T, embed_dim). For imagination
                    where there is no observation, pass zeros (the action token
                    still drives the prediction).
            action: actions, shape (B, T, action_dim).
            return_attn: if True, also return per-layer ``attn`` and ``tokens``
                    for the SE attention/token graph (see module docstring).

        Returns:
            dict with:
              - ``z``: (B, T, d_model) per-timestep latent feature. ``z[:, t]``
                summarises tokens 0..t and is the predictive state for step t.
              - ``attn`` (if return_attn): (n_layers, B, n_heads, T, T).
              - ``tokens`` (if return_attn): (n_layers+1, B, T, d_model).
        """
        B, T, _ = embed.shape

        # --- Tokenization v1: one token per timestep from [embed, action] ---
        tok_in = jnp.concatenate([embed, action], axis=-1)          # (B, T, e+a)
        x = nn.Dense(self.d_model, name="token_embed")(tok_in)      # (B, T, D)

        # --- Positional encoding ---
        if self.pos_encoding == "learned":
            pos_table = self.param(
                "pos_embed",
                nn.initializers.normal(stddev=0.02),
                (self.context_len, self.d_model),
            )
            x = x + pos_table[:T][None, :, :]
        else:
            x = x + sinusoidal_positional_encoding(T, self.d_model)[None, :, :]

        mask = causal_mask(T)                                       # (T, T)

        token_layers = [x]
        attn_layers = []
        for i in range(self.n_layers):
            x, attn = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                activation=self.activation,
                name=f"block_{i}",
            )(x, mask)
            token_layers.append(x)
            attn_layers.append(attn)

        x = nn.LayerNorm(name="ln_out")(x)                          # (B, T, D)

        out: dict[str, Any] = {"z": x}
        if return_attn:
            # (n_layers, B, n_heads, T, T) — directed weighted time-token graph.
            out["attn"] = jnp.stack(attn_layers, axis=0)
            # (n_layers+1, B, T, d_model) — node features per layer (0 = input).
            out["tokens"] = jnp.stack(token_layers, axis=0)
        return out


# ---------------------------------------------------------------------------
# Single-step incremental wrapper (acting / autoregressive imagination)
# ---------------------------------------------------------------------------


def transformer_step(
    module: TransformerWM,
    params: dict,
    embed_hist: jax.Array,
    action_hist: jax.Array,
    return_attn: bool = False,
) -> jax.Array | dict[str, Any]:
    """Run the transformer over a history window and return the LAST latent.

    Used for acting and for autoregressive imagination, where we maintain a
    rolling window of the most recent ``<= context_len`` (embed, action) pairs
    and only need the prediction at the final position.

    Args:
        module: a :class:`TransformerWM` instance.
        params: its Flax params.
        embed_hist:  (B, t, embed_dim) history of embeddings (t <= context_len).
        action_hist: (B, t, action_dim) history of actions.
        return_attn: forward the SE-substrate flag.

    Returns:
        If ``return_attn`` is False: ``z_last`` of shape (B, d_model).
        Else the full output dict (caller can index ``out['z'][:, -1]``).
    """
    out = module.apply(params, embed_hist, action_hist, return_attn=return_attn)
    if return_attn:
        return out
    return out["z"][:, -1]
