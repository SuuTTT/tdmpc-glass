"""Monolithic (flat-MLP) world model — the CONTROL baseline for the GWM-as-
simulator mechanism check.

============================================================================
WHAT THIS IS (and why)
============================================================================
This is the MONOLITHIC counterpart to ``helios.dynamics.entity_wm.EntityWM``
(the entity-factored / graph WM). Where EntityWM tokenizes per entity and runs a
transformer OVER ENTITIES (so the relational/contact structure is in the
architecture), this model has NO entity factorization: it flattens the per-entity
state into one big vector ``(B, N*d_in)`` and runs a plain MLP trunk, then the
SAME three heads as EntityWM:

  * a self-predictive next-state head: predicts the next per-entity state,
    reshaped back to ``(B, N, d_in)`` (residual prediction, like EntityWM),
  * a reward head (scalar per transition),
  * a Q head (scalar action-value).

It is the control the graph WM must BEAT. The bet (from the GWM survey) is that a
monolithic latent collapses on compositional-OOD value-decodability and on
contact-conditioned prediction, while the entity/graph latent holds.

============================================================================
PARAM-MATCHING (read this; the script reports exact counts)
============================================================================
A fair control must have a comparable parameter budget. EntityWM's parameter
count depends on ``d_model``, ``n_layers``, ``n_heads`` (and is ~N-INVARIANT:
attention/Dense weights do not grow with N). MonolithicWM's first/last layers DO
grow with N (flat input/output is ``N*d_in``), so an exact match for all N is
impossible. We therefore:

  * expose ``hidden`` and ``n_layers`` for the trunk MLP, and
  * provide ``matched_hidden(...)`` which, given the EntityWM config and the
    N used at TRAINING time, returns a ``hidden`` width whose total param count
    is as close as practical to EntityWM's at that N.

The mechanism-check script calls ``matched_hidden`` at ``N_train`` and reports
BOTH models' exact parameter counts in the output JSON (honest, approximate
match — see caveats). Because the trunk dominates, the match holds well at the
training N; at OOD N the monolithic input/output layers grow slightly (still the
intended apples-to-apples control: same family, comparable budget).

============================================================================
INTERFACE PARITY with EntityWM
============================================================================
``__call__(ent, action, *, return_attn=False)`` returns a dict with the SAME
keys EntityWM produces — ``next_ent (B,N,d_in)``, ``reward (B,)``, ``q (B,)`` —
so it is a drop-in for ``entity_wm_loss``. ``return_attn`` is accepted for API
parity but a monolithic model has no attention graph; when requested it returns
``tokens`` = the flat trunk latent broadcast to ``(1, B, 1, hidden)`` and
``attn`` = ``None`` (callers that need a graph should not use this model). The
shared ``monolithic_wm_loss`` mirrors ``entity_wm_loss`` exactly (same keys,
same weights) so both models train under an identical objective.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class MonolithicWM(nn.Module):
    """Flat-MLP world model over the concatenated entity state.

    Args (Flax fields):
        entity_dim:   per-entity input/output state dim (d_in).
        action_dim:   action dimensionality (fed to reward & Q heads).
        n_entities:   N. Recorded for bookkeeping only. The model is N-AGNOSTIC
                      at apply time: ``encode`` builds a FIXED-WIDTH input via
                      mean+max pooling over entities, so the SAME trained params
                      run at OOD N (exactly like EntityWM). This keeps the
                      monolithic control honest as an apples-to-apples baseline.
        hidden:       trunk MLP width.
        n_layers:     number of trunk MLP layers.
        activation:   trunk/head nonlinearity name (a flax.linen fn).
    """

    entity_dim: int
    action_dim: int
    n_entities: int = 5
    hidden: int = 128
    n_layers: int = 2
    activation: str = "gelu"

    # ------------------------------------------------------------------
    def encode(self, ent: jax.Array) -> jax.Array:
        """Flatten + MLP trunk -> a single monolithic latent (B, hidden).

        To stay N-AGNOSTIC at apply time (so the SAME trained params can be
        evaluated at OOD N, exactly like EntityWM), the flat input is built from
        an N-invariant summary: concat of [mean over entities, max over entities]
        of the per-entity states. This is the deliberately *unstructured*
        counterpart to per-entity tokens — it mixes all entities into one vector
        with no factorization, which is the whole point of the monolithic control.
        """
        # ent: (B, N, d_in). N-invariant pooled summary -> (B, 2*d_in).
        mean = jnp.mean(ent, axis=1)
        mx = jnp.max(ent, axis=1)
        x = jnp.concatenate([mean, mx], axis=-1)               # (B, 2*d_in)
        for i in range(self.n_layers):
            x = nn.Dense(self.hidden, name=f"trunk_{i}")(x)
            x = getattr(nn, self.activation)(x)
        return x                                               # (B, hidden)

    # ------------------------------------------------------------------
    def q_from_latent(self, latent: jax.Array, action: jax.Array) -> jax.Array:
        """Q = MLP([latent, action]) -> (B,). Mirrors EntityWM.q_from_tokens."""
        h = jnp.concatenate([latent, action], axis=-1)
        h = nn.Dense(self.hidden, name="q_in")(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden, name="q_hidden")(h)
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
        B, N, d_in = ent.shape
        latent = self.encode(ent)                              # (B, hidden)

        # Self-predictive next-state head: predict a per-entity DELTA from a
        # broadcast of the monolithic latent + the entity's own state. The latent
        # carries the (entangled) interaction info; concatenating each entity's
        # own state lets the flat head still place the prediction per entity. This
        # is N-agnostic (one shared head applied to every entity row).
        lat_b = jnp.broadcast_to(latent[:, None, :], (B, N, self.hidden))
        head_in = jnp.concatenate([lat_b, ent], axis=-1)       # (B,N,hidden+d_in)
        h = nn.Dense(self.hidden, name="next_in")(head_in)
        h = getattr(nn, self.activation)(h)
        delta = nn.Dense(d_in, name="next_out")(h)             # (B,N,d_in)
        next_ent = ent + delta

        # Reward head: latent + action -> scalar.
        rh = jnp.concatenate([latent, action], axis=-1)
        rh = nn.Dense(self.hidden, name="r_in")(rh)
        rh = nn.gelu(rh)
        reward = nn.Dense(1, name="r_out")(rh).squeeze(-1)     # (B,)

        # Q head.
        q = self.q_from_latent(latent, action)                # (B,)

        out = {"next_ent": next_ent, "reward": reward, "q": q}
        if return_attn:
            # No attention graph in a monolithic model; expose the latent for any
            # value-decodability readout, and a null attn for API parity.
            out["tokens"] = latent[None, :, None, :]          # (1,B,1,hidden)
            out["attn"] = None
        return out


# ---------------------------------------------------------------------------
# Loss (mirrors entity_wm_loss exactly: same keys, same weights)
# ---------------------------------------------------------------------------


def monolithic_wm_loss(
    params,
    apply_fn,
    batch: dict[str, jax.Array],
    *,
    w_self: float = 1.0,
    w_reward: float = 1.0,
    w_q: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Combined loss for one batch. Identical signature/keys to entity_wm_loss.

    Expected batch keys: ent (B,N,d_in), action (B,A), next_ent (B,N,d_in),
    reward (B,), q_target (B,). Returns (scalar loss, metrics dict).
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


# ---------------------------------------------------------------------------
# Param-matching helper
# ---------------------------------------------------------------------------


def _count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def matched_hidden(
    entity_param_count: int,
    entity_dim: int,
    action_dim: int,
    n_entities: int,
    *,
    n_layers: int = 2,
    activation: str = "gelu",
    lo: int = 16,
    hi: int = 2048,
) -> tuple[int, int]:
    """Pick the trunk ``hidden`` whose MonolithicWM param count is closest to
    ``entity_param_count`` at the given N (binary-search the monotone curve).

    Returns (best_hidden, best_param_count). Pure-ish: builds + inits tiny models
    to count params (needs jax/flax). Used by the mechanism-check at N_train.
    """
    import jax.numpy as _jnp

    key = jax.random.PRNGKey(0)
    dummy_ent = _jnp.zeros((2, n_entities, entity_dim))
    dummy_act = _jnp.zeros((2, action_dim))

    def count_for(hidden: int) -> int:
        model = MonolithicWM(
            entity_dim=entity_dim,
            action_dim=action_dim,
            n_entities=n_entities,
            hidden=int(hidden),
            n_layers=n_layers,
            activation=activation,
        )
        params = model.init(key, dummy_ent, dummy_act)["params"]
        return _count_params(params)

    # Binary search for the smallest hidden with count >= target, then compare
    # neighbours to take the closest.
    best_h, best_c, best_err = lo, count_for(lo), None
    a, b = lo, hi
    while a < b:
        mid = (a + b) // 2
        c = count_for(mid)
        err = abs(c - entity_param_count)
        if best_err is None or err < best_err:
            best_err, best_h, best_c = err, mid, c
        if c < entity_param_count:
            a = mid + 1
        else:
            b = mid
    # Check the final boundary too.
    c = count_for(a)
    if abs(c - entity_param_count) < (best_err if best_err is not None else 1 << 60):
        best_h, best_c = a, c
    return int(best_h), int(best_c)
