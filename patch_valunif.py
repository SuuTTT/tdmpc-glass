#!/usr/bin/env python3
"""Idempotent patcher: add a VALUE-AWARE anti-collapse arm (valunif) to tdmpc2.py."""
import io, sys

F = "/root/helios-rl/src/helios/algorithms/tdmpc2.py"
src = io.open(F, encoding="utf-8").read()

# ---- Edit 1: add value_aware_uniformity_loss after vicreg_loss --------------
FUNC = '''

def value_aware_uniformity_loss(z, v, t: float = 2.0, attract: float = 0.5):
    """VALUE-AWARE anti-collapse (valunif).

    Plain Wang&Isola UNIFORMITY repels *all* pairs equally, which spreads apart
    states that should be VALUE-CLOSE and destroys value-sufficiency (the failure
    mode we isolated on DMControl). This term makes the anti-collapse signal
    value-aware by GATING the repulsion with value distance and adding a matched
    attraction:

      v_i    : per-state scalar value V(s_i)=max(min_k Q_k(z_i, pi(z_i)),0), STOP-GRAD
               (the same quantity used to form the TD target; used here only as a label).
      sigma  : batch std of v (adaptive scale, no extra hyperparameter).
      close  = exp(-|v_i-v_j|/sigma)   ~1 value-similar, ~0 value-different
      far    = 1 - close               repulsion weight (value-different pairs only)

      repel  = log E_{i!=j} far_ij * exp(-t ||z_i-z_j||^2)      (uniformity, value-gated)
      attr   = E_{i!=j} close_ij * ||z_i-z_j||^2                (pull value-close together)
      loss   = repel + attract * attr

    Grad flows to the ENCODER through ||z_i-z_j||^2 (z carries grad); v is stop-grad.
    Collapse resistance is preserved (value-different pairs are pushed apart) WITHOUT
    repelling value-similar states; if uniformity hurts because it ignores value
    structure, this is the matched fix.
    """
    B = z.shape[0]
    g = z @ z.T
    dd = jnp.diag(g)
    D = jnp.clip(dd[:, None] + dd[None, :] - 2.0 * g, 0.0, None)   # ||z_i - z_j||^2
    vd = jnp.abs(v[:, None] - v[None, :])                          # value distance
    sigma = jnp.maximum(jnp.std(v), 1e-3)
    close = jnp.exp(-vd / sigma)
    far = 1.0 - close
    mask = 1.0 - jnp.eye(B)
    # value-gated repulsion (uniformity restricted to value-different pairs)
    logits = jnp.where(mask > 0, jnp.log(far + 1e-6) - t * D, -jnp.inf)
    repel = jax.nn.logsumexp(logits) - jnp.log(jnp.sum(mask * far) + 1e-6)
    # value-close attraction
    wclose = mask * close
    attr = jnp.sum(wclose * D) / (jnp.sum(wclose) + 1e-6)
    return repel + attract * attr

'''
anchor1 = "    cov_loss = jnp.sum(off ** 2) / D\n    return var_loss + cov_loss\n"
if "def value_aware_uniformity_loss" not in src:
    assert anchor1 in src, "anchor1 not found"
    src = src.replace(anchor1, anchor1 + FUNC, 1)
    print("Edit 1 applied: added value_aware_uniformity_loss")
else:
    print("Edit 1 skipped: function already present")

# ---- Edit 2: wire the env-gated block ---------------------------------------
old_block = '''        _vicreg_coef = float(_os.environ.get("VICREG_COEF", "0.0"))
        if _unif_coef > 0.0 or _vicreg_coef > 0.0:
            _zf = z_all.reshape(-1, z_all.shape[-1])
            if _unif_coef > 0.0:
                total = total + _unif_coef * uniformity_loss(_zf, t=_unif_t)
            if _vicreg_coef > 0.0:
                total = total + _vicreg_coef * vicreg_loss(_zf)'''
new_block = '''        _vicreg_coef = float(_os.environ.get("VICREG_COEF", "0.0"))
        _valunif_coef = float(_os.environ.get("VALUNIF_COEF", "0.0"))
        if _unif_coef > 0.0 or _vicreg_coef > 0.0 or _valunif_coef > 0.0:
            _zf = z_all.reshape(-1, z_all.shape[-1])
            if _unif_coef > 0.0:
                total = total + _unif_coef * uniformity_loss(_zf, t=_unif_t)
            if _vicreg_coef > 0.0:
                total = total + _vicreg_coef * vicreg_loss(_zf)
            if _valunif_coef > 0.0:
                # value-aware anti-collapse: derive a per-state value label (stop-grad)
                # from the current pi & Q nets, exactly as the TD target does.
                _zsg = jax.lax.stop_gradient(_zf)
                _pi_m = jnp.tanh(pi_net.apply(params["pi"], _zsg)[0])
                _q_log = q_net.apply(jax.lax.stop_gradient(params["q"]), _zsg, _pi_m)
                _v = jax.lax.stop_gradient(jnp.maximum(jnp.min(two_hot_inv(_q_log), -1), 0.0))
                _valunif_att = float(_os.environ.get("VALUNIF_ATTRACT", "0.5"))
                total = total + _valunif_coef * value_aware_uniformity_loss(
                    _zf, _v, t=_unif_t, attract=_valunif_att)'''
if "_valunif_coef" not in src:
    assert old_block in src, "old_block not found"
    src = src.replace(old_block, new_block, 1)
    print("Edit 2 applied: wired VALUNIF_COEF block")
else:
    print("Edit 2 skipped: block already wired")

io.open(F, "w", encoding="utf-8").write(src)
print("Wrote", F)
