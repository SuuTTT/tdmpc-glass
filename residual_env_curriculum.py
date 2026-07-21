#!/usr/bin/env python3
"""Experiment #8: HARD-CONFIG CURRICULUM (prioritized resets) on top of the
in-loop residual env (exp#5 ResidualPickCube, alpha=1.0).

GOAL: lift PandaPickCube real success on the HARD-CONFIG TAIL by OVERSAMPLING
recently-FAILED (box0, target) configs during training. ~20% of episodes fail on
hard configs (box poses / target positions at reach-edge); a curriculum that
focuses resets on the failing tail should lift overall success.

This is a SEPARATE file from the shared residual_env.py (we do NOT edit the shared
file). It subclasses ResidualPickCube so the in-loop analytic option-controller +
Markov (s,z) obs + alpha schedule are all inherited UNCHANGED; we only override the
RESET / config-sampling to implement prioritized resets.

================  PRIORITIZED-RESET MECHANISM (per-env, JAX-functional)  ========
The brax training pipeline wraps the env with BraxAutoResetWrapper. We run it with
full_reset=True (set by the launcher), so env.reset(rng) is CALLED on every
autoreset, and the wrapper passes the key 'AutoResetWrapper_preserve_info' through
from the prior step (its documented curriculum hook). We thread a per-env FAILED-
CONFIG RING BUFFER through that key:

  preserve_info ('AutoResetWrapper_preserve_info', step-controlled) = {
    'ep_max_bt' : ()    running max box_target of the CURRENT episode,
    'fail_box0' : (K,3) ring buffer of box0 from recently-FAILED episodes,
    'fail_targ' : (K,3) ring buffer of target  from recently-FAILED episodes,
    'fail_n'    : ()     number of valid entries in the buffer (capped at K),
    'fail_ptr'  : ()     ring write pointer,
  }
  plain info (propagated through autoreset via where_done) = {
    'cur_box0_reset' : (3,)  box xyz chosen by reset for the CURRENT episode,
    'cur_targ_reset' : (3,)  target xyz chosen by reset for the CURRENT episode,
  }

WRAPPER GOTCHA (full_reset=True): BraxAutoResetWrapper OVERWRITES the whole
preserve_info subtree with the PRIOR STEP's value for every env, so reset()'s
writes to preserve_info are DISCARDED -- the failed-buffer must be maintained
entirely in step(). reset()'s chosen config is instead published in PLAIN info
('cur_box0_reset'), which propagates correctly via where_done for done envs; step()
reads it to know the current episode's config when pushing failures.

reset(rng):  with prob P_REPLAY (~0.5) AND the buffer is non-empty, draw (box0,
  target) from a uniformly-random valid slot of THIS env's failed buffer (replay a
  hard config); otherwise sample box0/target UNIFORMLY (identical distribution to
  the stock PandaPickCube.reset). We build a valid State via super().reset (gets
  all residual/ctrl info threaded) then OVERWRITE the box qpos + target mocap with
  the chosen config and recompute obs. The chosen config is published in
  cur_box0_reset/cur_targ_reset (plain info).

step:  track running ep_max_bt = max(ep_max_bt, box_target). On done, detect the
  FAILURE of the just-finished episode (ep_max_bt < SUCCESS_THRESH) and, if failed,
  PUSH cur_box0_reset/cur_targ_reset into the ring buffer (advancing ptr / count).
  This push happens in step (when done is known and preserve_info is step-owned).

EVAL HONESTY:  curriculum is TRAIN-ONLY. We never enable it at eval time. Eval uses
the stock uniform config distribution (eval_residual_curve.py builds the plain
ResidualPickCube via residual_patch / residual_env, NOT this class). The reported
real-success number is therefore on the held-out UNIFORM distribution.

ENABLE:  set env var RES_CURRICULUM=1 (residual_patch_curriculum.py swaps in this
class). P_REPLAY via RES_REPLAY_P (default 0.5); buffer size via RES_FAILBUF_K
(default 64); success threshold matches the metric (box_target>=0.9).
"""
import os
from typing import Any, Dict

import jax
import jax.numpy as jp

from mujoco_playground._src.mjx_env import State

import residual_env as RE  # the shared exp#5 in-loop residual env (unedited)

# ---- curriculum hyperparameters (env-var tunable) ----
P_REPLAY = float(os.environ.get("RES_REPLAY_P", "0.5"))      # frac of resets from failed buffer
FAILBUF_K = int(os.environ.get("RES_FAILBUF_K", "64"))       # per-env ring-buffer capacity
SUCCESS_THRESH = float(os.environ.get("RES_SUCC_THRESH", "0.9"))  # == real-success metric

# uniform config bounds (IDENTICAL to stock PandaPickCube.reset)
_BOX_LO = jp.array([-0.2, -0.2, 0.0])
_BOX_HI = jp.array([0.2, 0.2, 0.0])
_TARG_LO = jp.array([-0.2, -0.2, 0.2])
_TARG_HI = jp.array([0.2, 0.2, 0.4])

_PI_KEY = "AutoResetWrapper_preserve_info"


class CurriculumResidualPickCube(RE.ResidualPickCube):
  """ResidualPickCube + prioritized resets (oversample recently-failed configs).

  Inherits the in-loop residual (alpha schedule, phase-z obs, milestone shaping)
  verbatim; only reset/config-sampling and a small failure-tracking hook in step
  are added.
  """

  # ---------- helpers ----------
  def _uniform_config(self, rng):
    rb, rt = jax.random.split(rng)
    box0 = jax.random.uniform(rb, (3,), minval=_BOX_LO, maxval=_BOX_HI) + self._init_obj_pos
    targ = jax.random.uniform(rt, (3,), minval=_TARG_LO, maxval=_TARG_HI) + self._init_obj_pos
    return box0, targ

  def _empty_buffer(self):
    return {
        "ep_max_bt": jp.array(0.0),     # running max box_target, current episode
        "cur_box0": jp.zeros((3,)),     # config of the currently-running episode
        "cur_targ": jp.zeros((3,)),
        "fail_box0": jp.zeros((FAILBUF_K, 3)),
        "fail_targ": jp.zeros((FAILBUF_K, 3)),
        "fail_n": jp.array(0, dtype=jp.int32),
        "fail_ptr": jp.array(0, dtype=jp.int32),
    }

  def _place_config(self, state: State, box0, targ) -> State:
    """Overwrite the box qpos + target mocap of an existing reset State with the
    chosen (box0, targ), and recompute obs. Reuses all base machinery."""
    data = state.data
    qpos = data.qpos.at[self._obj_qposadr : self._obj_qposadr + 3].set(box0)
    data = data.replace(qpos=qpos)
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(targ)
    )
    info = dict(state.info)
    info["target_pos"] = targ
    # recompute base obs (reward not used at reset), then re-augment with phase z
    base_obs = self._get_obs(data, info)
    cs = info["ctrl"]
    # rebuild a transient state view for _aug_obs (needs data.site_xpos/xpos);
    # site_xpos is stale until a forward step, but _aug_obs only needs box vs site
    # direction which the base reset obs already approximates; recompute via data.
    tmp = State(data, base_obs, state.reward, state.done, state.metrics, info)
    obs = self._aug_obs(base_obs, tmp, cs)
    return State(data, obs, state.reward, state.done, state.metrics, info)

  # ---------- API ----------
  # ARCHITECTURE NOTE (why selection happens in step, not reset):
  # BraxAutoResetWrapper calls env.reset(rng) with NO access to prior info, and then
  # OVERWRITES the preserve_info subtree (our failed-buffer) with the prior step's
  # value. So reset() cannot SEE the failed buffer -> it cannot choose a replay
  # config. We therefore let reset sample UNIFORMLY (and publish that config), and
  # do the CURRICULUM SELECTION + box re-placement at the FIRST step of each episode
  # inside step(), where the failed buffer IS available (it is step-owned). The box
  # has moved at most one tiny ctrl step from the uniform reset, so re-placing its
  # qpos/mocap at is_new is equivalent to choosing the config at reset.

  def reset(self, rng: jax.Array) -> State:
    # base residual reset (uniform box/target + all ctrl/residual info threaded)
    rng, rng_unif = jax.random.split(rng, 2)
    state = super().reset(rng)
    u_box0, u_targ = self._uniform_config(rng_unif)
    state = self._place_config(state, u_box0, u_targ)

    info = dict(state.info)
    # publish the uniform config (may be overridden by step's curriculum draw)
    info["cur_box0_reset"] = u_box0
    info["cur_targ_reset"] = u_targ
    # shape-correct buffer (its VALUE is overwritten by the wrapper from prior step)
    info[_PI_KEY] = self._empty_buffer()
    return State(state.data, state.obs, state.reward, state.done,
                 state.metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    # EPISODE BOUNDARY: this env is wrapped by EpisodeWrapper, which sets the
    # episode-length `done` AFTER our step returns, so nxt.done here is only the raw
    # out_of_bounds done. AutoResetWrapper resets info['steps'] to 0 on done BEFORE
    # calling our step, so the FIRST step of a new episode has state.info['steps']<0.5.
    is_new = state.info.get("steps", jp.array(1.0)) < 0.5

    buf_in = state.info.get(_PI_KEY, None)
    if buf_in is None:
      # curriculum hook missing (full_reset off / plain eval) -> plain residual
      return super().step(state, action)
    buf = dict(buf_in)

    # ===== (A) at episode start: CURRICULUM-SELECT this episode's config =====
    # uniform fallback = the config reset already placed (published in plain info)
    u_box0 = state.info["cur_box0_reset"]
    u_targ = state.info["cur_targ_reset"]
    rng = state.info.get("rng", jax.random.PRNGKey(0))
    rng, rk_pick, rk_slot = jax.random.split(rng, 3)
    n = buf["fail_n"]
    have_fail = n > 0
    slot = jax.random.randint(rk_slot, (), 0, jp.maximum(n, 1))
    r_box0 = buf["fail_box0"][slot]
    r_targ = buf["fail_targ"][slot]
    do_replay = is_new & (jax.random.uniform(rk_pick, ()) < P_REPLAY) & have_fail
    sel_box0 = jp.where(do_replay, r_box0, u_box0)
    sel_targ = jp.where(do_replay, r_targ, u_targ)

    # re-place the box qpos + target mocap for the SELECTED config (only at is_new).
    # Editing state BEFORE stepping so the option-controller acts on the right goal.
    data = state.data
    new_qpos = data.qpos.at[self._obj_qposadr : self._obj_qposadr + 3].set(sel_box0)
    qpos = jp.where(is_new, new_qpos, data.qpos)
    new_mocap = data.mocap_pos.at[self._mocap_target, :].set(sel_targ)
    mocap = jp.where(is_new, new_mocap, data.mocap_pos)
    data = data.replace(qpos=qpos, mocap_pos=mocap)
    si = dict(state.info)
    si["target_pos"] = jp.where(is_new, sel_targ, state.info["target_pos"])
    si["rng"] = rng
    state = State(data, state.obs, state.reward, state.done, state.metrics, si)

    # ===== (B) take the (residual + controller) step on the corrected state =====
    nxt = super().step(state, action)
    info = dict(nxt.info)
    buf = dict(info[_PI_KEY])  # super().step threads our (corrected) info through

    # ===== (C) push the JUST-FINISHED (prior) episode if it FAILED =====
    prev_max = buf["ep_max_bt"]
    prev_box0 = buf["cur_box0"]
    prev_targ = buf["cur_targ"]
    had_prior = jp.any(prev_box0 != 0.0)  # real configs are offset by _init_obj_pos
    failed = is_new & had_prior & (prev_max < SUCCESS_THRESH)
    ptr = buf["fail_ptr"]
    pb0 = jp.where(failed, prev_box0, buf["fail_box0"][ptr])
    pt0 = jp.where(failed, prev_targ, buf["fail_targ"][ptr])
    buf["fail_box0"] = buf["fail_box0"].at[ptr].set(pb0)
    buf["fail_targ"] = buf["fail_targ"].at[ptr].set(pt0)
    buf["fail_ptr"] = jp.where(failed, (ptr + 1) % FAILBUF_K, ptr).astype(jp.int32)
    buf["fail_n"] = jp.where(
        failed, jp.minimum(buf["fail_n"] + 1, FAILBUF_K), buf["fail_n"]).astype(jp.int32)

    # ===== (D) update current-episode bookkeeping =====
    buf["cur_box0"] = jp.where(is_new, sel_box0, prev_box0)
    buf["cur_targ"] = jp.where(is_new, sel_targ, prev_targ)
    bt = nxt.metrics["box_target"]
    buf["ep_max_bt"] = jp.where(is_new, bt, jp.maximum(prev_max, bt))

    info[_PI_KEY] = buf
    # keep the published config in sync with the SELECTED one (for diagnostics)
    info["cur_box0_reset"] = jp.where(is_new, sel_box0, info["cur_box0_reset"])
    info["cur_targ_reset"] = jp.where(is_new, sel_targ, info["cur_targ_reset"])
    return State(nxt.data, nxt.obs, nxt.reward, nxt.done, nxt.metrics, info)
