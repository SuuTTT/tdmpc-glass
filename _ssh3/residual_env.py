#!/usr/bin/env python3
"""Experiment #5: analytic option-controller KEPT LIVE as a residual base, with a
learned residual policy pi_res given GROWING authority via an annealed alpha, and
the controller PHASE z fed EXPLICITLY into the observation so pi_res is Markov in
(s, z).

Executed action (what the BASE PandaPickCube.step receives):

    a_t = clip( a_option(s_t, z_t) + alpha(t) * a_res_t , -1, 1 )

where
  * a_option = the analytic phase controller (param_controller.ParamPick, v9 BEST
    knobs via KV_BASE, d==0) executed live; z_t = its phase/option carry state.
  * a_res_t  = the action passed to THIS env's .step (i.e. brax PPO's policy
    output, in [-1,1]^8). PPO optimizes pi_res via gradient RL.
  * alpha(t) = clip(global_ctrl_steps / ANNEAL_STEPS, ALPHA_MIN, ALPHA_MAX),
    linearly annealing 0 -> 1 so early training inherits the controller's
    competence (fast escape) and late training gives pi_res full authority
    (exceed the 0.24 structural cap).

KEY FIX vs experiment #4 (BC/DAPG of a memoryless clone): the option-controller is
NON-MARKOV (action depends on hidden phase z). We do NOT distill it. Instead we
keep it live AND feed z into pi_res's observation, so the residual sees the same
phase context the controller acts on -> the (s, z)-conditioned problem IS Markov.

OBS AUGMENTATION (Markov-correct conditioning), appended to the 66-dim base obs:
  * one-hot phase z (7)         -> which option is executing
  * sub-goal offset goal - site (3) -> the controller's current cartesian target
  * grip command setpoint (1)   -> the controller's current gripper intent
  -> augmented obs = 66 + 11 = 77 dims.

SUB-GOAL MILESTONE SHAPING (Markov-checkable; avoids raw-action imitation): we ADD
small bonuses for reaching option milestones (reached box, grasped, lifted above a
threshold) on top of the env's task reward. SELECTION / REPORTING is always on the
TRUE metric box_target>=0.9 (via baselines_ppo_sac/eval_ckpts.py), never shaped
return.

AUTORESET / z THREADING: z lives in state.info. EpisodeWrapper sets info['steps']
to 0 at the first step of every episode (initial reset AND after BraxAutoReset).
We reset z to its init whenever info['steps']==0, so z is correctly re-initialized
on every (auto)reset regardless of full_reset. A monotonic per-env counter
info['ctrl_gsteps'] (never reset; survives default autoreset which keeps info)
drives the alpha schedule as a proxy for global training progress.
"""
import os
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State
from mujoco_playground._src.manipulation.franka_emika_panda.pick import (
    PandaPickCube,
    default_config as _pick_default_config,
)
import param_controller as PC

# alpha schedule (env-var tunable for ablations)
ANNEAL_STEPS = float(os.environ.get("RES_ANNEAL_STEPS", "16000000"))  # per-env ctrl steps to reach ALPHA_MAX
ALPHA_MIN = float(os.environ.get("RES_ALPHA_MIN", "0.0"))
ALPHA_MAX = float(os.environ.get("RES_ALPHA_MAX", "1.0"))
ALPHA_FIXED = os.environ.get("RES_ALPHA_FIXED", "")  # if set, constant alpha (ablation)
# learned per-state gate mode (#5b): policy outputs an EXTRA gate channel;
# alpha(s,z) = sigmoid(GATE_SCALE*gate_logit + GATE_BIAS), per-state authority.
GATE_MODE = os.environ.get("RES_GATE_MODE", "0") == "1"
GATE_SCALE = float(os.environ.get("RES_GATE_SCALE", "6.0"))
GATE_BIAS = float(os.environ.get("RES_GATE_BIAS", "-2.5"))  # init alpha=sigmoid(-2.5)~0.076 -> ~controller baseline

# milestone shaping coefficients (added on top of base task reward)
W_REACH = float(os.environ.get("RES_W_REACH", "0.0"))   # reached_box already shaped in base; keep 0 by default
W_GRASP = float(os.environ.get("RES_W_GRASP", "0.2"))
W_LIFT = float(os.environ.get("RES_W_LIFT", "0.3"))
LIFT_THRESH = float(os.environ.get("RES_LIFT_THRESH", "0.06"))

_N_PHASES = 7  # APPROACH..HOLD


class ResidualPickCube(PandaPickCube):
  """PandaPickCube where the policy output is a RESIDUAL on top of a live analytic
  option-controller, with phase z fed into the obs (Markov in (s,z))."""

  def __init__(self, config=None, config_overrides=None, sample_orientation=False):
    if config is None:
      config = _pick_default_config()
    super().__init__(config=config, config_overrides=config_overrides,
                     sample_orientation=sample_orientation)
    # build the analytic option-controller bound to THIS env's model params
    self._opt = PC.make_param_controller(self)
    # fixed v9 BEST knob vector (d==0 -> KV_BASE)
    self._kv = PC.d_to_knobs(jp.zeros((PC.KV_DIM,)))

  # ----- helpers -----
  def _ctrl_init(self):
    return self._opt.init_state()

  def _aug_obs(self, base_obs, state, cs):
    """Append Markov-completing context: one-hot phase + sub-goal + grip intent."""
    phase = cs["phase"]
    onehot = jax.nn.one_hot(phase, _N_PHASES)
    site = state.data.site_xpos[self._gripper_site]
    # recompute the controller's current cartesian goal for THIS phase cheaply:
    box = state.data.xpos[self._obj_body]
    target = state.info["target_pos"]
    # approximate current sub-goal offset: use box vs target depending on phase>=LIFT
    pre_grasp = phase < PC.GRASP
    goal_xyz = jp.where(pre_grasp, box, target)
    subgoal = goal_xyz - site                      # (3,) direction to current focus
    grip = cs["grip_cmd"][None]                    # (1,)
    return jp.concatenate([base_obs, onehot, subgoal, grip])

  # ----- API -----
  def reset(self, rng: jax.Array) -> State:
    state = super().reset(rng)
    cs = self._ctrl_init()
    info = dict(state.info)
    info["ctrl"] = cs
    info["ctrl_gsteps"] = jp.array(0.0)
    info["alpha"] = jp.array(ALPHA_MIN)
    info["phase"] = cs["phase"]
    obs = self._aug_obs(state.obs, state, cs)
    return State(state.data, obs, state.reward, state.done, state.metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    info = state.info
    cs = info["ctrl"]
    # reset z on the first step of every (auto)reset episode
    is_new = info.get("steps", jp.array(0.0)) < 0.5  # EpisodeWrapper sets steps=0 on reset
    init_cs = self._ctrl_init()
    cs = jax.tree_util.tree_map(
        lambda new, old: jp.where(is_new, new, old), init_cs, cs)

    # analytic option action + advance phase
    a_opt, cs_next = self._opt.act(state, cs, self._kv)

    # alpha schedule from monotonic per-env ctrl step counter
    gsteps = info["ctrl_gsteps"]
    if ALPHA_FIXED:
      alpha = jp.array(float(ALPHA_FIXED))
    else:
      frac = jp.clip(gsteps / ANNEAL_STEPS, 0.0, 1.0)
      alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * frac

    if GATE_MODE:
      a_res = action[..., :-1]               # first nu dims: residual
      gate_logit = action[..., -1]           # last dim: per-state gate
      alpha = jax.nn.sigmoid(GATE_SCALE * gate_logit + GATE_BIAS)
    else:
      a_res = action  # PPO policy output in [-1,1]^nu
    executed = jp.clip(a_opt + alpha * a_res, -1.0, 1.0)

    # step the BASE env with the executed (controller + residual) action
    base_next = super().step(state, executed)

    # milestone shaping (Markov-checkable), added to base reward
    box = base_next.data.xpos[self._obj_body]
    site = base_next.data.site_xpos[self._gripper_site]
    fw = base_next.data.qpos[7]
    d = jp.linalg.norm(box - site)
    grasped = ((fw < PC.FIX_close_threshold) & (fw > 0.005) & (d < 0.04)).astype(jp.float32)
    lift = jp.clip(box[2] - self._init_obj_pos[2], 0.0, None)
    lifted = ((lift > LIFT_THRESH) & (d < 0.05)).astype(jp.float32)
    reached = base_next.info["reached_box"]
    shaped = (W_REACH * reached + W_GRASP * grasped + W_LIFT * lifted)
    reward = base_next.reward + shaped

    # update threaded info
    new_info = dict(base_next.info)
    new_info["ctrl"] = cs_next
    new_info["ctrl_gsteps"] = gsteps + 1.0
    new_info["alpha"] = alpha
    new_info["phase"] = cs["phase"]  # phase that produced a_opt this step (for gate-by-phase analysis)

    obs = self._aug_obs(base_next.obs, base_next, cs_next)
    return State(base_next.data, obs, reward, base_next.done,
                 base_next.metrics, new_info)

  @property
  def observation_size(self):
    return 66 + _N_PHASES + 3 + 1  # 77

  @property
  def action_size(self):
    base = super().action_size
    return base + 1 if GATE_MODE else base
