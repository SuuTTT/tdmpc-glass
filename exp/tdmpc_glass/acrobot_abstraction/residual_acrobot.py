#!/usr/bin/env python3
"""Abstraction-in-the-loop residual env for AcrobotSwingup, mirroring the pendulum
residual (Part 45) but for the harder two-link underactuated swing-up.

Executed elbow command (what acrobot.Balance.step receives):
    a_t = clip( a_ctrl(obs_t) + alpha * pi_res(obs_aug_t), -1, 1 )

  * a_ctrl  = analytic Spong energy-shaping swing-up controller (acrobot_controller),
              PURE state feedback (collocated PFL pump + PD balance), validated alone.
  * pi_res  = brax PPO policy output in [-1,1].
  * alpha   = fixed authority (default 1.0; env var RES_ALPHA).
  * obs_aug = base 6-dim obs AUGMENTED with the controller PHASE one-hot
              (PUMP/BALANCE) so pi_res is Markov in (s, phase).
Reward UNCHANGED (true tolerance task return). Fair test: does a learned residual
on the analytic prior reach TD-MPC2's return faster/higher on the harder task?
"""
import os
from typing import Optional
import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src.dm_control_suite.acrobot import (
    Balance, default_config as _acrobot_default_config,
)
import acrobot_controller as ACTRL

RES_ALPHA = float(os.environ.get("RES_ALPHA", "1.0"))
_N_PHASES = 2  # PUMP=0, BALANCE=1
_BASE_OBS = 6


def default_config() -> config_dict.ConfigDict:
  cfg = _acrobot_default_config()
  cfg.impl = "jax"
  return cfg


class ResidualAcrobot(Balance):
  """AcrobotSwingup where the policy output is a residual on a live analytic
  controller, with the controller phase appended to the observation."""

  def __init__(self, config=None, config_overrides=None):
    if config is None:
      config = default_config()
    super().__init__(sparse=False, config=config, config_overrides=config_overrides)
    self._alpha = RES_ALPHA

  @property
  def observation_size(self) -> int:
    return _BASE_OBS + _N_PHASES

  def _aug_obs(self, base_obs):
    _a, phase = ACTRL.controller(base_obs)
    onehot = jax.nn.one_hot(phase, _N_PHASES)
    return jp.concatenate([base_obs, onehot], axis=-1)

  def reset(self, rng: jax.Array):
    state = super().reset(rng)
    return state.replace(obs=self._aug_obs(state.obs))

  def step(self, state, action: jax.Array):
    base_obs = state.obs[..., :_BASE_OBS]
    a_ctrl, _phase = ACTRL.controller(base_obs)
    a_exec = jp.clip(a_ctrl.reshape(action.shape) + self._alpha * action, -1.0, 1.0)
    next_state = super().step(state.replace(obs=base_obs), a_exec)
    return next_state.replace(obs=self._aug_obs(next_state.obs))
