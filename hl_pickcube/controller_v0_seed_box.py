#!/usr/bin/env python3
"""SCRIPTED (hand-coded, no learning) pick-and-place controller for
mujoco_playground PandaPickCube.

ACTION INTERFACE (from pick.py / mjx_panda.xml):
  - 8 position actuators. ctrl[0:7] = target joint angles (rad); ctrl[7] =
    gripper finger width (0=closed .. 0.04=open).
  - env.step: ctrl = clip(prev_ctrl + action*action_scale, lowers, uppers),
    action_scale=0.04 => ctrl moves <=0.04 (arm rad) / 0.0016 (grip m) per step;
    gripper needs ~25 steps to fully open<->close. action = per-step DELTA.
  - JOINT-SPACE position control => CPU MuJoCo damped-least-squares Jacobian IK
    converts a desired cartesian gripper position into joint targets; we emit the
    clipped joint delta toward those targets.

GEOMETRY: box half-extents (0.02,0.02,0.03) => 4cm wide, center z=0.03. Finger
pads ~0.095m apart open; closing clamps the cube. Site ~ fingertip level.

SUCCESS = box_target>=0.9 (box at target AND grasped); needs pos_err<=~2cm and
reached_box (norm(box-gripper_site)<0.012).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")

import numpy as np
import mujoco


class ScriptedPickController:
    def __init__(self, env):
        self.env = env
        self.m = env.mj_model
        self.gid = self.m.site("gripper").id
        self.lowers = np.array(env._lowers)
        self.uppers = np.array(env._uppers)
        self.scale = float(env._action_scale)
        self.arm_qadr = np.array(env._robot_arm_qposadr)
        self.obj_body = env._obj_body
        self._d = mujoco.MjData(self.m)
        self.reset()

    def reset(self):
        self.phase = "ABOVE"
        self.timer = 0
        self.grasp_cmd = 0.04
        self.grip_box_off = np.zeros(3)
        self.prev_fw = 0.04
        self.regrasped = False
        self.lock_xy = None

    def _ik(self, q_arm_init, target_xyz, q_full_ref, iters=120):
        d = self._d
        q = np.array(q_full_ref, dtype=float).copy()
        q_arm = np.array(q_arm_init, dtype=float).copy()
        jacp = np.zeros((3, self.m.nv))
        for _ in range(iters):
            q[self.arm_qadr] = q_arm
            d.qpos[:] = q
            mujoco.mj_forward(self.m, d)
            cur = d.site_xpos[self.gid].copy()
            err = target_xyz - cur
            if np.linalg.norm(err) < 5e-5:
                break
            mujoco.mj_jacSite(self.m, d, jacp, None, self.gid)
            J = jacp[:, self.arm_qadr]
            lam = 0.08
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
            q_arm = np.clip(q_arm + dq, self.lowers[:7], self.uppers[:7])
        return q_arm

    @staticmethod
    def _ratelimit(cur, goal, max_step):
        v = goal - cur
        n = np.linalg.norm(v)
        return goal if n < max_step else cur + v * (max_step / n)

    def __call__(self, state):
        d = state.data
        box = np.array(d.xpos[self.obj_body])
        grip = np.array(d.site_xpos[self.gid])
        target = np.array(state.info["target_pos"])
        cur_ctrl = np.array(d.ctrl)
        q_arm_cur = np.array(d.qpos)[self.arm_qadr]
        q_full = np.array(d.qpos)
        fw = float(np.array(d.qpos)[7])

        HOVER = 0.10
        GRASP_Z = box[2]
        horiz_err = np.linalg.norm(grip[:2] - box[:2])

        if self.phase == "ABOVE":
            self.grasp_cmd = 0.04
            goal = np.array([box[0], box[1], box[2] + HOVER])
            # Center tightly at hover height BEFORE descending. Crucial: if we
            # descend while still off-center, a finger hits the cube side and
            # shoves it away (positive feedback if xy chases the cube).
            if (horiz_err < 0.006 and abs(grip[2] - goal[2]) < 0.025) or self.timer > 22:
                self.lock_xy = box[:2].copy()      # freeze grasp xy
                self.phase = "DESCEND"; self.timer = 0
        elif self.phase == "DESCEND":
            self.grasp_cmd = 0.04
            # straight down on LOCKED xy so the open fingers drop AROUND the cube
            goal = np.array([self.lock_xy[0], self.lock_xy[1], GRASP_Z])
            if ((grip[2] - box[2]) < 0.006) or self.timer > 24:
                self.phase = "GRASP"; self.timer = 0
        elif self.phase == "GRASP":
            self.grasp_cmd = 0.0
            goal = np.array([self.lock_xy[0], self.lock_xy[1], GRASP_Z])
            # gripper is rate-limited (~25 steps open->closed); stop as soon as
            # it has clamped (width plateaus while still > 0 = blocked by cube).
            plateaued = self.timer > 22 and abs(fw - self.prev_fw) < 1e-3
            if (plateaued and fw > 0.004) or self.timer > 30:
                self.grip_box_off = grip - box
                self.phase = "LIFT"; self.timer = 0
        elif self.phase == "LIFT":
            # minimal vertical clearance, then go straight to transport (the
            # target is elevated, so TO_TARGET keeps lifting on the diagonal).
            self.grasp_cmd = 0.0
            goal = np.array([box[0], box[1], box[2] + 0.09])
            # one re-approach if the cube failed to come up with the gripper.
            if self.timer > 8 and box[2] < 0.045 and not self.regrasped:
                self.regrasped = True
                self.phase = "ABOVE"; self.timer = 0
            elif grip[2] > box[2] + 0.06 or self.timer > 8:
                self.phase = "TO_TARGET"; self.timer = 0
        elif self.phase == "TO_TARGET":
            self.grasp_cmd = 0.0
            # closed-loop: drive the BOX error to zero (more precise than a
            # static offset). goal = current gripper + remaining box->target err.
            err = target - box
            final = grip + err
            # rate-limit the cartesian goal so the held cube swings less and
            # tracks the target more precisely (best observed final placement).
            goal = self._ratelimit(grip, final, 0.09)
            if np.linalg.norm(err) < 0.012:
                self.phase = "HOLD"; self.timer = 0
        else:  # HOLD
            self.grasp_cmd = 0.0
            goal = grip + (target - box)   # keep correcting

        self.timer += 1
        self.prev_fw = fw

        q_arm_des = self._ik(q_arm_cur, goal, q_full)
        ctrl_des = np.concatenate([q_arm_des, [self.grasp_cmd]])
        action = np.clip((ctrl_des - cur_ctrl) / self.scale, -1.0, 1.0)
        return action.astype(np.float32)
