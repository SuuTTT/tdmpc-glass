#!/usr/bin/env python3
"""Analytic Spong energy-shaping swing-up controller (collocated partial feedback
linearization) for the mujoco_playground dm_control_suite AcrobotSwingup.

PHYSICS (xmls/acrobot.xml), measured empirically (diag_acrobot.py):
  * Two slender capsule links, each mass m=1, length l=1 (com at lc=0.5), hinge
    damping b=0.05 on both joints. UNDERACTUATED: torque only on the ELBOW
    (joint 2). gear=2, ctrlrange [-1,1]  => applied elbow torque tau = 2*u N*m.
  * qpos = [q1 (shoulder), q2 (elbow)] measured from UPRIGHT: q1=q2=0 -> both arms
    straight up, tip (0,0,4), reward 1.0 (target z=4).
  * SIGN: +u increases elbow velocity (measured u=+1 -> velbow +2.2 from rest).
  * obs (6) = [sin q1, sin(q1+q2), cos q1, cos(q1+q2), q1dot, q2dot]
      q1     = atan2(obs0, obs2)
      q1+q2  = atan2(obs1, obs3)  -> q2 = wrap((q1+q2) - q1)

STANDARD ACROBOT MODEL (Spong 1995) with our constants:
  m1=m2=1, l1=l2=1, lc1=lc2=0.5, I1=I2=(1/12) m l^2 = 1/12, g=9.81.
  Mass matrix  M(q2) = [[a+2b cos q2, c+b cos q2],[c+b cos q2, c]]
    a = I1 + I2 + m1 lc1^2 + m2(l1^2 + lc2^2)
    b = m2 l1 lc2
    c = I2 + m2 lc2^2
  Gravity (PE measured from UPRIGHT, so torque pushes AWAY from up):
    The standard Spong G is for the hanging-down reference; here up is q=0.
    Potential U = -[ (m1 lc1 + m2 l1) g cos q1 + m2 lc2 g cos(q1+q2) ]  (max at q=0)
    => G1 = dU/dq1 =  (m1 lc1 + m2 l1) g sin q1 + m2 lc2 g sin(q1+q2)
       G2 = dU/dq2 =  m2 lc2 g sin(q1+q2)
  Coriolis terms C(q,qd) for the 2-link chain (standard).

COLLOCATED PFL energy swing-up (Spong):
  Choose desired actuated (elbow) acceleration
      v = -kp2*q2 - kd2*q2dot + kE * Etilde * q1dot
  (Etilde = E - E_des; the last term injects/removes energy via the shoulder DOF.)
  Then the elbow torque that realizes elbow-accel = v, after eliminating the
  unactuated DOF, is the collocated PFL law:
      tau = Mbar * v + Hbar
  where Mbar = M22 - M21*M11^{-1}*M12,  Hbar = h2 + G2 - M21*M11^{-1}*(h1+G1).
  We map tau -> u via u = clip(tau / GEAR, -1, 1).

BALANCE near the top: PD/LQR on the 4-state (bounded elbow torque).
Phase z in {0:PUMP,1:BALANCE} exposed for the Markov residual. jit/vmap-able.
"""
import jax.numpy as jp

# physics constants
M1 = M2 = 1.0
L1 = L2 = 1.0
LC1 = LC2 = 0.5
G = 9.81
I1 = I2 = (1.0 / 12.0)
B_DAMP = 0.05
GEAR = 2.0
UMAX = 1.0

A_M = I1 + I2 + M1 * LC1**2 + M2 * (L1**2 + LC2**2)
B_M = M2 * L1 * LC2
C_M = I2 + M2 * LC2**2

# Energy at upright = 0 (PE max=0 at up, KE=0). E_des = 0.
E_DES = 0.0

# --- gains (KE=8 reaches top energy in ~80% of episodes given enough time;
#  the ~10s/1000-step budget is too short for the +-2 N*m torque to FULLY swing
#  up + settle, so controller-alone return is low. The residual learns the
#  faster/cleaner timing on top of this analytic pump.) ---
KE = 8.0          # energy injection gain
KP2 = 16.0        # elbow regulation P (keep elbow near straight during pump)
KD2 = 2.0         # elbow regulation D
# balance PD
KP_S = 22.0; KP_E2 = 18.0; KD_S = 6.0; KD_E2 = 5.0
CATCH_ANG = jp.deg2rad(25.0)
CATCH_VEL = 8.0


def _angles(obs):
  q1 = jp.arctan2(obs[..., 0], obs[..., 2])
  q12 = jp.arctan2(obs[..., 1], obs[..., 3])
  q2 = jp.arctan2(jp.sin(q12 - q1), jp.cos(q12 - q1))
  return q1, q2, q12, obs[..., 4], obs[..., 5]


def total_energy(obs):
  """Total mechanical energy, E=0 at upright rest, <0 below."""
  q1, q2, q12, d1, d2 = _angles(obs)
  # PE measured from upright (cos terms; at q=0 PE=0, below PE<0)
  PE = (M1 * LC1 + M2 * L1) * G * (jp.cos(q1) - 1.0) \
       + M2 * LC2 * G * (jp.cos(q12) - 1.0)
  # KE = 0.5 qd^T M qd
  cos2 = jp.cos(q2)
  M11 = A_M + 2 * B_M * cos2
  M12 = C_M + B_M * cos2
  M22 = C_M
  KE = 0.5 * (M11 * d1 * d1 + 2 * M12 * d1 * d2 + M22 * d2 * d2)
  return PE + KE


def controller(obs, ke=KE, kp2=KP2, kd2=KD2,
               kp_s=KP_S, kp_e2=KP_E2, kd_s=KD_S, kd_e2=KD_E2,
               catch_ang=CATCH_ANG, catch_vel=CATCH_VEL):
  q1, q2, q12, d1, d2 = _angles(obs)
  E = total_energy(obs)
  Et = E - E_DES                       # <0 below upright energy

  cos2 = jp.cos(q2)
  sin2 = jp.sin(q2)
  M11 = A_M + 2 * B_M * cos2
  M12 = C_M + B_M * cos2
  M21 = M12
  M22 = C_M

  # Coriolis (standard 2-link, h = C qd):
  #   h1 = -B*sin2*(2 d1 d2 + d2^2),  h2 = B*sin2*d1^2
  h1 = -B_M * sin2 * (2 * d1 * d2 + d2 * d2) + B_DAMP * d1
  h2 = B_M * sin2 * d1 * d1 + B_DAMP * d2

  # gravity (PE from upright): G_i = dU/dq_i with U = -(...)*cos terms
  Gq1 = (M1 * LC1 + M2 * L1) * G * jp.sin(q1) + M2 * LC2 * G * jp.sin(q12)
  Gq2 = M2 * LC2 * G * jp.sin(q12)

  # desired elbow acceleration: energy injection - elbow regulation
  v = -kp2 * q2 - kd2 * d2 + ke * Et * d1
  # near a dead/symmetric state (d1~0) inject a small constant to break symmetry
  v = v + 0.3 * (jp.abs(d1) < 0.02).astype(v.dtype) * jp.sign(jp.where(q1 >= 0, 1.0, -1.0))

  # collocated PFL: eliminate unactuated DOF (q1).
  Minv11 = 1.0 / M11
  Mbar = M22 - M21 * Minv11 * M12
  Hbar = (h2 + Gq2) - M21 * Minv11 * (h1 + Gq1)
  tau_pump = Mbar * v + Hbar
  u_pump = jp.clip(tau_pump / GEAR, -UMAX, UMAX)

  # balance PD on 4-state
  u_stab = -(kp_s * q1 + kp_e2 * q2 + kd_s * d1 + kd_e2 * d2)
  u_stab = jp.clip(u_stab, -UMAX, UMAX)

  near_top = (jp.abs(q1) < catch_ang) & (jp.abs(q2) < catch_ang) \
             & ((jp.abs(d1) + jp.abs(d2)) < catch_vel)
  phase = jp.where(near_top, 1, 0)
  a = jp.where(near_top, u_stab, u_pump)
  return jp.clip(a, -UMAX, UMAX), phase
