import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.2");os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics as pk
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
home=np.array(env._init_q)[np.asarray(env._robot_arm_qposadr)]
T=np.asarray(pk.compute_franka_fk(jp.asarray(home)))
np.set_printoptions(precision=3,suppress=True)
print("home FK rotation (gripper frame in world):\n",T[:3,:3])
print("home FK pos:",T[:3,3].round(3))
# Also check the actual SITE xmat at reset (the real gripper site orientation)
import jax
s=jax.jit(env.reset)(jax.random.PRNGKey(0)); s=jax.jit(env.step)(s,jp.zeros(8))
print("real site xmat at home:\n",np.asarray(s.data.site_xmat[env._gripper_site]).reshape(3,3).round(3))
print("box xmat at home:\n",np.asarray(s.data.xmat[env._obj_body]).reshape(3,3).round(3))
