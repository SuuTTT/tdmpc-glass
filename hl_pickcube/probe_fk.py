import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false"); os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.2")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics as pk
env = registry.load("PandaPickCube", config_overrides={"impl":"jax"})
st = jax.jit(env.reset)(jax.random.PRNGKey(0))
st = jax.jit(env.step)(st, jp.zeros(8))
d = st.data
gid = env._gripper_site; arm = np.array(env._robot_arm_qposadr)
q_arm = np.array(d.qpos)[arm]; site = np.array(d.site_xpos[gid])
T = np.array(pk.compute_franka_fk(jp.array(q_arm))); ee = T[:3,3]
print("q_arm", np.round(q_arm,3))
print("site_xpos", np.round(site,4))
print("FK ee", np.round(ee,4), "z-axis", np.round(T[:3,2],3))
for off in [0.1104,0.107,0.1,0.0]:
    p = ee - off*T[:3,2]
    print(f"  off={off}: pred={np.round(p,4)} err={np.linalg.norm(p-site):.5f}")
print("box", np.round(np.array(d.xpos[env._obj_body]),4))
print("target", np.round(np.array(st.info["target_pos"]),4))
print("base offset? site-ee", np.round(site-ee,4))
