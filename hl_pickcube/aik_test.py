import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.2");os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics as pk
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
ctrl=C.make_controller(env,C.Knobs())
home=np.array(env._init_q)[np.asarray(env._robot_arm_qposadr)]
Rh=np.asarray(ctrl.R_home)
np.set_printoptions(precision=3,suppress=True)
# Build a desired EE transform: level orientation Rh, position = box at r=0.5
def make_T(pos,R): 
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=pos; return jp.asarray(T)
for pos in [[0.5,0.0,0.10],[0.5,0.0,0.40],[0.6,-0.2,0.30],[0.45,0.1,0.035]]:
    T=make_T(pos,Rh)
    # sweep q7
    best=None
    for q7 in np.linspace(-2.8,2.8,15):
        q=np.asarray(pk.compute_franka_ik(T,float(q7),jp.asarray(home)))
        if np.isnan(q).any(): continue
        p=np.asarray(C.site_fk(jp.asarray(q))); R=np.asarray(C.site_fk_R(jp.asarray(q)))
        perr=np.linalg.norm(np.array(pos)-p); oerr=np.linalg.norm(np.asarray(C._so3_err(jp.asarray(Rh@R.T))))
        within=(q>=np.asarray(ctrl.lowers7)-1e-3).all() and (q<=np.asarray(ctrl.uppers7)+1e-3).all()
        if perr<0.01 and oerr<0.1 and within:
            best=(q7,perr,oerr); break
    print("pos",pos,"-> solved:",best is not None, best)
