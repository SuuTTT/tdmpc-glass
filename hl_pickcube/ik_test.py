import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.2");os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
ctrl=C.make_controller(env,C.Knobs())
home=jp.array(env._init_q)[ctrl.arm_qadr]
Rh=ctrl.R_home
np.set_printoptions(precision=3,suppress=True)
# target a box at r=0.55, slightly forward
tgt=jp.array([0.55,0.0,0.035])
for ow in [0.0,0.1,0.3,0.6]:
    q=C.ik_step(home,tgt,ctrl.lowers7,ctrl.uppers7,1.0,40,R_des=Rh,ori_w=ow)
    pos=np.asarray(C.site_fk(q)); R=np.asarray(C.site_fk_R(q))
    oerr=np.asarray(C._so3_err(Rh@R.T))
    print("ow=%.1f -> pos=%s perr=%.3f oerr_norm=%.3f q_nan=%s"%(ow,pos.round(3),float(np.linalg.norm(np.asarray(tgt)-pos)),float(np.linalg.norm(oerr)),bool(np.isnan(np.asarray(q)).any())))
