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
# Start AT home, target = home pos (should stay, oerr=0)
p_home=np.asarray(C.site_fk(home))
print("home pos",p_home)
# perturb start q a bit, target home pos + level orientation
q0=home+0.2
for it in [40,100,200]:
  for ow in [0.5,1.0,2.0]:
    q=C.ik_step(q0,jp.asarray(p_home),ctrl.lowers7,ctrl.uppers7,0.5,it,R_des=Rh,ori_w=ow)
    p=np.asarray(C.site_fk(q)); R=np.asarray(C.site_fk_R(q))
    oerr=float(np.linalg.norm(np.asarray(C._so3_err(Rh@R.T))))
    perr=float(np.linalg.norm(p_home-p))
    print("it=%d ow=%.1f perr=%.3f oerr=%.3f"%(it,ow,perr,oerr))
