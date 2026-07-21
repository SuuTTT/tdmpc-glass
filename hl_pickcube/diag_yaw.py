import os
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false");os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.3")
import numpy as np, jax, jax.numpy as jp
from mujoco_playground import registry
import controller as C
env=registry.load("PandaPickCube",config_overrides={"impl":"jax"})
k=C.Knobs(); k.ori_w=0.0  # position-only
ctrl=C.make_controller(env,k)
reset=jax.jit(env.reset); step=jax.jit(env.step); act=jax.jit(ctrl.act)
np.set_printoptions(precision=2,suppress=True)
for sd in [0,6,11,15,3,8]:
    s=reset(jax.random.PRNGKey(sd)); s=step(s,jp.zeros(8)); cs=ctrl.init_state()
    box0xy=np.asarray(s.data.xpos[env._obj_body])[:2]
    for t in range(40):  # through grasp+early lift
        a,cs=act(s,cs); s=step(s,a)
    bm=np.asarray(s.data.xmat[env._obj_body]).reshape(3,3)
    sm=np.asarray(s.data.site_xmat[env._gripper_site]).reshape(3,3)
    # yaw of box (atan2 of first column)
    box_yaw=np.degrees(np.arctan2(bm[1,0],bm[0,0]))
    print("sd=%2d box0xy=%s box_yaw_deg=%.1f box_x_axis=%s"%(sd,box0xy.round(2),box_yaw,bm[:,0].round(2)))
