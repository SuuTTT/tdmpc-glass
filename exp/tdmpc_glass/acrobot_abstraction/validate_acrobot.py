#!/usr/bin/env python3
"""Validate the Spong energy-shaping AcrobotSwingup controller ALONE on the real
mujoco_playground AcrobotSwingup env, Protocol A (n parallel, 1000-step episode,
mean return). Reports controller-alone return + an 'up_frac' (fraction of steps
with reward>0.5, i.e. tip near target). 'single' = default gains multi-seed;
'grid' = coarse gain search."""
import os, sys, json, itertools, statistics
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
os.environ.setdefault("MUJOCO_GL", "egl")
import jax, jax.numpy as jp
from mujoco_playground import registry, wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acrobot_controller as AC

EPLEN = 1000
N = int(os.environ.get("VAL_N", "256"))


def build_env():
  env = registry.load("AcrobotSwingup", config_overrides={"impl": "jax"})
  return wrapper.wrap_for_brax_training(env, episode_length=EPLEN, action_repeat=1)


def eval_controller(env, gains, seed=0):
  _step = jax.jit(env.step)

  @jax.jit
  def rollout(key):
    st = env.reset(jax.random.split(key, N))
    ep = jp.zeros(N); up = jp.zeros(N)
    def body(carry, _):
      s, ep, up = carry
      a, _phase = AC.controller(s.obs, **gains)
      a = a.reshape(N, 1)
      ns = _step(s, a)
      return (ns, ep + ns.reward, up + (ns.reward > 0.5).astype(jp.float32)), None
    (_, ep, up), _ = jax.lax.scan(body, (st, ep, up), None, length=EPLEN)
    return ep, up / EPLEN

  ep, up = rollout(jax.random.PRNGKey(seed))
  import numpy as np
  ep = np.array(ep); up = np.array(up)
  return ep, up


DEFAULT = dict(ke=AC.KE, kp2=AC.KP2, kd2=AC.KD2, kp_s=AC.KP_S,
               kp_e2=AC.KP_E2, kd_s=AC.KD_S, kd_e2=AC.KD_E2,
               catch_ang=AC.CATCH_ANG, catch_vel=AC.CATCH_VEL)


def main():
  env = build_env()
  mode = sys.argv[1] if len(sys.argv) > 1 else "single"
  if mode == "grid":
    results = []
    for ke, kp2, kd2, catch_deg in itertools.product(
        [0.3, 0.8, 1.5, 3.0], [4.0, 9.0, 16.0], [2.0, 4.0], [25.0, 35.0]):
      g = dict(DEFAULT); g["ke"] = ke; g["kp2"] = kp2; g["kd2"] = kd2
      g["catch_ang"] = float(jp.deg2rad(catch_deg))
      ep, up = eval_controller(env, g)
      results.append((float(ep.mean()), float(up.mean()), ke, kp2, kd2, catch_deg))
      print(f"ke={ke} kp2={kp2} kd2={kd2} catch={catch_deg} "
            f"ret={ep.mean():7.1f} up={up.mean():.3f}", flush=True)
    results.sort(reverse=True)
    print("\nTOP 8:")
    for r in results[:8]:
      print(f"  ret={r[0]:7.1f} up={r[1]:.3f} ke={r[2]} kp2={r[3]} kd2={r[4]} catch={r[5]}")
    b = results[0]
    json.dump({"best_ret": b[0], "best_up": b[1], "ke": b[2], "kp2": b[3],
               "kd2": b[4], "catch_deg": b[5]}, open("controller_best.json", "w"), indent=2)
  else:
    rets, ups, mins = [], [], []
    n_fail = 0; total = 0
    for sd in range(int(os.environ.get("VAL_SEEDS", "3"))):
      ep, up = eval_controller(env, DEFAULT, seed=sd)
      rets.append(float(ep.mean())); ups.append(float(up.mean()))
      mins.append(float(ep.min()))
      # swing-up "failure": episode return < 100 (never gets near upright)
      n_fail += int((ep < 100).sum()); total += len(ep)
      print(f"seed={sd} ret_mean={ep.mean():7.1f} ret_min={ep.min():7.1f} "
            f"up_frac={up.mean():.3f} n_fail(<100)={(ep<100).sum()}/{len(ep)}", flush=True)
    print(f"\ncontroller-alone (default): ret_mean={statistics.mean(rets):.1f} "
          f"std={statistics.pstdev(rets):.1f}  up_frac={statistics.mean(ups):.3f} "
          f"min={min(mins):.0f}  swingup_fail={n_fail}/{total}")
    json.dump({"rets": rets, "ups": ups, "mins": mins, "mean": statistics.mean(rets),
               "up_frac": statistics.mean(ups), "swingup_fail": n_fail, "total": total,
               "n_per_seed": N}, open("controller_default.json", "w"), indent=2)


if __name__ == "__main__":
  main()
