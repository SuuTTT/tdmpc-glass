"""Occupancy probe that loads the checkpoint's OWN config from flags.json (fixes shape
mismatch for fourier/dim-changing variants). Usage: probe2.py <ghm_dir> <epoch> [env] [agent_mod]"""
import importlib, json, os, sys
import numpy as np
sys.path.insert(0, "/root/ghm/infom")
import planning.occupancy_probe as P

ghm_dir = sys.argv[1]
epoch = int(sys.argv[2])
env = sys.argv[3] if len(sys.argv) > 3 else P.ANTMAZE_ENV
amod = sys.argv[4] if len(sys.argv) > 4 else "ghm"

def _load(gd, ge, obs_dim, action_dim):
    P._clear_pkgs("utils", "agents")
    sys.path.insert(0, P.INFOM_DIR)
    import jax.numpy as jnp
    mod = importlib.import_module("agents." + amod)
    from utils.flax_utils import restore_agent
    config = mod.get_config()
    fj = os.path.join(gd, "flags.json")
    if os.path.exists(fj):
        saved = json.load(open(fj)).get("agent", {})
        for k, v in saved.items():
            try:
                if k in config:
                    config[k] = v
            except Exception:
                pass
    ex_obs = np.zeros((4, obs_dim), np.float32)
    ex_act = np.zeros((4, action_dim), np.float32)
    agent = mod.GHMAgent.create(0, jnp.asarray(ex_obs), jnp.asarray(ex_act), config)
    agent = restore_agent(agent, gd, ge)
    print("[probe2] " + amod + " restored from " + gd + " (config merged from flags.json)")
    return agent

P.load_ghm_agent = _load
r = P.probe_ghm("run", env, ghm_dir, epoch)
out = os.path.join(ghm_dir, "probe2_result.json")
json.dump(r, open(out, "w"), indent=2, default=float)
print("PROBE2_JSON " + out)
print(json.dumps(r, indent=2, default=float))
