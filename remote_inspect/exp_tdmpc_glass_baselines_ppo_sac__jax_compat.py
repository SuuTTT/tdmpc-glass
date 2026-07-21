"""Restore jax.device_put_replicated removed in jax>=0.10 for brax 0.14.2.
Drop-in: stack the pytree across devices along a new leading axis, with replica i
placed on device i (what brax pmap code expects). Implemented via pmap(identity),
which shards the leading axis one-per-device and works on modern jax.
"""
import jax
import jax.numpy as jnp

if not hasattr(jax, "device_put_replicated"):
    def device_put_replicated(x, devices):
        n = len(devices)
        # broadcast each leaf to (n, ...)
        stacked = jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(jnp.asarray(leaf),
                                          (n,) + jnp.asarray(leaf).shape),
            x,
        )
        # pmap(identity) over devices[:n] places shard i on device i.
        ident = jax.pmap(lambda y: y, devices=list(devices))
        return ident(stacked)
    jax.device_put_replicated = device_put_replicated


# --- Patch brax checkpoint.load_config: it crashes on KERNEL_INITIALIZER[None]
# when a *_kernel_init_fn kwarg was saved as null. Save-side skips None; load-side
# does not. Re-implement load_config to skip None (and tolerate the dtype field).
import json as _json
from etils import epath as _epath
from ml_collections import config_dict as _cd
from brax.training import checkpoint as _ckpt
from brax.training import networks as _nets

def _patched_load_config(config_path):
    config_path = _epath.Path(config_path)
    if not config_path.exists():
        raise ValueError(f"Config file not found at {config_path.as_posix()}")
    loaded = _json.loads(config_path.read_text())
    nfk = loaded.get("network_factory_kwargs", {})
    if "activation" in nfk and nfk["activation"] is not None:
        nfk["activation"] = _nets.ACTIVATION[nfk["activation"]]
    for k in _ckpt._KERNEL_INIT_FN_KEYWORDS:
        if k not in nfk:
            continue
        v = nfk[k]
        if v is None:
            del nfk[k]  # let network factory use its own default
        else:
            nfk[k] = _nets.KERNEL_INITIALIZER[v]
    loaded["network_factory_kwargs"] = nfk
    # Fix observation_size: saved as {"shape":[N],"dtype":...} which is a Mapping
    # without the policy_obs_key -> breaks _get_obs_state_size. Flatten to int N.
    osz = loaded.get("observation_size")
    if isinstance(osz, dict) and "shape" in osz:
        shp = osz["shape"]
        loaded["observation_size"] = int(shp[-1]) if isinstance(shp, (list, tuple)) and shp else int(shp)
    return _cd.create(**loaded)

_ckpt.load_config = _patched_load_config
# also patch the per-agent modules that imported the name (they call checkpoint.load_config)
try:
    from brax.training.agents.ppo import checkpoint as _ppock
    _ppock.checkpoint.load_config = _patched_load_config
except Exception:
    pass
try:
    from brax.training.agents.sac import checkpoint as _sacck
    _sacck.checkpoint.load_config = _patched_load_config
except Exception:
    pass
