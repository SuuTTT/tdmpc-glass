#!/usr/bin/env bash
# bootstrap_worker_env.sh — set up a fresh vast.ai box to run TD-MPC-Glass.
#
# Replicates the working worker env (CUDA-12 Ampere): /root/venv with jax 0.10.1
# cuda12 + mujoco 3.8.0 + mjx + warp + brax/flax/optax, and clones
# mujoco_playground @ the pinned commit to /root/mujoco_playground_repo.
# The queue daemon rsyncs scripts/+src/ to /root/helios-rl at launch; the
# launcher imports helios + mujoco_playground via PYTHONPATH (no pip-install of
# either needed beyond registering mujoco_playground).
#
# Idempotent-ish: skips venv creation if /root/venv already imports jax+mjx_pg.
# Logs to /root/tdmpc_env_setup.log. Run detached: nohup bash this.sh &
set -u
LOG=/root/tdmpc_env_setup.log
exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap start $(date -u +%FT%TZ) on $(hostname) ==="

MJX_COMMIT=33f1b2843a7ec5537c4882177aa2a9f236e9b692

if /root/venv/bin/python -c "import jax,mujoco_playground" 2>/dev/null; then
  echo "env already present; skipping install"
else
  python3 -m venv /root/venv || { echo "venv create FAILED"; exit 1; }
  /root/venv/bin/pip install -U pip wheel setuptools || exit 1
  # Pinned to match the known-good worker (ssh2_a4000) freeze.
  /root/venv/bin/pip install \
      "jax[cuda12]==0.10.1" jaxlib==0.10.1 \
      mujoco==3.8.0 mujoco-mjx==3.8.0 warp-lang==1.12.1 \
      brax==0.14.2 flax==0.12.7 optax==0.2.8 jaxopt==0.8.5 || { echo "pip core FAILED"; exit 1; }
fi

if [ ! -d /root/mujoco_playground_repo ]; then
  git clone https://github.com/google-deepmind/mujoco_playground.git /root/mujoco_playground_repo || { echo "clone FAILED"; exit 1; }
fi
git -C /root/mujoco_playground_repo checkout -q "$MJX_COMMIT" || echo "WARN: checkout $MJX_COMMIT failed (using default branch)"
# Register without disturbing the pinned deps we just installed.
/root/venv/bin/pip install -e /root/mujoco_playground_repo --no-deps || echo "WARN: editable install failed (PYTHONPATH fallback still works)"

mkdir -p /root/helios-rl/tmp
echo "=== verify ==="
/root/venv/bin/python - <<'PY'
import jax, mujoco, warp
import mujoco_playground  # noqa
print("jax", jax.__version__, "devices", jax.devices())
print("mujoco", mujoco.__version__, "warp", warp.__version__, "mjx_playground OK")
PY
echo "=== bootstrap done $(date -u +%FT%TZ) ==="
