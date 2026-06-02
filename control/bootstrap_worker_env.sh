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
  # jax 0.10.1 needs python>=3.11. vastai/pytorch ships 3.10 -> install 3.11 via deadsnakes.
  PY=""
  for c in python3.12 python3.11; do command -v "$c" >/dev/null 2>&1 && { PY=$c; break; }; done
  if [ -z "$PY" ]; then
    echo "no python>=3.11 found; installing python3.11 (deadsnakes)..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq software-properties-common >/dev/null 2>&1
    add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
    apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv python3.11-dev >/dev/null 2>&1
    command -v python3.11 >/dev/null 2>&1 && PY=python3.11
  fi
  [ -z "$PY" ] && { echo "FAILED to obtain python>=3.11"; exit 1; }
  echo "using $PY ($($PY --version 2>&1)) for venv"
  rm -rf /root/venv   # drop any wrong-python venv from a prior attempt
  "$PY" -m venv /root/venv || { echo "venv create FAILED"; exit 1; }
  /root/venv/bin/pip install -U pip wheel setuptools || exit 1
  # Pinned to match the known-good worker (ssh2_a4000) freeze.
  /root/venv/bin/pip install \
      "jax[cuda12]==0.10.1" jaxlib==0.10.1 \
      mujoco==3.8.0 mujoco-mjx==3.8.0 warp-lang==1.12.1 \
      brax==0.14.2 flax==0.12.7 optax==0.2.8 jaxopt==0.8.5 \
      lxml etils ml_collections tqdm mediapy || { echo "pip core FAILED"; exit 1; }
  # NOTE: do NOT add dm_control — it forces mujoco>=3.8.1 and drifts off the pinned 3.8.0.
fi

# EGL/GL system libs — MUJOCO_GL=egl needs these or run_benchmark dies at eglQueryString.
# (vastai/pytorch base lacks them.) Harmless if already present.
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq libegl1 libgl1 libgles2 libglvnd0 libglew2.2 libosmesa6 libgl1-mesa-glx >/dev/null 2>&1 || true

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
