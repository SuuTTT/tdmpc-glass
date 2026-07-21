#!/usr/bin/env bash
# One-shot worker env setup for the D2 suite. Assumes /root/helios-rl/{src,scripts} and
# /root/mujoco_playground_repo already rsynced from EC2. Idempotent.
set -e
cd /root/helios-rl
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install "jax[cuda12]" flax optax hydra-core omegaconf numpy huggingface_hub
cd /root/mujoco_playground_repo && python -m pip install -e .
# The mujoco 3.9 warp backend is broken on released warp-lang; we run MJPG_IMPL=jax on the
# coherent 3.8.0 stack (proven working on g4070).
python -m pip install "mujoco==3.8.0" "mujoco-mjx==3.8.0"
cd /root/helios-rl
python -c "import jax,mujoco; from mujoco_playground import registry; print('WORKER_OK jax',jax.__version__,'devs',len(jax.devices()),'mujoco',mujoco.__version__)"
# Pre-fetch mujoco_menagerie assets SERIALLY (first Panda load downloads them; concurrent jobs
# would race the download and crash with "Error opening file 'mjx_panda.xml'").
export PYTHONPATH=/root/helios-rl/src:/root/mujoco_playground_repo MUJOCO_GL=egl
python -c "from mujoco_playground import registry; registry.load('PandaPickCube', config_overrides={'impl':'jax'}); print('MENAGERIE_PREFETCH_OK')"
echo WORKER_SETUP_DONE
