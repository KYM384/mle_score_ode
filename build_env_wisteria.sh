#!/bin/bash
# =============================================================================
# build_env_wisteria.sh — PJM job that CONSTRUCTS the Singularity runtime image
# for mle_score_ode entirely on Wisteria (no external Docker daemon required).
#
# Single source of truth = the repo-root `Dockerfile`. This job:
#   1. converts  Dockerfile  ->  .docker/mle_score_ode.def   (via spython)
#   2. builds    .docker/mle_score_ode.def  ->  .docker/mle_score_ode.sif
#      with `singularity build --fakeroot`
# The resulting .sif is exactly what train_wisteria.sh / eval_wisteria.sh run.
#
# Submit with:   pjsub build_env_wisteria.sh
#
# GROUP: uses `prepost`, the group where Wisteria permits `--fakeroot` image
# builds (login/compute groups do not) and which gives ample wall-time without
# tying up A100s. The reference used debug-a; it is kept commented as a fallback.
# Confirm the group/fakeroot policy for your project before the first submit
# (site docs / `pjstat --rsc`).
#
# NETWORK: a from-scratch build pulls the CUDA base image from Docker Hub and
# pip-installs from PyPI + storage.googleapis.com inside %post, so the build node
# needs outbound HTTPS. On Wisteria that is via the site proxy — uncomment and
# set HTTP_PROXY/HTTPS_PROXY below if your build node has no direct internet.
# =============================================================================

#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j

set -e
set -o pipefail

module load singularity/3.9.5
# gcc/cuda modules are NOT needed to build the image: the container carries its
# own CUDA toolkit and nothing is compiled on the host.

# --- Outbound proxy for the build node (uncomment + set to your site proxy) ---
# export HTTP_PROXY=http://<proxy-host>:<port>
# export HTTPS_PROXY=${HTTP_PROXY}
# export http_proxy=${HTTP_PROXY} ; export https_proxy=${HTTPS_PROXY}

# Run in the directory the job was submitted from (i.e. the repo root).
cd "${PJM_O_WORKDIR:-${PWD}}"
mkdir -p .docker

DOCKERFILE="Dockerfile"                 # single source of truth (repo root)
DEF=".docker/mle_score_ode.def"         # generated Singularity definition
SIF=".docker/mle_score_ode.sif"         # built image (consumed by train/eval)

# --- 1. Dockerfile -> Singularity def via spython (isolated throwaway venv) ---
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install --no-cache-dir spython
spython recipe "${DOCKERFILE}" "${DEF}"
deactivate
rm -rf .venv

# Make the generated %post fail-fast + traced, so any pip/apt error aborts the
# build (spython does not add this itself). Inserted right after the %post line.
sed -i '/^%post/a set -ex' "${DEF}"

# --- 2. def -> .sif with --fakeroot (allowed in the prepost group) ------------
rm -f "${SIF}"
singularity build --fakeroot "${SIF}" "${DEF}"

# --- 3. Smoke test: confirm the built image imports the core stack ------------
# No --nv / no GPU on the build node: this only imports + prints versions (JAX
# warns "No GPU/TPU found" on CPU, which is expected here). Non-fatal so a
# printing hiccup cannot fail an otherwise-good build.
echo ">> Built ${SIF}"
# PYTHONNOUSERSITE=1 stops the host's ~/.local (aarch64 packages from the Odyssey
# side of the shared $HOME) from leaking in and shadowing the image's numpy.
SINGULARITYENV_PYTHONNOUSERSITE=1 singularity exec "${SIF}" python3.8 -c \
    "import jax, tensorflow as tf, flax, numpy; print('image OK: jax', jax.__version__, '| tf', tf.__version__, '| flax', flax.__version__, '| numpy', numpy.__version__)" \
    || echo ">> WARN: smoke test failed — image built but import check errored; inspect the log above."
