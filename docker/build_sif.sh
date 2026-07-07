#!/bin/bash
# =============================================================================
# build_sif.sh — build the Docker image and convert it to a Singularity .sif
# WITHOUT a Docker daemon on the HPC login node.
#
# Wisteria has no dockerd, but `singularity build` can bootstrap straight from a
# `docker save` tar archive (docker-archive://) with no daemon and no registry.
#
# Two-machine flow:
#   1. On a workstation WITH Docker (linux/amd64): build -> docker save -> scp.
#   2. On Wisteria, inside a *prepost* job (NEVER a login node): singularity
#      build ... docker-archive://<tar>. The resulting .sif lands at
#      .docker/mle_score_ode.sif to mirror the reference script's .docker/ layout.
#
# Usage:
#   ./docker/build_sif.sh docker     # step 1: on your Docker-capable machine
#   ./docker/build_sif.sh sif        # step 2: on Wisteria in a prepost job
#   ./docker/build_sif.sh            # runs whichever step fits the host
# =============================================================================
set -euo pipefail

# ---- Generic, override-able paths -------------------------------------------
IMAGE_NAME="${IMAGE_NAME:-mle_score_ode:cuda111}"      # docker tag
TAR_PATH="${TAR_PATH:-mle_score_ode.tar}"             # docker save output
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_DIR="${SIF_DIR:-${REPO_ROOT}/.docker}"
SIF_PATH="${SIF_PATH:-${SIF_DIR}/mle_score_ode.sif}"

# ---- Step 1: build the image + export a portable tar (Docker-capable host) ---
build_docker() {
  echo ">> docker build ${IMAGE_NAME} (linux/amd64 to match Wisteria x86_64)"
  # --platform linux/amd64 is REQUIRED if you build on Apple Silicon / arm64.
  docker build --platform linux/amd64 -t "${IMAGE_NAME}" "${REPO_ROOT}"
  echo ">> docker save -> ${TAR_PATH}"
  docker save "${IMAGE_NAME}" -o "${TAR_PATH}"
  echo ">> Done. Transfer the tar to Wisteria, e.g.:"
  echo "     scp ${TAR_PATH} <user>@wisteria.cc.u-tokyo.ac.jp:${REPO_ROOT}/"
  echo "   then run:  ./docker/build_sif.sh sif   (inside a prepost job)"
}

# ---- Step 2: convert the tar to a .sif (Wisteria, prepost group only) --------
build_sif() {
  if ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: 'singularity' not on PATH. Load it first: module load singularity/3.9.5" >&2
    exit 1
  fi
  if [[ ! -f "${TAR_PATH}" ]]; then
    echo "ERROR: ${TAR_PATH} not found. Run step 1 (docker) and scp it here first." >&2
    exit 1
  fi
  echo ">> Building ${SIF_PATH} from docker-archive://${TAR_PATH}"
  echo "   (must run in a prepost job, e.g.:"
  echo "      pjsub --interact -g gb20 -L rscgrp=prepost -L jobenv=singularity"
  echo "      module load singularity/3.9.5 )"
  mkdir -p "${SIF_DIR}"
  # docker-archive:// just unpacks saved layers -> no --fakeroot, no daemon.
  singularity build "${SIF_PATH}" "docker-archive://${TAR_PATH}"
  echo ">> Done: ${SIF_PATH}"
}

# =============================================================================
# ALTERNATIVE — build the .sif directly from a registry with --fakeroot, no tar:
#   # on your machine (push the image somewhere you can pull):
#   docker push <registry>/<you>/mle_score_ode:cuda111
#   # on Wisteria, in a prepost job:
#   module load singularity/3.9.5
#   singularity build --fakeroot .docker/mle_score_ode.sif \
#       docker://<registry>/<you>/mle_score_ode:cuda111
#   # private registry: export SINGULARITY_DOCKER_USERNAME / _PASSWORD first.
# --fakeroot is permitted on Wisteria (subuid/subgid configured) but ONLY in the
# prepost group; it is killed on a login node or a normal compute group.
# =============================================================================

case "${1:-auto}" in
  docker) build_docker ;;
  sif)    build_sif ;;
  auto)
    if command -v docker >/dev/null 2>&1; then build_docker
    elif command -v singularity >/dev/null 2>&1; then build_sif
    else
      echo "ERROR: neither docker nor singularity on PATH." >&2
      echo "  - On your workstation: install Docker, then './docker/build_sif.sh docker'." >&2
      echo "  - On Wisteria: 'module load singularity/3.9.5' then './docker/build_sif.sh sif'." >&2
      exit 1
    fi ;;
  *) echo "Usage: $0 [docker|sif]" >&2; exit 1 ;;
esac
