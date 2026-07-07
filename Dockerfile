# syntax=docker/dockerfile:1
# =============================================================================
# mle_score_ode (ICML 2022) reproducible JAX/Flax/TensorFlow GPU environment.
#
# Target: single Wisteria/Aquarius node = 8x A100-SXM4-40GB (sm_80), driven at
# runtime via `singularity exec --nv` with the repo bind-mounted at /workspace.
# The image therefore carries ONLY the CUDA runtime/toolkit + Python stack; the
# host NVIDIA kernel driver is injected by --nv (CUDA is forward-compatible, so
# a CUDA 11.1 image runs fine under the node's newer R535/cuda-12.2 driver).
#
# Version matrix reproduces yang-song/score_flow's authoritative pip-freeze
# (TF 2.5.0 + jax 0.2.18 / jaxlib 0.1.69+cuda111 + flax 0.3.3, numpy 1.19.5).
#
# Build for x86_64:  docker build --platform linux/amd64 -t mle_score_ode:cuda111 .
# Source is NOT copied in (bind-mounted at run time per the Singularity workflow).
# =============================================================================

# CUDA 11.1.1 + cuDNN 8 devel: ships ptxas 11.1 in /usr/local/cuda/bin, which
# jaxlib 0.1.69+cuda111 needs to JIT sm_80 kernels for the A100.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# ---- Environment ------------------------------------------------------------
# Locale + Python I/O behaviour. PYTHONNOUSERSITE=1 makes the container ignore
# the host's ~/.local/lib/python3.8/site-packages: Wisteria shares $HOME between
# Odyssey (A64FX/aarch64) and Aquarius (x86_64), and Singularity auto-mounts
# $HOME, so any host `pip install --user` packages would otherwise leak in and
# shadow (or ABI-clash with) the image's own packages.
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1
# JAX allocates GPU memory on demand (40GB cards); quiet TF's C++ INFO/WARNING
# logs (0=all,1=noINFO,2=noWARN,3=noERROR); put the container's own CUDA toolkit
# first so the host ptxas cannot shadow ptxas 11.1. NOTE: comments are kept OUT
# of the ENV line-continuations so `spython recipe` converts this cleanly.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_CPP_MIN_LOG_LEVEL=1 \
    PATH=/usr/local/cuda/bin:${PATH}

# ---- System / apt dependencies ---------------------------------------------
# python3.8 = Ubuntu 20.04 default (matches jaxlib cp38 + TF 2.5 cp38 wheels).
# libgl1/glib/etc are runtime libs matplotlib+Pillow load; *-dev headers let a
# few 2021-era sdist deps compile if no wheel is found.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3.8-distutils python3-pip \
        build-essential pkg-config git wget ca-certificates \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
        zlib1g-dev libjpeg-dev libpng-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- Python interpreter + modern pip ---------------------------------------
# Ubuntu 20.04's python3 is already 3.8; pin build backend <60 (setuptools 59.8)
# to dodge the distutils-precedence breakage that bites these 2021-era wheels.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && python3.8 -m pip install --no-cache-dir --upgrade \
        'pip==21.3.1' 'setuptools==59.8.0' 'wheel==0.37.1'

# ---- Step 1: low-level pins FIRST so nothing below can move them -------------
# Two hard ceilings: numpy<1.20 (TF 2.5 / jaxlib ABI) and protobuf<3.20 (TF
# descriptor cap). absl-py==0.12.0 (repo requirement) is fixed here so every
# later install binds against it. contextlib2 is a hard, easy-to-miss dep of
# ml-collections 0.1.0 (imported the moment any configs/*.py builds a ConfigDict).
RUN python3.8 -m pip install --no-cache-dir \
        'numpy==1.19.5' 'scipy==1.6.3' 'six==1.15.0' 'absl-py==0.12.0' \
        'protobuf==3.19.6' 'typing-extensions==3.7.4.3' 'gast==0.4.0' \
        'grpcio==1.34.1' 'h5py==3.1.0' 'flatbuffers==1.12' 'opt-einsum==3.3.0' \
        'PyYAML==5.4.1' 'msgpack==1.0.2' 'dm-tree==0.1.6' 'toolz==0.11.1' \
        'contextlib2==0.6.0.post1'

# ---- Step 2: jaxlib GPU wheel + jax -----------------------------------------
# The '+cuda111' local-version tag exists ONLY on the jax_releases find-links
# index (PyPI has plain CPU jaxlib), so the GPU wheel can never be clobbered by
# a CPU one. --no-deps keeps the Step-1 pins (jax's deps already satisfied).
# NCCL for 8-GPU pmap is bundled inside this wheel; no host libnccl is needed.
RUN python3.8 -m pip install --no-cache-dir --no-deps 'jaxlib==0.1.69+cuda111' \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && python3.8 -m pip install --no-cache-dir --no-deps 'jax==0.2.18'

# ---- Step 3: TF 2.5 companion runtime deps (not already pinned) --------------
RUN python3.8 -m pip install --no-cache-dir \
        'keras-nightly==2.5.0.dev2021032900' 'Keras-Preprocessing==1.1.2' \
        'google-pasta==0.2.0' 'astunparse==1.6.3' 'termcolor==1.1.0' \
        'wrapt==1.12.1' 'tensorflow-estimator==2.5.0' 'tensorboard==2.5.0'

# ---- Step 4: TensorFlow itself via --no-deps --------------------------------
# --no-deps freezes the exact companion set (all of TF 2.5's deps are already
# installed in Steps 1/3) so the resolver cannot nudge the load-bearing pins
# (numpy/protobuf/grpcio/h5py/gast/typing-extensions). TF is used only for the
# tf.data pipeline + FID/Inception eval; its GPU is hidden in main.py.
RUN python3.8 -m pip install --no-cache-dir --no-deps 'tensorflow==2.5.0'

# ---- Step 5: TF ecosystem ---------------------------------------------------
# tensorflow-io via --no-deps so it cannot try to reinstall/move TF (it hard-
# requires tensorflow>=2.5,<2.6 -- proof this stack is TF 2.5, not 2.4).
# tensorflow-probability==0.12.2 is the version in score_flow's proven freeze
# (pulled in by tensorflow-gan for FID); Step 9 imports tf_gan to verify it.
RUN python3.8 -m pip install --no-cache-dir --no-deps \
        'tensorflow-io-gcs-filesystem==0.18.0' 'tensorflow-io==0.18.0' \
    && python3.8 -m pip install --no-cache-dir \
        'tensorflow-metadata==0.30.0' 'tensorflow-hub==0.12.0' \
        'tensorflow-probability==0.12.2' 'tensorflow-addons==0.13.0' \
        'tensorflow-datasets==4.3.0' 'tensorflow-gan==2.0.0'

# ---- Step 6: Flax + JAX ecosystem -------------------------------------------
# flax/chex/optax via --no-deps so they cannot pull a newer matplotlib/jax.
# ml-collections is installed WITHOUT --no-deps so its contextlib2 dep resolves
# (already pinned in Step 1; nothing else moves).
RUN python3.8 -m pip install --no-cache-dir --no-deps \
        'flax==0.3.3' 'chex==0.0.7' 'optax==0.0.9' \
    && python3.8 -m pip install --no-cache-dir 'ml-collections==0.1.0'

# ---- Step 7: leaf utilities (safe to resolve against the frozen ceilings) ----
RUN python3.8 -m pip install --no-cache-dir \
        'Pillow==8.2.0' 'matplotlib==3.4.2' 'tqdm==4.60.0' 'wandb==0.10.30'

# ---- Step 8: belt-and-suspenders re-assert of the load-bearing pins ---------
# Re-pin the four values any leaf resolver might have nudged, and re-force the
# GPU jaxlib so a CPU wheel can never survive.
RUN python3.8 -m pip install --no-cache-dir --no-deps --force-reinstall \
        'numpy==1.19.5' 'absl-py==0.12.0' 'protobuf==3.19.6' \
        'jax==0.2.18' 'jaxlib==0.1.69+cuda111' \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ---- Step 9: build-time sanity check ----------------------------------------
# No GPU during build, so assert versions/local-tag only (not device init).
# Importing tensorflow_gan pulls tensorflow_probability, so this also proves the
# tfp 0.12.2 <-> TF 2.5 pairing imports cleanly; ml_collections import proves the
# contextlib2 dep is present -> the build fails here rather than at job startup.
RUN python3.8 -c "import jaxlib, jax, tensorflow as tf, flax, numpy, scipy, ml_collections, tensorflow_datasets, tensorflow_gan, tensorflow_hub; import pkg_resources as p; assert p.get_distribution('jaxlib').version=='0.1.69+cuda111', p.get_distribution('jaxlib').version; assert jax.__version__=='0.2.18'; assert tf.__version__=='2.5.0'; assert numpy.__version__=='1.19.5'; print('OK jaxlib', jaxlib.__version__, '| tf', tf.__version__, '| flax', flax.__version__)"

# singularity exec drives the real command; bash is only a convenience default.
CMD ["bash"]
