#!/bin/bash

#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=2:00:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j


module load gcc/8.3.1
module load cuda/12.2          # provides the host driver plumbing for --nv
module load singularity/3.9.5


# --- Weights & Biases (optional) --------------------------------------------
# NOTE: wandb calls are currently COMMENTED OUT in run_lib.py (lines 35,71,186,215).
# The key below is a no-op until you uncomment them. To enable logging:
#   1) uncomment `import wandb` and the wandb.init/log lines in run_lib.py, and
#   2) `export WANDB_API_KEY=xxxxx` in the SUBMITTING shell before `pjsub`
#      (never hardcode it here -- it would be committed to git history).
# singularity forwards host env vars by default, so the container inherits it.

singularity exec --nv \
    --bind ${PWD}:/workspace \
    .docker/mle_score_ode.sif bash -c \
    "cd /workspace && \
     export WANDB_API_KEY=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 && \
     export PYTHONNOUSERSITE=1 && \
     export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
     python main.py \
         --config configs/ve/cifar10_ncsnpp_continuous.py \
         --mode train \
         --workdir experiments/cifar10_ve \
         --config.training.score_matching_order=2 \
         --config.training.batch_size=128"
