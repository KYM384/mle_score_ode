#!/bin/bash

#PBS -q debug-g
#PBS -l select=1
#PBS -l walltime=10:00
#PBS -W group_list=gj26
#PBS -j oe

cd ${PBS_O_WORKDIR}

module load python/3.10.16
module load singularity/4.2.1

python -m venv tmp_env
source tmp_env/bin/activate
pip install spython

spython recipe .docker/Dockerfile .docker/mle.def

deactivate
rm -rf .tmp_env
rm .docker/mle.sif

singularity build --fakeroot .docker/mle.sif .docker/mle.def

singularity exec --nv .docker/mle.sif python -c "import torch; print(torch.cuda.is_available())"

