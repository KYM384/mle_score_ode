#!/bin/bash

#PBS -q regular-g
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=20:00:00
#PBS -W group_list=gj26
#PBS -j oe

cd ${PBS_O_WORKDIR}

module purge
module load singularity/4.2.1
module load python/3.10.16
module load cuda/12.4
module load nvidia/24.9
module load nv-hpcx/24.9


export NUM_NODES=`wc -l $PBS_NODEFILE | awk '{print $1}'`
export MASTER_ADDR=`head -1 $PBS_NODEFILE`
export MASTER_PORT=2345
unset OMPI_MCA_mca_base_env_list


rm -rf .tmp
mkdir .tmp
for ((i=0; i<NUM_NODES; i++)); do
    mkdir ".tmp/$i"
done


mpirun -np $NUM_NODES --hostfile $PBS_NODEFILE -bind-to none -map-by node \
    -x MASTER_ADDR \
    -x MASTER_PORT \
    -x PATH \
    -x LD_LIBRARY_PATH \
    singularity exec --bind ${PWD}:/workspace --bind ${PWD}/../imagenet32:/data --nv .docker/mle.sif \
        python3 main.py \
        --config configs/ve/imagenet32_ncsnpp_continuous.py \
        --workdir ./results/ve_imagenet32_ncsnpp_continuous2/ \
        --mode train \
        --loss new \
        --name 'VE imagenet high2'

