#!/bin/bash

#SBATCH --job-name=HIP_Test
#SBATCH --output=cg_n32_%j.o
#SBATCH --error=cg_n32_%j.e
#SBATCH --account=project_465002709
#SBATCH --time 00:20:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8

module purge
module load PrgEnv-cray craype-accel-amd-gfx90a craype-x86-trento rocm perftools-base perftools-lite-gpu
rm -rf select_gpu
export MPICH_GPU_SUPPORT_ENABLED=1
#printenv | grep MPICH

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/ConjugateGradient/unitt_ConjugateGradient_32
