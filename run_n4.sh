#!/bin/bash

#SBATCH --job-name=HIP_Test
#SBATCH --output=examplejob.o%j
#SBATCH --error=examplejob.e%j
#SBATCH --account=project_465002709
#SBATCH --time 00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=4

module purge
module load PrgEnv-cray craype-accel-amd-gfx90a craype-x86-trento rocm perftools-base perftools-lite-events
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
srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/Comm_Utils/unitt_Comm_Utils_32