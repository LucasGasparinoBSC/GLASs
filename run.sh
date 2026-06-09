#!/bin/bash

#SBATCH --job-name=HIP_Test
#SBATCH --output=examplejob.o%j
#SBATCH --error=examplejob.e%j
#SBATCH --account=project_465002709
#SBATCH --time 00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1

module purge
module load PrgEnv-cray craype-accel-amd-gfx90a craype-x86-trento rocm perftools-base perftools-lite-events

## If select_gpu file exists, remove it to avoid confusion. We will create a new one with the correct content.
if [ -f select_gpu ]; then
    echo "select_gpu file already exists. Removing it to avoid confusion."
    rm -rf select_gpu
fi
export MPICH_GPU_SUPPORT_ENABLED=1
#printenv | grep MPICH

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
#srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/DeviceUtils/unitt_DeviceUtils_32
#srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/DeviceMemory/unitt_DeviceMemory_32
#srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/DeviceMemory/unitt_DeviceMemory_64
#srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/Kernels_Lv1/unitt_Kernels_Lv1_32
#srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/Kernels_Lv1/unitt_Kernels_Lv1_64
#srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/IterSolvers/unitt_IterSolvers_32
#srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/IterSolvers/unitt_IterSolvers_64
srun --cpu-bind=${CPU_BIND} ./select_gpu ./build/Tests/ConjugateGradient/unitt_ConjugateGradient_32
