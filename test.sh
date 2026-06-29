#!/bin/bash
#SBATCH --job-name=mpi_job
#SBATCH --output=mpi_%j.out
#SBATCH --error=mpi_%j.err
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --account=bsc21
#SBATCH --qos=ngp_bsccase

module purge
module load nvidia-hpc-sdk/26.1
module list
mpirun -np 4 /gpfs/scratch/bsc21/bsc21387/Repos/GLASs/build_mpi/Tests/Comm_Utils/unitt_Comm_Utils_32
