#!/bin/sh
#SBATCH --time=36:00:00
#SBATCH --account=kdantu
#SBATCH --cluster=faculty
#SBATCH --partition=hal
#SBATCH --qos=hal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name="MRTA_Flood_PO_stevepau_fa"
#SBATCH --output=output_MRTA_Flood_PO_fa.out
#SBATCH --error=error_MRTA_Flood_PO_fa.err
#SBATCH --mail-user=stevepau@buffalo.edu
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#Specifies that the job will be requeued after a node failure.
#The default is that the job will not be requeued.

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR

#module load python/anaconda
#module load intel-mpi
#module load mpi4py
ulimit -s unlimited
#export PYTHONPATH= /util/academic/python/mpi4py/v2.0.0/lib/python2.7/site-packages:$PYTHONPATH
#export I_MPI_DEBUG=4
#export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
#export I_MPI_FABRICS_LIST=tcp
#source activate py27-mpi
eval "$(/util/common/python/py37/anaconda-2020.02/bin/conda shell.bash hook)"
#conda activate pytorch-1.5-dgl
conda activate /projects/academic/soumacho/steve/conda_envs/jsp
#conda install -p /home/stevepau tqdm
#conda activate tqdm
CUDA_VISIBLE_DEVICES=0,1
python training_mrta_po.py

echo "***********************************************************************"


#
echo "All Done!"