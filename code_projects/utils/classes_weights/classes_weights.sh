#!/bin/bash
#SBATCH -A "project02496" 
#SBATCH -J "classes_weights"
#SBATCH --mail-type=END
#SBATCH -e /work/scratch/ks51walo/MA/classes_weights/%x.err.%j
#SBATCH -o /work/scratch/ks51walo/MA/classes_weights/%x.out.%j
#SBATCH -n 1
#SBATCH -c 24
#SBATCH
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu # GPU
#SBATCH --time 1:00:00 # expected runtime. programm will stop if the job takes longer

# scontrol show job $SLURM_JOB_ID > sbatch_params_${SLURM_JOB_ID}.log

module purge
module load gcc/8.5 cuda/11.8 python/3.10

nvidia-smi 1>&2


source /work/home/ks51walo/test/myenv/bin/activate #add path to your own conda shell or use venv

cd /work/scratch/ks51walo/MA/classes_weights # navigates to file you want to execute


python classes_weights.py \
  --label_dir /work/scratch/ks51walo/MA/data/synthetic_100000/train/labels \
  --output_dir /work/scratch/ks51walo/MA/classes_weights