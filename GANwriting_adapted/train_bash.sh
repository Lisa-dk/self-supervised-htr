#!/bin/bash
#SBATCH --job-name=GANwriting
#SBATCH --output=job-%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=l.d.koopmans@rug.nl
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=14G
#SBATCH --time=4:00:00

module load Python/3.10.8-GCCcore-12.2.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

bash run_train_pretrain.sh
#python3 main_run.py 0 25chan_corr
