#!/bin/bash
#SBATCH --job-name=weather_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

module load class/default
module load cs137/2026spring

export OMP_NUM_THREADS=4

cd /cluster/tufts/c26sp1cs0137/ashen05

python cnn2.py