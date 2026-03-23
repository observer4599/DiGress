#!/bin/bash
# SLURM job script for training DiGress on the SBM graph benchmark dataset.
# Run this script from the DiGress repository root: sbatch sbm_job.slurm
#
# SLURM will use your default account automatically. To use a specific account,
# add: #SBATCH --account=<name>  (find yours with: sacctmgr show assoc format=Account%15,User | grep $USER)
#
# Monitor with: squeue --me

#SBATCH --job-name=digress-sbm
#SBATCH --partition=GPUQ
#SBATCH --time=0-04:00:00        # SBM trains for 50 000 epochs; max on GPUQ is 14 days
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1        # Request one A100; replace with v100 if unavailable
#SBATCH --output=logs/sbm_%j.out
#SBATCH --error=logs/sbm_%j.err

set -eo pipefail

mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Ensure pixi (installed in home directory) is on PATH
source "$HOME/.bashrc"

# $SLURM_SUBMIT_DIR is the directory from which sbatch was called
cd "$SLURM_SUBMIT_DIR"

# +experiment=sbm is required: sets batch_size=12 to avoid OOM (default is 512)
pixi run python src/main.py dataset=sbm +experiment=sbm

echo "=== Job finished: $(date) ==="
