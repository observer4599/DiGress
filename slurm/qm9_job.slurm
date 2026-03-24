#!/bin/bash
# SLURM job script for training DiGress on the QM9 molecular dataset.
# Run this script from the DiGress repository root: sbatch slurm/qm9_job.slurm
#
# Two experiment variants are available — pick one by editing the pixi run line:
#   +experiment=qm9_no_h   — removes hydrogens, 1 000 epochs, batch_size=1 024 (faster)
#   +experiment=qm9_with_h — retains hydrogens, 2 000 epochs, batch_size=512
#
# SLURM will use your default account automatically. To use a specific account,
# add: #SBATCH --account=<name>  (find yours with: sacctmgr show assoc format=Account%15,User | grep $USER)
#
# Monitor with: squeue --me

#SBATCH --job-name=digress-qm9
#SBATCH --partition=GPUQ
#SBATCH --time=1-00:00:00        # QM9 no_h: ~1 000 epochs; with_h needs up to 2 days
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16       # Matches num_workers=16 in qm9_no_h experiment config
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1        # Request one A100; replace with v100 if unavailable
#SBATCH --output=logs/qm9_%j.out
#SBATCH --error=logs/qm9_%j.err

set -eo pipefail

mkdir -p logs

echo "=== Job started: $(date) ==="
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Ensure pixi (installed in home directory) is on PATH
export PATH="$HOME/.pixi/bin:$PATH"

# $SLURM_SUBMIT_DIR is the directory from which sbatch was called
cd "$SLURM_SUBMIT_DIR"

pixi run python src/main.py dataset=qm9 +experiment=qm9_no_h

echo "=== Job finished: $(date) ==="
