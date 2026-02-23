#!/bin/bash

#SBATCH --job-name=superposition
#SBATCH --output=logs/coconut_%j.out
#SBATCH --error=logs/coconut_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=768GB
#SBATCH --time=72:00:00
#SBATCH --account=uvasrg_paid
#SBATCH --constraint=a100_80gb

# Create logs directory
mkdir -p /scratch/mj6ux/Projects/superposition_replication/logs

# Initialize cuda
module load cuda/12.4.1

source .venv/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Torchrun: $(which torchrun)"

cd /scratch/mj6ux/Projects/superposition_replication

# # ContinuousThought with different bottleneck sizes
# uv run python -m superposition train --preset continuous_thought_d32   # Severe: 768→32
# uv run python -m superposition train --preset continuous_thought_d64   # Heavy: 768→64
# uv run python -m superposition train --preset continuous_thought_d128  # Moderate: 768→128
# uv run python -m superposition train --preset continuous_thought_d256  # Light: 768→256

# # Coconut with different bottleneck sizes
# uv run python -m superposition train --preset coconut_d32
# uv run python -m superposition train --preset coconut_d64
# uv run python -m superposition train --preset coconut_d128
# uv run python -m superposition train --preset coconut_d256


# uv run python -m superposition train --model coconut

python -m superposition train --preset coconut_no_bottleneck
