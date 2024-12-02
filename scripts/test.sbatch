#!/bin/bash
 
#SBATCH --job-name="intro_translation_analysis"
#SBATCH --output=job_%x-%j.out
#SBATCH --error=job_%x-%j.err
#SBATCH --account=uvasrg_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpupod
#SBATCH --mem=256GB

python3 translation_superposition_test.py