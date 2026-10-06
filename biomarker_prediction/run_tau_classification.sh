#!/bin/bash
#SBATCH --job-name=tau_pca_search
#SBATCH --output=/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/logs/tau_classification_%j.out
#SBATCH --error=/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/logs/tau_classification_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn2

# Create logs directory if it doesn't exist
mkdir -p /ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/logs
mkdir -p /ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/figures
mkdir -p /ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/models
# Move to the correct directory
cd /ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction

# Run your script
echo "SLURM job started in $(pwd) at $(date)"
python tau_classification.py --use_gpu "$@"
echo "Job finished at $(date)"