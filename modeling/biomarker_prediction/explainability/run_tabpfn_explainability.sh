#!/bin/bash
#SBATCH --job-name=tabpfn_explainability
#SBATCH --output=/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/logs/tabpfn_explainability_%j.out
#SBATCH --error=/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/logs/tabpfn_explainability_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn2

# Create logs directory if it doesn't exist
mkdir -p /ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/logs

# Move to the correct directory
cd /ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/explainability

# Run the explainability script
# Note: GPU is automatically used by TabPFN models if available.
# Outputs go to <output_dir>/test/, <output_dir>/train/, <output_dir>/full/ (train+full when train data exists).
# Plots use top 15 features by default (--top_n_plot 15); full results are always saved in CSVs.
# Example: sbatch run_tabpfn_explainability.sh --use_shap --use_ale
echo "SLURM job started in $(pwd) at $(date)"
echo "Running TabPFN explainability script with arguments: $@"
python tabpfn_explainability.py "$@"
echo "Job finished at $(date)"

