module purge
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
module load cuda/11.8.0-lpttyok

conda deactivate
conda activate csci2951i
