#!/bin/bash
#SBATCH --job-name=precompute_features
#SBATCH --output=logs/precompute_%j.out
#SBATCH --error=logs/precompute_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Precomputa MS-PCEN una sola vez (mismo K=5/max_samples/seed que los 4 .sh
# de entrenamiento, para que el cache cubra exactamente el subconjunto que
# usaran). Correr esto ANTES de CMLC/BR/HPS/NDDR_khipu.sh.

module purge
module load gnu12/12.4.0
module load python3/3.11.11
module load cuda/12.6
source /home/anyela.rosado/venvs/cmlc/bin/activate

CODE_DIR="/home/anyela.rosado/dl/lab01/cmlc"
HOME_DIR="/home/anyela.rosado/dl/lab01"
DATASET_DIR="/home/anyela.rosado/dl/lab01/dataset"
CACHE_DIR="${HOME_DIR}/features_cache"

mkdir -p logs "$CACHE_DIR"
cd "$CODE_DIR"

echo "Nodo: $(hostname) | Precomputando en: $CACHE_DIR"

python3 precompute_features.py \
    --data_dir "$DATASET_DIR" \
    --train_csv "$DATASET_DIR/train.csv" \
    --test_dir test \
    --output_dir "$CACHE_DIR" \
    --num_workers 16 \
    --k_classes 42 \
    --top_n_classes 5 \
    --max_samples 10000 \
    --seed 42

echo ""
echo "Precomputo completo. Contenido de $CACHE_DIR:"
ls -la "$CACHE_DIR"
