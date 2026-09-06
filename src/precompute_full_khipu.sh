#!/bin/bash
#SBATCH --job-name=precompute_full
#SBATCH --output=logs/precompute_full_%j.out
#SBATCH --error=logs/precompute_full_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Precomputa MS-PCEN para el dataset COMPLETO (42 especies, 62,191 clips,
# sin reducir). Se guarda en una carpeta SEPARADA (features_cache_full/)
# para no pisar el cache reducido (features_cache/, usado por CMLC/BR/HPS/NDDR).
# No incluye test: el test set YA esta cacheado completo en features_cache/
# (nunca se redujo por clase), se reutiliza directamente desde ahi.

module purge
module load gnu12/12.4.0
module load python3/3.11.11
module load cuda/12.6
source /home/anyela.rosado/venvs/cmlc/bin/activate

CODE_DIR="/home/anyela.rosado/dl/lab01/cmlc"
HOME_DIR="/home/anyela.rosado/dl/lab01"
DATASET_DIR="/home/anyela.rosado/dl/lab01/dataset"
CACHE_DIR="${HOME_DIR}/features_cache_full"

mkdir -p logs "$CACHE_DIR"
cd "$CODE_DIR"

echo "Nodo: $(hostname) | Precomputando dataset completo en: $CACHE_DIR"

python3 precompute_features.py \
    --data_dir "$DATASET_DIR" \
    --train_csv "$DATASET_DIR/train.csv" \
    --output_dir "$CACHE_DIR" \
    --num_workers 16 \
    --k_classes 42

# Copiar (no recalcular) el test cache ya existente, para que features_cache_full/
# quede autocontenido y predict_test.py / compute_paper_metrics.py funcionen
# apuntando a un solo --cache_dir, igual que con los demas sistemas.
EXISTING_TEST_CACHE="${HOME_DIR}/features_cache"
if [ -f "${EXISTING_TEST_CACHE}/test_features.npy" ]; then
    echo "Copiando test cache existente (no se recalcula)..."
    cp "${EXISTING_TEST_CACHE}/test_features.npy" "$CACHE_DIR/"
    cp "${EXISTING_TEST_CACHE}/test_filenames.csv" "$CACHE_DIR/"
else
    echo "ADVERTENCIA: no se encontro test cache en ${EXISTING_TEST_CACHE}. "
    echo "predict_test.py no podra usarse con este cache hasta generarlo."
fi

echo ""
echo "Precomputo (dataset completo) terminado. Contenido de $CACHE_DIR:"
ls -la "$CACHE_DIR"
