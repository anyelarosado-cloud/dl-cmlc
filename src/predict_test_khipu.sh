#!/bin/bash
#SBATCH --job-name=predict_test
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Genera predicciones de test para un sistema ya entrenado (solo inferencia,
# no reentrena nada). Corre en CPU por defecto -- no compite por GPU.
# Uso: sbatch predict_test_khipu.sh <cmlc|br|hps|nddr>

SYSTEM="$1"
if [ -z "$SYSTEM" ]; then
    echo "Uso: sbatch predict_test_khipu.sh <cmlc|br|hps|nddr>"
    exit 1
fi

module purge
module load gnu12/12.4.0
module load python3/3.11.11
source /home/anyela.rosado/venvs/cmlc/bin/activate

CODE_DIR="/home/anyela.rosado/dl/lab01/cmlc"
HOME_DIR="/home/anyela.rosado/dl/lab01"
CACHE_DIR="${HOME_DIR}/features_cache"
CHECKPOINT_DIR="${HOME_DIR}/checkpoints/${SYSTEM}"
RESULTS_DIR="${HOME_DIR}/resultados/${SYSTEM}"

mkdir -p logs "$RESULTS_DIR"
cd "$CODE_DIR"

echo "Prediciendo test set para: $SYSTEM | Nodo: $(hostname)"

python3 predict_test.py \
    --system "$SYSTEM" \
    --cache_dir "$CACHE_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --folds 5 \
    --output_csv "${RESULTS_DIR}/test_predictions.csv"

echo "Listo. Ver ${RESULTS_DIR}/test_predictions.csv"
