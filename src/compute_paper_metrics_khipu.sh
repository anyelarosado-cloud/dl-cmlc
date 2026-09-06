#!/bin/bash
#SBATCH --job-name=paper_metrics
#SBATCH --output=logs/paper_metrics_%j.out
#SBATCH --error=logs/paper_metrics_%j.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:30:00

# Calcula Accuracy y Exact Match (Ec. 1 y 2 del paper) para un sistema ya
# entrenado, SIN reentrenar (solo inferencia sobre los folds de validacion
# ya definidos). Corre en CPU -- no compite por GPU.
# Uso: sbatch compute_paper_metrics_khipu.sh <cmlc|br|hps|nddr>

SYSTEM="$1"
if [ -z "$SYSTEM" ]; then
    echo "Uso: sbatch compute_paper_metrics_khipu.sh <cmlc|br|hps|nddr>"
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

echo "Calculando accuracy/exact match para: $SYSTEM | Nodo: $(hostname)"

python3 compute_paper_metrics.py \
    --system "$SYSTEM" \
    --cache_dir "$CACHE_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --folds 5 \
    --seed 42 \
    --output_csv "${RESULTS_DIR}/results_paper_metrics.csv"

echo "Listo. Ver ${RESULTS_DIR}/results_paper_metrics.csv"
