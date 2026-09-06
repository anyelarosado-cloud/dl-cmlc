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
#
# Uso normal (CMLC/BR/HPS/NDDR):
#   sbatch compute_paper_metrics_khipu.sh <cmlc|br|hps|nddr>
#
# Uso para variantes que usan otra carpeta de checkpoints/cache pero la MISMA
# arquitectura (ej. la ablacion cmlc_nopcen, o cmlc_full):
#   sbatch compute_paper_metrics_khipu.sh <carpeta> <arquitectura> <subcarpeta_cache>
#   sbatch compute_paper_metrics_khipu.sh cmlc_nopcen cmlc features_cache_nopcen
#   sbatch compute_paper_metrics_khipu.sh cmlc_full cmlc features_cache_full

SYSTEM="$1"                        # carpeta de checkpoints/resultados
ARCH="${2:-$SYSTEM}"               # arquitectura real para --system (default: igual a SYSTEM)
CACHE_SUBDIR="${3:-features_cache}"  # subcarpeta de cache a usar (default: la de K=5/10k)

if [ -z "$SYSTEM" ]; then
    echo "Uso: sbatch compute_paper_metrics_khipu.sh <cmlc|br|hps|nddr> [arquitectura] [subcarpeta_cache]"
    exit 1
fi

module purge
module load gnu12/12.4.0
module load python3/3.11.11
source /home/anyela.rosado/venvs/cmlc/bin/activate

CODE_DIR="/home/anyela.rosado/dl/lab01/cmlc"
HOME_DIR="/home/anyela.rosado/dl/lab01"
CACHE_DIR="${HOME_DIR}/${CACHE_SUBDIR}"
CHECKPOINT_DIR="${HOME_DIR}/checkpoints/${SYSTEM}"
RESULTS_DIR="${HOME_DIR}/resultados/${SYSTEM}"

mkdir -p logs "$RESULTS_DIR"
cd "$CODE_DIR"

echo "Calculando accuracy/exact match para: $SYSTEM (arquitectura: $ARCH, cache: $CACHE_SUBDIR) | Nodo: $(hostname)"

python3 compute_paper_metrics.py \
    --system "$ARCH" \
    --cache_dir "$CACHE_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --folds 5 \
    --seed 42 \
    --output_csv "${RESULTS_DIR}/results_paper_metrics.csv"

echo "Listo. Ver ${RESULTS_DIR}/results_paper_metrics.csv"
