#!/bin/bash
#SBATCH --job-name=hps_anuros
#SBATCH --output=logs/hps_%j.out
#SBATCH --error=logs/hps_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Correr despues de CMLC_khipu.sh (o en paralelo; usan carpetas separadas).
# Para forzar orden: sbatch --dependency=afterok:<CMLC_JOB_ID> HPS_khipu.sh
# Si no alcanzan las 8h: ./submit_chain.sh HPS_khipu.sh hps

module purge
module load gnu12/12.4.0
module load python3/3.11.11
module load cuda/12.6
source /home/anyela.rosado/venvs/cmlc/bin/activate

SYSTEM="hps"
CODE_DIR="/home/anyela.rosado/dl/lab01/cmlc"
HOME_DIR="/home/anyela.rosado/dl/lab01"
CHECKPOINT_DIR="${HOME_DIR}/checkpoints/${SYSTEM}"
RESULTS_DIR="${HOME_DIR}/resultados/${SYSTEM}"
EPOCHS_CSV="${RESULTS_DIR}/results_epochs.csv"
FINAL_CSV="${RESULTS_DIR}/results_final.csv"
USED_DATA_CSV="${RESULTS_DIR}/used_dataset.csv"

WORKDIR="/local/anyela.rosado/${SLURM_JOB_ID}"
mkdir -p "$WORKDIR" "$WORKDIR/checkpoints"
mkdir -p logs "$CHECKPOINT_DIR" "$RESULTS_DIR"

cp "$CODE_DIR"/*.py "$WORKDIR/"

CACHE_DIR_HOME="${HOME_DIR}/features_cache"
if [ -f "${CACHE_DIR_HOME}/train_features.npy" ]; then
    echo "Usando cache de features precomputadas."
    mkdir -p "$WORKDIR/features_cache"
    cp "${CACHE_DIR_HOME}/train_features.npy" "$WORKDIR/features_cache/"
    cp "${CACHE_DIR_HOME}/train_filenames.csv" "$WORKDIR/features_cache/"
    cp "${CACHE_DIR_HOME}/used_dataset.csv" "$WORKDIR/features_cache/"
    CACHE_ARG="--cache_dir $WORKDIR/features_cache"
    TRAIN_CSV_ARG="$WORKDIR/features_cache/used_dataset.csv"
    K_CLASSES_ARG=5
    SELECTION_ARGS=""
else
    echo "No se encontro cache en ${CACHE_DIR_HOME}; corriendo sin cache (correr precompute_khipu.sh primero para acelerar)."
    cp /home/anyela.rosado/dl/lab01/dataset/train.csv "$WORKDIR/"
    cp -r /home/anyela.rosado/dl/lab01/dataset/train "$WORKDIR/"
    cp -r /home/anyela.rosado/dl/lab01/dataset/test "$WORKDIR/" 2>/dev/null
    CACHE_ARG=""
    TRAIN_CSV_ARG="$WORKDIR/train.csv"
    K_CLASSES_ARG=42
    SELECTION_ARGS="--top_n_classes 5 --max_samples 10000"
fi

SUBSAMPLE_FRAC=1.0

cp -r "$CHECKPOINT_DIR"/. "$WORKDIR/checkpoints/" 2>/dev/null
cp "$EPOCHS_CSV" "$WORKDIR/results_epochs.csv" 2>/dev/null
cp "$FINAL_CSV" "$WORKDIR/results_final.csv" 2>/dev/null

cd "$WORKDIR"

echo "Ejecutando HPS MTL | Job ID: $SLURM_JOB_ID | Nodo: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# HPS MTL: backbone convolucional compartido + una cabeza recurrente por clase.
python3 train.py \
    --system hps \
    --data_dir "$WORKDIR" \
    --train_csv "$TRAIN_CSV_ARG" \
    --k_classes $K_CLASSES_ARG \
    $SELECTION_ARGS \
    --folds 5 \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-3 \
    --lr_decay_steps 27 \
    --seed 42 \
    --num_workers 6 \
    --checkpoint_dir "$WORKDIR/checkpoints" \
    --checkpoint_every 5 \
    --resume \
    --max_seconds 28300 \
    --output_csv_epochs "$WORKDIR/results_epochs.csv" \
    --output_csv_final "$WORKDIR/results_final.csv" \
    --output_used_data_csv "$WORKDIR/used_dataset.csv" \
    --subsample_frac $SUBSAMPLE_FRAC \
    $CACHE_ARG

cp -r "$WORKDIR"/checkpoints/. "$CHECKPOINT_DIR/"
cp "$WORKDIR"/results_epochs.csv "$EPOCHS_CSV"
cp "$WORKDIR"/results_final.csv  "$FINAL_CSV"
cp "$WORKDIR"/used_dataset.csv    "$USED_DATA_CSV" 2>/dev/null

cp "$WORKDIR"/results_epochs.csv "${RESULTS_DIR}/results_epochs_${SLURM_JOB_ID}.csv"
cp "$WORKDIR"/results_final.csv  "${RESULTS_DIR}/results_final_${SLURM_JOB_ID}.csv"

N_DONE=$(ls "$CHECKPOINT_DIR"/${SYSTEM}_fold*.done 2>/dev/null | wc -l)
echo "Folds completos (${SYSTEM}): ${N_DONE}/5"
if [ "$N_DONE" -ge 5 ]; then
    touch "${HOME_DIR}/${SYSTEM}_ALL_FOLDS_DONE"
    echo "${SYSTEM} completo. Ver ${EPOCHS_CSV} y ${FINAL_CSV}"
else
    echo "Faltan folds. Reenviar este script (o usar submit_chain.sh) para continuar."
fi
