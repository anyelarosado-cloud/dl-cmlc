#!/bin/bash
#SBATCH --job-name=cmlc_anuros
#SBATCH --output=logs/cmlc_%j.out
#SBATCH --error=logs/cmlc_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=ag001
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

module purge
module load gnu12/12.4.0
module load python3/3.11.11
module load cuda/12.6
source /home/anyela.rosado/venvs/cmlc/bin/activate

# CODE_DIR: solo codigo, se reemplaza libremente.
# HOME_DIR: datos persistentes (cache/checkpoints/resultados), fuera de CODE_DIR.
SYSTEM="cmlc"
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

# Si el cache de features precomputadas existe, usar el used_dataset.csv que
# genero precompute_khipu.sh (ya reducido a 5 clases / <=10,000 filas) y saltar
# la copia de WAV crudos. Si no existe, entrenar sobre el CSV completo con
# seleccion/reduccion en vivo (mas lento, requiere los WAV).
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

# Traer checkpoints/resultados previos para que --resume continue donde quedo
cp -r "$CHECKPOINT_DIR"/. "$WORKDIR/checkpoints/" 2>/dev/null
cp "$EPOCHS_CSV" "$WORKDIR/results_epochs.csv" 2>/dev/null
cp "$FINAL_CSV" "$WORKDIR/results_final.csv" 2>/dev/null

cd "$WORKDIR"

echo "Ejecutando CMLC | Job ID: $SLURM_JOB_ID | Nodo: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python3 train.py \
    --system cmlc \
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

# Copiar de vuelta a la ruta persistente (sobreescribe con el estado actualizado)
cp -r "$WORKDIR"/checkpoints/. "$CHECKPOINT_DIR/"
cp "$WORKDIR"/results_epochs.csv "$EPOCHS_CSV"
cp "$WORKDIR"/results_final.csv  "$FINAL_CSV"
cp "$WORKDIR"/used_dataset.csv    "$USED_DATA_CSV" 2>/dev/null

# Respaldo puntual de esta corrida (no se usa para resume)
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
