#!/bin/bash
# submit_chain.sh
# Encadena reenvios de un .sh de Khipu con --dependency=afterany, para que
# el entrenamiento continue automaticamente si no alcanza en 8h. Se detiene
# al encontrar /home/anyela.rosado/dl/lab01/${SYSTEM}_ALL_FOLDS_DONE.
#
# Uso:
#   ./submit_chain.sh CMLC_khipu.sh cmlc
#   ./submit_chain.sh BR_khipu.sh br

set -e

SCRIPT="$1"
SYSTEM="$2"

if [ -z "$SCRIPT" ] || [ -z "$SYSTEM" ]; then
    echo "Uso: ./submit_chain.sh <script.sh> <cmlc|br|hps|nddr>"
    exit 1
fi

HOME_DIR="/home/anyela.rosado/dl/lab01"
DONE_MARKER="${HOME_DIR}/${SYSTEM}_ALL_FOLDS_DONE"
MAX_JOBS=15

if [ -f "$DONE_MARKER" ]; then
    echo "Ya existe ${DONE_MARKER} -- ${SYSTEM} ya esta completo. Nada que encadenar."
    exit 0
fi

echo "Encadenando ${SYSTEM} via ${SCRIPT} (hasta ${MAX_JOBS} jobs)..."

JOB_ID=$(sbatch --parsable "$SCRIPT")
echo "Job 1/${MAX_JOBS}: $JOB_ID"

for i in $(seq 2 $MAX_JOBS); do
    JOB_ID=$(sbatch --parsable --dependency=afterany:${JOB_ID} "$SCRIPT")
    echo "Job ${i}/${MAX_JOBS}: $JOB_ID (dependency=afterany)"
done

echo ""
echo "Los jobs sobrantes terminan casi de inmediato una vez que ${DONE_MARKER} existe."
echo "Monitorear con: squeue -u \$USER"
echo "Cancelar la cadena restante con: scancel <JOB_ID_desde_donde_ya_no_quieres_mas>"
