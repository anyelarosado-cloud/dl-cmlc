# CMLC/BR/HPS/NDDR

Implementacion de los 4 sistemas de clasificacion multi-etiqueta comparados en
Olcay et al. (2026), *"How to analyse overlapping sounds in the marine
environment using supervised multi-label classification"* (npj Acoustics),
adaptados al dataset de anuros de la Amazonia (42 especies, 62,191 clips).

## Archivos

| Archivo | Que hace |
|---|---|
| `config.py` | Hiperparametros fijos de audio (MS-PCEN) y arquitectura CRNN |
| `audio_features.py` | Calculo de MS-PCEN / Mel-spectrogram desde WAV |
| `model.py` | Las 4 arquitecturas (CMLC, BR, HPS MTL, NDDR MTL) en PyTorch |
| `train.py` | Script de entrenamiento unificado (`--system cmlc\|br\|hps\|nddr`) |
| `precompute_features.py` | Precomputa y cachea features MS-PCEN |
| `precompute_khipu.sh` | Job Slurm para correr el precomputo |
| `CMLC_khipu.sh` / `BR_khipu.sh` / `HPS_khipu.sh` / `NDDR_khipu.sh` | Jobs Slurm de entrenamiento |
| `submit_chain.sh` | Reenvia un job automaticamente si no termina en 8h |

## Configuracion del entorno (una vez, en el nodo de acceso de KHIPU)

```bash
module purge
module load gnu12/12.4.0
module load python3/3.11.11

python3 -m venv ~/venvs/cmlc
source ~/venvs/cmlc/bin/activate
pip install --upgrade pip

module load cuda/12.6
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas librosa scikit-learn
```

Verificar: `python3 -c "import torch; print(torch.__version__)"` debe mostrar
algo como `2.5.1+cu121` (no `+cpu`).

## Orden de ejecucion

```bash
sbatch precompute_khipu.sh      # una sola vez (~5 min)
sbatch CMLC_khipu.sh            # medido: ~1.24h
sbatch BR_khipu.sh              # medido: ~6.17h (25 redes independientes)
sbatch HPS_khipu.sh             # medido: ~1.22h
sbatch NDDR_khipu.sh            # medido: ~2.61h

# si alguno no termina en 8h:
./submit_chain.sh BR_khipu.sh br
```

**Resultado (F1-macro promedio, 5 folds x 5 clases):** BR 0.9745 > NDDR MTL
0.9506 > HPS MTL 0.9485 > CMLC 0.9460 -- coincide con el hallazgo del paper
original (BR mejor, NDDR MTL de cerca en segundo lugar). Por Accuracy/Exact
Match los 4 sistemas quedan mas parejos (diferencia maxima ~0.3 puntos
porcentuales), con NDDR MTL levemente adelante. BR y NDDR MTL cuestan
significativamente mas computo (5x y 2.1x el tiempo de CMLC/HPS,
respectivamente) para su mejora de desempeño -- de ahi que CMLC (el mas
eficiente) sea el elegido para escalar a las 42 especies completas.

## Decisiones metodologicas

**CMLC/BR/HPS/NDDR sobre K=5 clases, max 10,000 muestras** (`--top_n_classes 5
--max_samples 10000`): iguala la escala exacta del paper original (5 clases),
lo que permite implementar y comparar los 4 sistemas de forma fiel en vez de
limitarse a uno solo por restricciones computacionales del dataset completo
(42 especies, 62k clips). El muestreo preserva todos los positivos de las
clases elegidas antes de rellenar con negativos (`subsample_preserving_positives`
en `train.py`), evitando dejar alguna clase con muy pocos ejemplos.

**K-Fold con K=5 (no 4)**: el paper usa K=4 (75/25 por fold); se ajusto a K=5
para que cada particion respete la proporcion 80/20, manteniendo el enfoque de
validacion cruzada del paper (reporte de variabilidad entre folds).

**30 epocas (no 100)**: el paper usa 100 epocas fijas sin early stopping; en
pruebas reales se observo convergencia empirica muy temprana (F1-macro de 0.58
a 0.84 en las primeras 2 epocas). `--lr_decay_steps` se escalo proporcionalmente
de 90 a 27 para mantener el mismo patron relativo de decaimiento del paper.

**Cache de features precomputadas** (`precompute_features.py` /
`--cache_dir`): MS-PCEN se calcula una sola vez y se guarda en un `.npy`
empaquetado, en vez de recalcularse desde WAV en cada epoca. `precompute_khipu.sh`
usa los mismos `--top_n_classes`/`--max_samples`/`--seed` que los 4 `.sh` de
entrenamiento, asi el cache cubre exactamente el subconjunto que se va a usar.

**BR**: entrena un modelo independiente por clase y por fold (K=5, 5 folds =
25 entrenamientos), a diferencia de CMLC/HPS/NDDR que entrenan un modelo
conjunto por fold.

**NDDR MTL**: la arquitectura mas pesada (K ramas paralelas); usa
`batch_size=32` (vs 64 en las demas) como margen de seguridad de memoria.

## Notas practicas

- Checkpointing: cada fold guarda su estado cada `--checkpoint_every` epocas;
  con `--resume`, un job interrumpido retoma exactamente donde quedo.
- `--max_seconds 28300` corta el entrenamiento limpiamente (guardando
  checkpoint) antes de que Slurm mate el job por el limite de 8h.
- Si `results_final.csv`/`results_epochs.csv` de un sistema necesitan
  rehacerse desde cero, borrar checkpoints y resultados juntos:
  `rm -rf checkpoints/<sistema> resultados/<sistema>` (nunca borrar solo uno).
