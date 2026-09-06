"""
config.py
Hiperparametros fijos de audio (MS-PCEN) y arquitectura CRNN. Los
hiperparametros de entrenamiento (epocas, batch size, lr, split, etc.) se
pasan por linea de comandos a train.py, no viven aqui.

Valores segun Olcay et al. (2026), "How to analyse overlapping sounds in the
marine environment using supervised multi-label classification", npj Acoustics.
Los parametros dependientes del sample rate (N_FFT, HOP_LENGTH) se recalcularon
porque el paper usa 96 kHz y este dataset esta a 22.05 kHz, preservando la
misma duracion de ventana (21 ms) y el mismo numero de bandas Mel (64).
"""

N_CLASSES = 42
CLIP_DURATION_SEC = 3.0
SAMPLE_RATE = 22050

# MS-PCEN
N_FFT = 512                    # ~23.2 ms a 22050 Hz (equivalente a 2048 @ 96kHz)
HOP_LENGTH = int(N_FFT * 0.25) # 75% overlap, igual al paper
N_MELS = 64

PCEN_DELTA = 0.05
PCEN_ALPHA = 0.98
PCEN_R = 0.5
PCEN_EPS = 1.4
PCEN_GAIN_TIME_CONSTANT = 0.4

N_CHANNELS = 3   # MS-PCEN replicado en 3 canales, igual al paper

# Arquitectura CRNN
CONV_KERNELS = [160, 192, 512]         # filtros por bloque (CMLC)
CONV_KERNEL_SIZE = 5
POOL_SIZES = [(5, 1), (4, 1), (2, 1)]  # (freq, time) por bloque
CONV_DROPOUT = 0.25

GRU_HIDDEN = 128
GRU_DROPOUT = 0.2
N_RECURRENT_BLOCKS = 3
