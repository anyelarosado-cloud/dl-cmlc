"""
model.py
Implementacion propia (PyTorch) de las 4 arquitecturas CRNN comparadas en
Olcay et al. (2026): CMLC, BR, HPS MTL y NDDR MTL.

Backbone base compartido: 3 bloques Conv2D(5x5)+BN+ReLU+MaxPool+Dropout(0.25),
seguidos de 3 bloques GRU bidireccional (128 unidades, dropout 0.2, direcciones
combinadas por promedio), Temporal Max Pooling, y capa(s) densa(s) con sigmoide.

Diferencias entre sistemas (Fig. 8 del paper):
    - CMLC: 1 backbone compartido, salida densa de K unidades.
    - BR: K backbones independientes de salida 1 (reusa CMLC_CRNN(n_classes=1),
      entrenado K veces por separado, ver train.py --system br).
    - HPS MTL: backbone conv compartido + K cabezas recurrentes independientes.
    - NDDR MTL: K ramas conv independientes fusionadas por capas NDDR (1x1 conv
      con inicializador diagonal) tras cada bloque, con skip connections entre
      escalas, seguidas de K cabezas recurrentes (igual a HPS MTL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, dropout=cfg.CONV_DROPOUT):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=cfg.CONV_KERNEL_SIZE,
            padding=cfg.CONV_KERNEL_SIZE // 2,  # zero-padding, igual al paper
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)  # pool_size = (freq, time)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CMLC_CRNN(nn.Module):
    """
    Conventional Multi-Label Classifier: una sola red CRNN compartida,
    con una capa densa final de `n_classes` unidades (sigmoide).
    """

    def __init__(self, n_classes: int = cfg.N_CLASSES, in_channels: int = cfg.N_CHANNELS,
                 conv_kernels=None, gru_hidden: int = cfg.GRU_HIDDEN, n_mels: int = cfg.N_MELS):
        super().__init__()
        conv_kernels = conv_kernels or cfg.CONV_KERNELS

        # --- Bloques convolucionales ---
        blocks = []
        c_in = in_channels
        for c_out, pool in zip(conv_kernels, cfg.POOL_SIZES):
            blocks.append(ConvBlock(c_in, c_out, pool))
            c_in = c_out
        self.conv_blocks = nn.Sequential(*blocks)
        self.final_conv_channels = c_in

        # Tamano de entrada de la GRU calculado analiticamente (no lazy), para
        # que las capas existan desde __init__ y el checkpoint se cargue bien.
        freq_dim = n_mels
        for pool in cfg.POOL_SIZES:
            freq_dim = freq_dim // pool[0]  # pool[0] = pooling en el eje de frecuencia
        gru_input_size = c_in * freq_dim
        assert freq_dim >= 1, (
            f"n_mels={n_mels} es demasiado pequeno para los pool_sizes configurados "
            f"(freq_dim resultante = {freq_dim}). Ajustar N_MELS o POOL_SIZES en config.py."
        )

        self.gru_hidden = gru_hidden
        self.gru_dropout = cfg.GRU_DROPOUT
        self.n_recurrent_blocks = cfg.N_RECURRENT_BLOCKS

        layers = []
        in_size = gru_input_size
        for i in range(self.n_recurrent_blocks):
            layers.append(nn.GRU(
                input_size=in_size,
                hidden_size=self.gru_hidden,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,  # el dropout se aplica manualmente a la entrada, como en el paper
            ))
            in_size = self.gru_hidden  # tras promediar direcciones, la siguiente capa recibe gru_hidden
        self.grus = nn.ModuleList(layers)
        self.input_dropout = nn.Dropout(self.gru_dropout)

        self.tmp = nn.AdaptiveMaxPool1d(1)  # Temporal Max Pooling -> 1 vector por segmento
        self.classifier = nn.Linear(gru_hidden, n_classes)  # bidireccional promediado -> gru_hidden (no *2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 3, n_mels, n_frames)
        x = self.conv_blocks(x)  # (batch, C, F', T')
        b, c, f, t = x.shape

        # Time-distributed flatten: cada paso temporal se aplana (C*F')
        x = x.permute(0, 3, 1, 2).contiguous()   # (batch, T', C, F')
        x = x.view(b, t, c * f)                   # (batch, T', C*F')

        h = x
        for gru in self.grus:
            h = self.input_dropout(h)
            out, _ = gru(h)                              # (batch, T', 2*gru_hidden)
            out = out.view(b, out.shape[1], 2, self.gru_hidden)
            h = out.mean(dim=2)                            # promedio de direcciones -> (batch, T', gru_hidden)

        # Temporal Max Pooling
        h = h.permute(0, 2, 1)          # (batch, gru_hidden, T')
        h = self.tmp(h).squeeze(-1)     # (batch, gru_hidden)

        logits = self.classifier(h)     # (batch, n_classes)
        probs = self.sigmoid(logits)
        return probs


def compute_freq_dim(n_mels: int, pool_sizes) -> int:
    """Dimension de frecuencia tras aplicar N maxpools sucesivos (misma logica
    usada en CMLC_CRNN, factorizada para reusarla en HPS/NDDR MTL)."""
    freq_dim = n_mels
    for pool in pool_sizes:
        freq_dim = freq_dim // pool[0]
    assert freq_dim >= 1, (
        f"n_mels={n_mels} demasiado pequeno para los pool_sizes {pool_sizes} "
        f"(freq_dim resultante = {freq_dim})."
    )
    return freq_dim


class RecurrentHead(nn.Module):
    """
    Cabeza recurrente independiente por clase: 3x GRU bidireccional (128
    unidades, dropout 0.2 en la entrada, direcciones combinadas por
    promedio) + Temporal Max Pooling + Linear(1) + sigmoide.

    Usada como "rama especifica de clase" tanto en HPS MTL (donde el
    backbone convolucional que la alimenta es compartido) como en NDDR MTL
    (donde cada clase tiene ademas su propio backbone convolucional).
    """

    def __init__(self, input_size: int, gru_hidden: int = cfg.GRU_HIDDEN,
                 n_recurrent_blocks: int = cfg.N_RECURRENT_BLOCKS,
                 dropout: float = cfg.GRU_DROPOUT):
        super().__init__()
        self.gru_hidden = gru_hidden
        layers = []
        in_size = input_size
        for _ in range(n_recurrent_blocks):
            layers.append(nn.GRU(
                input_size=in_size, hidden_size=gru_hidden,
                batch_first=True, bidirectional=True, dropout=0.0,
            ))
            in_size = gru_hidden
        self.grus = nn.ModuleList(layers)
        self.input_dropout = nn.Dropout(dropout)
        self.tmp = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(gru_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, T', input_size)
        b = x.shape[0]
        h = x
        for gru in self.grus:
            h = self.input_dropout(h)
            out, _ = gru(h)
            out = out.view(b, out.shape[1], 2, self.gru_hidden)
            h = out.mean(dim=2)
        h = h.permute(0, 2, 1)
        h = self.tmp(h).squeeze(-1)         # (batch, gru_hidden)
        logit = self.classifier(h)          # (batch, 1)
        return self.sigmoid(logit)          # (batch, 1)


class HPS_MTL_CRNN(nn.Module):
    """
    Hard Parameter Sharing MTL: UN backbone convolucional compartido entre
    todas las clases (kernels [96,128,192], igual al paper), seguido de K
    cabezas recurrentes INDEPENDIENTES (una por clase), cuyas salidas
    (cada una un escalar sigmoide) se concatenan en un vector de K unidades.
    """

    def __init__(self, n_classes: int = cfg.N_CLASSES, in_channels: int = cfg.N_CHANNELS,
                 conv_kernels=(96, 128, 192), gru_hidden: int = cfg.GRU_HIDDEN,
                 n_mels: int = cfg.N_MELS):
        super().__init__()
        self.n_classes = n_classes

        blocks = []
        c_in = in_channels
        for c_out, pool in zip(conv_kernels, cfg.POOL_SIZES):
            blocks.append(ConvBlock(c_in, c_out, pool))
            c_in = c_out
        self.conv_blocks = nn.Sequential(*blocks)

        freq_dim = compute_freq_dim(n_mels, cfg.POOL_SIZES)
        head_input_size = c_in * freq_dim

        self.heads = nn.ModuleList([
            RecurrentHead(head_input_size, gru_hidden=gru_hidden) for _ in range(n_classes)
        ])

    def forward(self, x):
        x = self.conv_blocks(x)                    # (batch, C, F', T') -- COMPARTIDO
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (batch, T', C*F')

        outs = [head(x) for head in self.heads]     # K x (batch, 1)
        return torch.cat(outs, dim=1)                # (batch, K)


class NDDRConvBranch(nn.Module):
    """Bloque convolucional (Conv2D+BN+ReLU+MaxPool+Dropout) para UNA rama
    de clase en NDDR MTL. Identico en forma a ConvBlock pero se instancia
    K veces (una por clase) en vez de compartirse."""

    def __init__(self, in_channels, out_channels, pool_size, dropout=cfg.CONV_DROPOUT):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, pool_size, dropout)

    def forward(self, x):
        return self.block(x)


class NDDRLayer(nn.Module):
    """
    Fusiona los feature maps de las K ramas tras un bloque conv: concatena
    a lo largo del canal, BatchNorm, y una Conv2D 1x1 POR CLASE que reduce
    de (K*C) canales de vuelta a C canales, con un inicializador diagonal
    que prioriza la propia rama (self_weight=0.6) sobre las demas
    (other_weight=0.1), tal como describe el paper para las capas NDDR.
    """

    def __init__(self, n_classes: int, channels_per_branch: int,
                 self_weight: float = 0.6, other_weight: float = 0.1, weight_decay: float = 0.01):
        super().__init__()
        self.n_classes = n_classes
        self.c = channels_per_branch
        self.bn = nn.BatchNorm2d(n_classes * channels_per_branch)
        self.fuse = nn.ModuleList([
            nn.Conv2d(n_classes * channels_per_branch, channels_per_branch, kernel_size=1)
            for _ in range(n_classes)
        ])
        self._diagonal_init(self_weight, other_weight)
        self.weight_decay = weight_decay  # aplicado externamente via optimizer (param group), ver train.py

    def _diagonal_init(self, self_weight, other_weight):
        with torch.no_grad():
            for k, conv in enumerate(self.fuse):
                conv.weight.zero_()
                conv.bias.zero_()
                for c_out in range(self.c):
                    for src_class in range(self.n_classes):
                        src_channel = src_class * self.c + c_out
                        w = self_weight if src_class == k else other_weight
                        conv.weight[c_out, src_channel, 0, 0] = w

    def forward(self, branch_outputs):
        # branch_outputs: lista de K tensores (batch, C, F, T)
        concat = torch.cat(branch_outputs, dim=1)   # (batch, K*C, F, T)
        concat = self.bn(concat)
        fused = [self.fuse[k](concat) for k in range(self.n_classes)]  # K x (batch, C, F, T)
        return fused


class NDDR_MTL_CRNN(nn.Module):
    """
    Neural Discriminative Dimensionality Reduction MTL: K ramas
    convolucionales independientes (kernels=64 cada una, igual al paper)
    que se fusionan via capas NDDR tras cada uno de los 3 bloques conv,
    mas conexiones de salto que combinan las 3 escalas (bilinear resize a
    la resolucion del ultimo bloque + concat + Conv2D 1x1 con inicializador
    diagonal ponderado 0.6/0.2/0.2), seguido de K cabezas recurrentes
    independientes (igual a HPS MTL).

    ADVERTENCIA DE ESCALA: con K=42 esta arquitectura crea 42 ramas
    convolucionales+recurrentes en paralelo -> es, por lejos, la mas pesada
    en memoria/computo de las 4. Usar --classes_subset en train.py si no es
    viable entrenarla sobre las 42 clases completas.
    """

    def __init__(self, n_classes: int = cfg.N_CLASSES, in_channels: int = cfg.N_CHANNELS,
                 branch_kernels: int = 64, gru_hidden: int = cfg.GRU_HIDDEN, n_mels: int = cfg.N_MELS):
        super().__init__()
        self.n_classes = n_classes
        self.branch_kernels = branch_kernels

        # K ramas convolucionales independientes, 3 bloques cada una
        self.branches = nn.ModuleList([
            nn.ModuleList([
                NDDRConvBranch(in_channels if i == 0 else branch_kernels, branch_kernels, pool)
                for i, pool in enumerate(cfg.POOL_SIZES)
            ])
            for _ in range(n_classes)
        ])

        # Una capa NDDR tras cada uno de los 3 bloques conv
        self.nddr_layers = nn.ModuleList([
            NDDRLayer(n_classes, branch_kernels) for _ in cfg.POOL_SIZES
        ])

        # Conexion de salto: 1x1 conv por clase que reduce 3*branch_kernels -> branch_kernels,
        # con inicializador diagonal ponderado (0.6 capa final, 0.2 cada capa anterior)
        self.skip_reduce = nn.ModuleList([
            nn.Conv2d(3 * branch_kernels, branch_kernels, kernel_size=1) for _ in range(n_classes)
        ])
        self._init_skip_reduce()

        freq_dim = compute_freq_dim(n_mels, cfg.POOL_SIZES)
        head_input_size = branch_kernels * freq_dim
        self.heads = nn.ModuleList([
            RecurrentHead(head_input_size, gru_hidden=gru_hidden) for _ in range(n_classes)
        ])

    def _init_skip_reduce(self):
        # Pesos 0.6 (capa NDDR final / ultimo bloque) y 0.2 (las 2 capas anteriores),
        # aplicados a la sub-porcion de canales que corresponde a cada escala.
        weights = [0.2, 0.2, 0.6]  # orden: bloque1, bloque2, bloque3(final)
        with torch.no_grad():
            for conv in self.skip_reduce:
                conv.weight.zero_()
                conv.bias.zero_()
                for c_out in range(self.branch_kernels):
                    for scale_idx, w in enumerate(weights):
                        src_channel = scale_idx * self.branch_kernels + c_out
                        conv.weight[c_out, src_channel, 0, 0] = w

    def forward(self, x):
        b = x.shape[0]
        branch_feats = [x for _ in range(self.n_classes)]  # entrada identica a cada rama
        scale_outputs = []  # guarda la salida NDDR de cada uno de los 3 bloques, por clase

        for block_idx in range(len(cfg.POOL_SIZES)):
            conv_outs = [self.branches[k][block_idx](branch_feats[k]) for k in range(self.n_classes)]
            fused = self.nddr_layers[block_idx](conv_outs)   # K x (batch, C, F, T) fusionados
            branch_feats = fused
            scale_outputs.append(fused)

        # Skip connections: resize (bilinear) las 3 escalas a la resolucion final y concatenar
        final_shape = scale_outputs[-1][0].shape[-2:]  # (F_final, T_final)
        outs = []
        for k in range(self.n_classes):
            resized = [
                F.interpolate(scale_outputs[s][k], size=final_shape, mode="bilinear", align_corners=False)
                for s in range(len(scale_outputs))
            ]
            concat = torch.cat(resized, dim=1)          # (batch, 3*branch_kernels, F_final, T_final)
            reduced = self.skip_reduce[k](concat)        # (batch, branch_kernels, F_final, T_final)
            outs.append(reduced)

        # Cabezas recurrentes por clase (igual que HPS MTL)
        logits = []
        for k in range(self.n_classes):
            feat = outs[k]
            _, c, f, t = feat.shape
            feat = feat.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
            logits.append(self.heads[k](feat))          # (batch, 1)

        return torch.cat(logits, dim=1)                  # (batch, K)
