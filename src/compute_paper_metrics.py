#!/usr/bin/env python3
"""
compute_paper_metrics.py

Calcula Accuracy y Exact Match (Ecuaciones 1 y 2 del paper) por fold, para
un sistema ya entrenado -- SIN reentrenar nada. Reproduce el mismo split
K-Fold que uso train.py (misma semilla), carga los checkpoints existentes,
corre inferencia sobre cada fold de validacion, y calcula:

    Accuracy    = proporcion de etiquetas individuales correctamente
                  predichas sobre el total (N*K). Equivale a 1 - hamming_loss.
    Exact Match = proporcion de segmentos donde TODAS las K etiquetas
                  predichas coinciden exactamente con el ground truth.

Para BR, se combinan las predicciones de los K modelos independientes (uno
por clase) de ese fold en un solo vector por segmento antes de calcular
ambas metricas, igual a como lo hace el paper.

Uso:
    python3 compute_paper_metrics.py \
        --system cmlc \
        --cache_dir /home/anyela.rosado/dl/lab01/features_cache \
        --checkpoint_dir /home/anyela.rosado/dl/lab01/checkpoints/cmlc \
        --folds 5 \
        --output_csv /home/anyela.rosado/dl/lab01/resultados/cmlc/results_paper_metrics.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from train import build_model, CachedFoldDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system", type=str, required=True, choices=["cmlc", "br", "hps", "nddr"])
    p.add_argument("--cache_dir", type=str, required=True,
                    help="Directorio con train_features.npy, train_filenames.csv, used_dataset.csv.")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output_csv", type=str, required=True)
    return p.parse_args()


def run_inference(model, cache_array_path, row_indices, device, batch_size):
    ds = CachedFoldDataset(cache_array_path, row_indices, labels=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    n = len(ds)
    out_dim = None
    probs = None
    with torch.no_grad():
        for feats, idxs in loader:
            feats = feats.to(device)
            out = model(feats).cpu().numpy()
            if probs is None:
                out_dim = out.shape[1]
                probs = np.zeros((n, out_dim), dtype=np.float32)
            probs[idxs.numpy()] = out
    return probs


def load_model(system, n_classes, ckpt_path, device):
    model = build_model(system, n_classes=n_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    used_df = pd.read_csv(os.path.join(args.cache_dir, "used_dataset.csv"))
    class_names = used_df.columns[1:].tolist()
    filenames = used_df.iloc[:, 0].values
    labels = used_df[class_names].values.astype(int)
    print(f"Clases ({len(class_names)}): {class_names} | {len(used_df)} clips totales", flush=True)

    filename_map_df = pd.read_csv(os.path.join(args.cache_dir, "train_filenames.csv"))
    filename_to_row = {fn: i for i, fn in enumerate(filename_map_df.iloc[:, 0].tolist())}
    cache_array_path = os.path.join(args.cache_dir, "train_features.npy")

    # Mismo split que train.py: misma semilla -> mismos folds
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_indices = list(kf.split(used_df))

    rows = []
    for fold, (_, val_idx) in enumerate(fold_indices, start=1):
        val_filenames = filenames[val_idx]
        val_labels = labels[val_idx]
        row_indices = np.array([filename_to_row[fn] for fn in val_filenames])

        if args.system == "br":
            probs = np.zeros((len(val_idx), len(class_names)), dtype=np.float32)
            for k, cname in enumerate(class_names):
                ckpt_path = os.path.join(args.checkpoint_dir, f"br_fold{fold}_{cname}_checkpoint.pt")
                if not os.path.exists(ckpt_path):
                    print(f"AVISO: falta {ckpt_path}", flush=True)
                    continue
                model = load_model("br", 1, ckpt_path, device)
                probs[:, k] = run_inference(model, cache_array_path, row_indices, device, args.batch_size)[:, 0]
        else:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.system}_fold{fold}_checkpoint.pt")
            if not os.path.exists(ckpt_path):
                print(f"AVISO: falta {ckpt_path}, se omite fold {fold}", flush=True)
                continue
            model = load_model(args.system, len(class_names), ckpt_path, device)
            probs = run_inference(model, cache_array_path, row_indices, device, args.batch_size)

        preds = (probs >= args.threshold).astype(int)
        accuracy = float((preds == val_labels).mean())          # Ecuacion 1
        exact_match = float((preds == val_labels).all(axis=1).mean())  # Ecuacion 2

        rows.append({"fold": fold, "n_val": len(val_idx), "accuracy": accuracy, "exact_match": exact_match})
        print(f"[fold {fold}] n_val={len(val_idx)} accuracy={accuracy:.4f} exact_match={exact_match:.4f}", flush=True)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nGuardado: {args.output_csv}", flush=True)
    print(f"Accuracy promedio: {out_df['accuracy'].mean():.4f}", flush=True)
    print(f"Exact Match promedio: {out_df['exact_match'].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
