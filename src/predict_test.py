#!/usr/bin/env python3
"""
predict_test.py

Genera predicciones para el test set (sin etiquetas), promediando las
probabilidades de los N checkpoints de fold ya entrenados (ensemble simple).

No requiere reentrenar nada: solo carga los checkpoints existentes y corre
inferencia sobre test_features.npy (generado por precompute_features.py).

Uso:
    python3 predict_test.py \
        --system cmlc \
        --cache_dir /home/anyela.rosado/dl/lab01/features_cache \
        --checkpoint_dir /home/anyela.rosado/dl/lab01/checkpoints/cmlc \
        --folds 5 \
        --output_csv /home/anyela.rosado/dl/lab01/resultados/cmlc/test_predictions.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from train import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Genera predicciones de test via ensemble de fold checkpoints.")
    p.add_argument("--system", type=str, required=True, choices=["cmlc", "br", "hps", "nddr"])
    p.add_argument("--cache_dir", type=str, required=True,
                    help="Directorio con test_features.npy, test_filenames.csv y used_dataset.csv.")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--classes", type=str, default=None,
                    help="Nombres de clase separados por coma. Si se omite, se leen de used_dataset.csv.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output_csv", type=str, required=True)
    return p.parse_args()


class CachedTestDataset(Dataset):
    """Lee test_features.npy (memory-mapped, 1 canal) y lo expande a 3 canales."""

    def __init__(self, cache_array_path):
        self.array = np.load(cache_array_path, mmap_mode="r")

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx):
        feat_1ch = np.array(self.array[idx])
        feats = np.repeat(feat_1ch[np.newaxis, :, :], 3, axis=0)
        return torch.from_numpy(feats.astype(np.float32)), idx


def load_model_weights(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def run_inference(model, loader, n_test, n_classes, device):
    probs = np.zeros((n_test, n_classes), dtype=np.float32)
    with torch.no_grad():
        for feats, idxs in loader:
            feats = feats.to(device)
            out = model(feats).cpu().numpy()
            probs[idxs.numpy()] = out
    return probs


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    if args.classes is not None:
        class_names = [c.strip() for c in args.classes.split(",")]
    else:
        used_df = pd.read_csv(os.path.join(args.cache_dir, "used_dataset.csv"))
        class_names = used_df.columns[1:].tolist()
    print(f"Clases ({len(class_names)}): {class_names}", flush=True)

    test_filenames = pd.read_csv(os.path.join(args.cache_dir, "test_filenames.csv")).iloc[:, 0].tolist()
    test_ds = CachedTestDataset(os.path.join(args.cache_dir, "test_features.npy"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    n_test = len(test_ds)
    print(f"Test set: {n_test} clips.", flush=True)

    ensemble_probs = np.zeros((n_test, len(class_names)), dtype=np.float32)

    if args.system == "br":
        # BR: un modelo INDEPENDIENTE por clase y por fold -> ensemble por separado por clase
        for k, class_name in enumerate(class_names):
            class_probs = np.zeros(n_test, dtype=np.float32)
            n_found = 0
            for fold in range(1, args.folds + 1):
                ckpt_path = os.path.join(args.checkpoint_dir, f"br_fold{fold}_{class_name}_checkpoint.pt")
                if not os.path.exists(ckpt_path):
                    print(f"AVISO: falta {ckpt_path}, se omite ese fold.", flush=True)
                    continue
                model = build_model("br", n_classes=1)
                model = load_model_weights(ckpt_path, model, device)
                probs = run_inference(model, test_loader, n_test, 1, device)
                class_probs += probs[:, 0]
                n_found += 1
            ensemble_probs[:, k] = class_probs / max(n_found, 1)
            print(f"[{class_name}] ensemble de {n_found}/{args.folds} folds.", flush=True)
    else:
        # cmlc / hps / nddr: un modelo conjunto por fold -> ensemble directo
        n_found = 0
        for fold in range(1, args.folds + 1):
            ckpt_path = os.path.join(args.checkpoint_dir, f"{args.system}_fold{fold}_checkpoint.pt")
            if not os.path.exists(ckpt_path):
                print(f"AVISO: falta {ckpt_path}, se omite ese fold.", flush=True)
                continue
            model = build_model(args.system, n_classes=len(class_names))
            model = load_model_weights(ckpt_path, model, device)
            probs = run_inference(model, test_loader, n_test, len(class_names), device)
            ensemble_probs += probs
            n_found += 1
            print(f"[fold {fold}] inferencia completa.", flush=True)
        ensemble_probs /= max(n_found, 1)
        print(f"Ensemble de {n_found}/{args.folds} folds.", flush=True)

    binary_preds = (ensemble_probs >= args.threshold).astype(int)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    df_bin = pd.DataFrame(binary_preds, columns=class_names)
    df_bin.insert(0, "filename", test_filenames)
    df_bin.to_csv(args.output_csv, index=False)
    print(f"Predicciones binarias guardadas en: {args.output_csv}", flush=True)

    probs_path = args.output_csv.replace(".csv", "_probs.csv")
    df_probs = pd.DataFrame(ensemble_probs, columns=class_names)
    df_probs.insert(0, "filename", test_filenames)
    df_probs.to_csv(probs_path, index=False)
    print(f"Probabilidades guardadas en: {probs_path}", flush=True)


if __name__ == "__main__":
    main()
