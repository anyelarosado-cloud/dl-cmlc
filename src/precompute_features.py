#!/usr/bin/env python3
"""
precompute_features.py

Calcula MS-PCEN una sola vez para los clips de audio (train y, opcionalmente,
test) y los guarda en un archivo .npy empaquetado (N, n_mels, n_frames, 1
canal -- se replica a 3 canales al cargar en el Dataset). Evita recalcular el
espectrograma en cada epoca de entrenamiento.

Acepta los mismos argumentos de seleccion/reduccion que train.py
(--top_n_classes, --classes_subset, --max_samples, --seed) para precomputar
exactamente el subconjunto que se va a entrenar despues.

Genera en --output_dir:
    train_features.npy, train_filenames.csv, used_dataset.csv
    test_features.npy, test_filenames.csv (si --test_dir se especifica)

Es CPU-bound (no usa GPU) -- correr en una particion sin GPU es mas eficiente.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from audio_features import extract_features
from train import select_classes, subsample_preserving_positives  # misma logica que usa train.py


def parse_args():
    p = argparse.ArgumentParser(description="Precomputa y cachea features MS-PCEN para el dataset (o un subconjunto).")
    p.add_argument("--data_dir", type=str, required=True, help="Directorio con subcarpetas train/ (y test/ opcional).")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--train_subdir", type=str, default="train")
    p.add_argument("--test_dir", type=str, default=None, help="Subcarpeta de test dentro de data_dir (opcional).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--use_pcen", action="store_true", default=True)
    p.add_argument("--no_pcen", dest="use_pcen", action="store_false",
                    help="Usar Mel-spectrogram simple en vez de MS-PCEN (para la version 'sin PCEN' de la ablacion).")
    p.add_argument("--suffix", type=str, default="",
                    help="Sufijo para los archivos de salida (ej. '_nopcen' para cachear la variante sin PCEN aparte).")

    # Mismos argumentos de seleccion/reduccion que train.py, para precomputar
    # exactamente el subconjunto que se va a entrenar despues.
    p.add_argument("--k_classes", type=int, default=None,
                    help="Numero total de columnas de clase esperado en el CSV (validacion opcional).")
    p.add_argument("--classes_subset", type=str, default=None)
    p.add_argument("--top_n_classes", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--subsample_frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _compute_one(path, use_pcen):
    # extract_features ya devuelve 3 canales identicos; solo nos quedamos con 1
    # (se re-expande a 3 al cargar en el Dataset, para no triplicar el tamano del cache)
    feats = extract_features(path, use_pcen=use_pcen)  # (3, n_mels, n_frames)
    return feats[0]  # (n_mels, n_frames)


def precompute_split(filenames, audio_dir, use_pcen, num_workers, label):
    n = len(filenames)
    print(f"[{label}] {n} archivos a procesar (use_pcen={use_pcen})...", flush=True)

    # Primero un archivo para conocer las dimensiones exactas
    first_path = os.path.join(audio_dir, filenames[0])
    first_feat = _compute_one(first_path, use_pcen)
    n_mels, n_frames = first_feat.shape
    print(f"[{label}] dimension detectada: ({n_mels}, {n_frames})", flush=True)

    out_array = np.empty((n, n_mels, n_frames), dtype=np.float32)
    out_array[0] = first_feat

    t0 = time.time()
    done = 1
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_compute_one, os.path.join(audio_dir, fn), use_pcen): idx
            for idx, fn in enumerate(filenames) if idx != 0
        }
        for future in as_completed(futures):
            idx = futures[future]
            out_array[idx] = future.result()
            done += 1
            if done % 5000 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else float("nan")
                print(f"[{label}] {done}/{n} ({elapsed:.0f}s transcurridos, "
                      f"~{rate:.1f} clips/s, ETA {eta:.0f}s)", flush=True)

    total_time = time.time() - t0
    print(f"[{label}] completo en {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    return out_array


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = args.suffix

    df = pd.read_csv(args.train_csv)
    all_class_names = df.columns[1:].tolist()
    if args.k_classes is not None:
        assert len(all_class_names) == args.k_classes, (
            f"--k_classes={args.k_classes} no coincide con las columnas del CSV ({len(all_class_names)})."
        )

    # Misma logica que train.py: al usar el mismo seed, el subconjunto resultante es identico.
    selected_classes = select_classes(df, all_class_names, args.classes_subset, args.top_n_classes)
    if args.max_samples is not None and args.max_samples < len(df):
        n_before = len(df)
        df = subsample_preserving_positives(df, selected_classes, args.max_samples, args.seed)
        print(f"Dataset reducido con --max_samples: {len(df)} de {n_before} clips.", flush=True)
    elif args.subsample_frac < 1.0:
        n_before = len(df)
        df = df.sample(frac=args.subsample_frac, random_state=args.seed).reset_index(drop=True)
        print(f"Subsample aplicado: {len(df)} de {n_before} clips ({args.subsample_frac:.0%}).", flush=True)

    if len(selected_classes) < len(all_class_names):
        print(f"Clases seleccionadas ({len(selected_classes)}/{len(all_class_names)}): {selected_classes}", flush=True)

    used_data_path = os.path.join(args.output_dir, f"used_dataset{suffix}.csv")
    used_cols = [df.columns[0]] + selected_classes
    df[used_cols].to_csv(used_data_path, index=False)
    print(f"Subconjunto usado guardado en: {used_data_path} "
          f"({len(df)} filas, columnas: {used_cols})", flush=True)

    train_filenames = df.iloc[:, 0].tolist()
    train_audio_dir = os.path.join(args.data_dir, args.train_subdir)

    train_array = precompute_split(train_filenames, train_audio_dir, args.use_pcen,
                                    args.num_workers, label="train")
    train_out_path = os.path.join(args.output_dir, f"train_features{suffix}.npy")
    np.save(train_out_path, train_array)
    pd.DataFrame({"filename": train_filenames}).to_csv(
        os.path.join(args.output_dir, f"train_filenames{suffix}.csv"), index=False
    )
    print(f"Guardado: {train_out_path} (shape {train_array.shape}, "
          f"{train_array.nbytes / 1e9:.2f} GB)", flush=True)
    del train_array  # liberar memoria antes de procesar test, si aplica

    if args.test_dir is not None:
        test_audio_dir = os.path.join(args.data_dir, args.test_dir)
        test_filenames = sorted([f for f in os.listdir(test_audio_dir) if f.lower().endswith(".wav")])
        test_array = precompute_split(test_filenames, test_audio_dir, args.use_pcen,
                                       args.num_workers, label="test")
        test_out_path = os.path.join(args.output_dir, f"test_features{suffix}.npy")
        np.save(test_out_path, test_array)
        pd.DataFrame({"filename": test_filenames}).to_csv(
            os.path.join(args.output_dir, f"test_filenames{suffix}.csv"), index=False
        )
        print(f"Guardado: {test_out_path} (shape {test_array.shape}, "
              f"{test_array.nbytes / 1e9:.2f} GB)", flush=True)

    print("\nPrecomputo de features completo.", flush=True)


if __name__ == "__main__":
    main()
