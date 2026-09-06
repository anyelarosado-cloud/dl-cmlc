#!/usr/bin/env python3
"""
train.py

Script de entrenamiento unificado para los 4 sistemas de Olcay et al. (2026):
CMLC, BR, HPS MTL y NDDR MTL. El sistema se elige con --system {cmlc,br,hps,nddr}.

Formato de salida (comun a los 4 sistemas, para compararlos en un notebook):
    - results_epochs.csv: fold, epoch, class, train_loss, val_loss,
      val_f1_macro, val_f1_micro, val_hamming_loss, lr, epoch_time_sec
      ('class' vacio para cmlc/hps/nddr; nombre de especie solo para 'br')
    - results_final.csv: fold, class, precision, recall, f1, support
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support

import config as cfg
from audio_features import extract_features
from model import CMLC_CRNN, HPS_MTL_CRNN, NDDR_MTL_CRNN


# ----------------------------------------------------------------------
# Argparse
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento CRNN multi-etiqueta (CMLC / BR / HPS MTL / NDDR MTL).")

    p.add_argument("--system", type=str, required=True, choices=["cmlc", "br", "hps", "nddr"],
                    help="Que sistema entrenar: cmlc, br, hps (Hard Parameter Sharing MTL), "
                         "o nddr (Neural Discriminative Dimensionality Reduction MTL).")

    # Datos
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_dir", type=str, default=None)
    p.add_argument("--train_subdir", type=str, default="train")

    # Problema
    p.add_argument("--k_classes", type=int, default=42,
                    help="Numero total de columnas de clase en el CSV.")
    p.add_argument("--classes_subset", type=str, default=None,
                    help="Nombres de clase separados por coma, para entrenar solo sobre ese subconjunto.")
    p.add_argument("--top_n_classes", type=int, default=None,
                    help="Alternativa a --classes_subset: usar las N clases mas frecuentes.")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Entrenamiento
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay_rate", type=float, default=0.75)
    p.add_argument("--lr_decay_steps", type=int, default=90)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--resume", action="store_true")

    # Salidas
    p.add_argument("--output_csv_epochs", type=str, required=True)
    p.add_argument("--output_csv_final", type=str, required=True)
    p.add_argument("--output_used_data_csv", type=str, default=None,
                    help="Si se especifica, guarda el subconjunto final (filename + clases) usado en la corrida.")

    p.add_argument("--max_seconds", type=float, default=None)

    # Cache de features precomputadas y reduccion de dataset
    p.add_argument("--cache_dir", type=str, default=None,
                    help="Directorio con features precomputadas (ver precompute_features.py).")
    p.add_argument("--subsample_frac", type=float, default=1.0,
                    help="Fraccion (0-1] del dataset de train a usar.")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Limite absoluto de clips, preservando filas con positivos. Prioridad sobre --subsample_frac.")

    return p.parse_args()


# ----------------------------------------------------------------------
# Seleccion de subconjunto de clases
# ----------------------------------------------------------------------
def select_classes(df, all_class_names, classes_subset, top_n_classes):
    if classes_subset is not None:
        names = [c.strip() for c in classes_subset.split(",")]
        missing = [c for c in names if c not in all_class_names]
        assert not missing, f"Clases en --classes_subset no encontradas en el CSV: {missing}"
        return names
    if top_n_classes is not None:
        counts = df[all_class_names].sum().sort_values(ascending=False)
        return counts.head(top_n_classes).index.tolist()
    return all_class_names


def subsample_preserving_positives(df, selected_classes, max_samples, seed):
    """
    Reduce el dataset a max_samples filas, preservando todas las que tienen al
    menos un positivo en las clases seleccionadas, y rellenando el resto con
    filas negativas al azar. Si los positivos superan max_samples, tambien se
    muestrea entre ellos (con la semilla fija).
    """
    rng = np.random.default_rng(seed)
    has_positive = df[selected_classes].sum(axis=1) > 0
    positive_df = df[has_positive]
    negative_df = df[~has_positive]

    if len(positive_df) >= max_samples:
        idx = rng.choice(len(positive_df), size=max_samples, replace=False)
        result = positive_df.iloc[idx]
    else:
        n_fill = min(max_samples - len(positive_df), len(negative_df))
        neg_idx = rng.choice(len(negative_df), size=n_fill, replace=False)
        result = pd.concat([positive_df, negative_df.iloc[neg_idx]], ignore_index=True)

    # barajar el orden final (si no, quedarian todos los positivos primero)
    result = result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return result


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class FoldAudioDataset(Dataset):
    def __init__(self, audio_dir, filenames, labels=None):
        self.audio_dir = audio_dir
        self.filenames = filenames
        self.labels = labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.audio_dir, fname)
        feats = extract_features(path, use_pcen=True)
        feats_t = torch.from_numpy(feats)
        if self.labels is not None:
            y_t = torch.from_numpy(self.labels[idx].astype(np.float32))
            return feats_t, y_t
        return feats_t, fname


class CachedFoldDataset(Dataset):
    """
    Version que lee features MS-PCEN ya precomputadas (ver precompute_features.py)
    desde un array .npy empaquetado (memory-mapped, 1 canal), en vez de recalcular
    el espectrograma desde el WAV crudo en cada acceso. El array se abre con
    mmap_mode='r' -- bajo el metodo de arranque 'fork' de multiprocessing (default
    en Linux), cada worker del DataLoader hereda el mapeo sin necesidad de volver
    a leer el archivo completo a memoria.

    `row_indices` son los indices (dentro del array grande, ya cacheado para TODO
    el dataset) correspondientes a este fold/split especifico.
    """

    def __init__(self, cache_array_path, row_indices, labels=None):
        self.cache_array_path = cache_array_path
        self.array = np.load(cache_array_path, mmap_mode="r")
        self.row_indices = row_indices
        self.labels = labels  # (len(row_indices), K) o None

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, idx):
        row = self.row_indices[idx]
        feat_1ch = np.array(self.array[row])          # copia fuera del mmap, (n_mels, n_frames)
        feats = np.repeat(feat_1ch[np.newaxis, :, :], 3, axis=0)  # 3 canales, igual a extract_features
        feats_t = torch.from_numpy(feats.astype(np.float32))
        if self.labels is not None:
            y_t = torch.from_numpy(self.labels[idx].astype(np.float32))
            return feats_t, y_t
        return feats_t, idx


def build_filename_to_row(cache_filenames_csv):
    """Mapea filename -> indice de fila en el array .npy cacheado (mismo orden
    en que precompute_features.py los proceso, preservado desde train.csv)."""
    cache_df = pd.read_csv(cache_filenames_csv)
    return {fn: i for i, fn in enumerate(cache_df.iloc[:, 0].tolist())}


# ----------------------------------------------------------------------
# CSV incremental
# ----------------------------------------------------------------------
def open_csv_writer(path, fieldnames):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer


# ----------------------------------------------------------------------
# Checkpointing (parametrizado por un "tag" que identifica la unidad de
# entrenamiento: para cmlc/hps/nddr el tag es solo el fold; para br el tag
# combina fold + nombre de clase, porque cada clase se entrena por separado)
# ----------------------------------------------------------------------
def checkpoint_path(checkpoint_dir, tag):
    return os.path.join(checkpoint_dir, f"{tag}_checkpoint.pt")


def done_marker(checkpoint_dir, tag):
    return os.path.join(checkpoint_dir, f"{tag}.done")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_f1):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_val_f1"]


# ----------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)
    return f1_macro, f1_micro, h_loss, y_pred


def compute_per_class_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    rows = []
    for k, cname in enumerate(class_names):
        rows.append({"class": cname, "precision": precision[k], "recall": recall[k],
                     "f1": f1[k], "support": int(support[k])})
    return rows


def build_model(system, n_classes):
    if system == "cmlc" or system == "br":
        # BR reusa CMLC_CRNN pero instanciado con n_classes=1 por cada clase (ver train_br)
        return CMLC_CRNN(n_classes=n_classes)
    elif system == "hps":
        return HPS_MTL_CRNN(n_classes=n_classes)
    elif system == "nddr":
        return NDDR_MTL_CRNN(n_classes=n_classes)
    else:
        raise ValueError(f"Sistema desconocido: {system}")


# ----------------------------------------------------------------------
# Entrenamiento de un fold (comun a cmlc / hps / nddr; para br se usa una
# variante con n_classes=1 llamada una vez por clase, ver train_br)
# ----------------------------------------------------------------------
def run_fold(tag, model, train_ds, val_ds, class_names_for_log,
             args, device, epochs_writer, epochs_file, start_time, fold_num):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    criterion = nn.BCELoss()

    ckpt_path = checkpoint_path(args.checkpoint_dir, tag)
    start_epoch = 1
    best_val_f1 = -1.0

    if args.resume and os.path.exists(ckpt_path):
        last_epoch, best_val_f1 = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        start_epoch = last_epoch + 1
        print(f"[{tag}] checkpoint encontrado: reanudando desde epoca {start_epoch}", flush=True)
    else:
        print(f"[{tag}] iniciando desde cero", flush=True)

    if start_epoch > args.epochs:
        print(f"[{tag}] ya completo ({start_epoch - 1}/{args.epochs} epocas). Saltando.", flush=True)
        return best_val_f1, model, val_ds

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        for feats, targets in train_loader:
            feats, targets = feats.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            probs = model(feats)
            loss = criterion(probs, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        val_loss_sum, n_val_batches = 0.0, 0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                probs = model(feats)
                loss = criterion(probs, targets)
                val_loss_sum += loss.item()
                n_val_batches += 1
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        val_loss = val_loss_sum / max(n_val_batches, 1)

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        f1_macro, f1_micro, h_loss, _ = compute_metrics(all_targets, all_probs, args.threshold)

        scheduler.step()
        epoch_time = time.time() - t0

        row = {
            "fold": fold_num, "epoch": epoch, "class": class_names_for_log,
            "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6),
            "val_f1_macro": round(f1_macro, 6), "val_f1_micro": round(f1_micro, 6),
            "val_hamming_loss": round(h_loss, 6),
            "lr": optimizer.param_groups[0]["lr"], "epoch_time_sec": round(epoch_time, 2),
        }
        epochs_writer.writerow(row)
        epochs_file.flush()

        print(f"[{tag}] epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"f1_macro={f1_macro:.4f} f1_micro={f1_micro:.4f} hamming={h_loss:.4f} | "
              f"{epoch_time:.1f}s/epoch", flush=True)

        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_f1)
            print(f"[{tag}] checkpoint guardado (epoca {epoch})", flush=True)

        if args.max_seconds is not None and (time.time() - start_time) > (args.max_seconds - 120):
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_f1)
            print(f"[{tag}] limite de tiempo cercano. Checkpoint guardado en epoca {epoch}. Terminando.", flush=True)
            sys.exit(0)

    open(done_marker(args.checkpoint_dir, tag), "w").close()
    return best_val_f1, model, val_ds


def evaluate_and_log_final(model, val_ds, class_names, args, device, fold_num, final_writer, final_file):
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for feats, targets in val_loader:
            feats = feats.to(device)
            probs = model(feats).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    _, _, _, y_pred = compute_metrics(all_targets, all_probs, args.threshold)
    rows = compute_per_class_metrics(all_targets, y_pred, class_names)
    for r in rows:
        r["fold"] = fold_num
        final_writer.writerow(r)
    final_file.flush()


# ----------------------------------------------------------------------
# Modos de entrenamiento
def make_datasets(args, filenames_subset, labels_subset, audio_dir, filename_to_row=None):
    """Construye el Dataset correcto segun si se esta usando el cache de
    features precomputadas (--cache_dir) o extraccion on-the-fly desde WAV."""
    if args.cache_dir is not None:
        cache_array_path = os.path.join(args.cache_dir, "train_features.npy")
        row_indices = np.array([filename_to_row[fn] for fn in filenames_subset])
        return CachedFoldDataset(cache_array_path, row_indices, labels_subset)
    return FoldAudioDataset(audio_dir, filenames_subset, labels_subset)


# ----------------------------------------------------------------------
def train_joint(args, df, selected_classes, device, epochs_writer, epochs_file, final_writer, final_file, start_time,
                 filename_to_row=None):
    """cmlc / hps / nddr: un solo modelo conjunto por fold, sobre TODAS las clases seleccionadas."""
    audio_dir = os.path.join(args.data_dir, args.train_subdir)
    filenames = df.iloc[:, 0].values
    labels = df[selected_classes].values.astype(np.float32)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_indices = list(kf.split(df))

    for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        tag = f"{args.system}_fold{fold}"
        if args.resume and os.path.exists(done_marker(args.checkpoint_dir, tag)):
            print(f"[{tag}] ya completo. Saltando.", flush=True)
            continue

        print(f"\n=== {args.system.upper()} FOLD {fold}/{args.folds} ({len(selected_classes)} clases) ===", flush=True)
        torch.manual_seed(args.seed + fold)
        model = build_model(args.system, n_classes=len(selected_classes)).to(device)

        train_ds = make_datasets(args, filenames[train_idx], labels[train_idx], audio_dir, filename_to_row)
        val_ds = make_datasets(args, filenames[val_idx], labels[val_idx], audio_dir, filename_to_row)

        best_f1, model, val_ds = run_fold(
            tag, model, train_ds, val_ds, "",
            args, device, epochs_writer, epochs_file, start_time, fold,
        )
        evaluate_and_log_final(model, val_ds, selected_classes, args, device, fold, final_writer, final_file)
        print(f"[{tag}] completo. best_val_f1_macro={best_f1:.4f}", flush=True)


def train_br(args, df, selected_classes, device, epochs_writer, epochs_file, final_writer, final_file, start_time,
              filename_to_row=None):
    """br: K modelos INDEPENDIENTES (uno por clase), cada uno entrenado por separado, x fold."""
    audio_dir = os.path.join(args.data_dir, args.train_subdir)
    filenames = df.iloc[:, 0].values

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_indices = list(kf.split(df))

    n_total = args.folds * len(selected_classes)
    done_count = 0

    for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        for class_name in selected_classes:
            tag = f"br_fold{fold}_{class_name}"
            done_count += 1
            if args.resume and os.path.exists(done_marker(args.checkpoint_dir, tag)):
                print(f"[{tag}] ({done_count}/{n_total}) ya completo. Saltando.", flush=True)
                continue

            print(f"\n=== BR FOLD {fold}/{args.folds} -- clase {class_name} ({done_count}/{n_total}) ===", flush=True)
            labels_single = df[[class_name]].values.astype(np.float32)

            torch.manual_seed(args.seed + fold)
            model = build_model("br", n_classes=1).to(device)

            train_ds = make_datasets(args, filenames[train_idx], labels_single[train_idx], audio_dir, filename_to_row)
            val_ds = make_datasets(args, filenames[val_idx], labels_single[val_idx], audio_dir, filename_to_row)

            best_f1, model, val_ds = run_fold(
                tag, model, train_ds, val_ds, class_name,
                args, device, epochs_writer, epochs_file, start_time, fold,
            )
            evaluate_and_log_final(model, val_ds, [class_name], args, device, fold, final_writer, final_file)
            print(f"[{tag}] completo. best_val_f1={best_f1:.4f}", flush=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sistema: {args.system} | Device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv_epochs)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv_final)) or ".", exist_ok=True)

    df = pd.read_csv(args.train_csv)
    all_class_names = df.columns[1:].tolist()
    assert len(all_class_names) == args.k_classes, (
        f"--k_classes={args.k_classes} no coincide con las columnas del CSV ({len(all_class_names)})."
    )
    assert args.folds >= 2, f"--folds={args.folds} invalido: K-Fold CV requiere al menos 2."

    selected_classes = select_classes(df, all_class_names, args.classes_subset, args.top_n_classes)

    if args.max_samples is not None and args.max_samples < len(df):
        n_before = len(df)
        df = subsample_preserving_positives(df, selected_classes, args.max_samples, args.seed)
        n_with_positive = int((df[selected_classes].sum(axis=1) > 0).sum())
        print(f"Dataset reducido con --max_samples: {len(df)} de {n_before} clips "
              f"({n_with_positive} con al menos un positivo de las clases seleccionadas).", flush=True)
    elif args.subsample_frac < 1.0:
        assert args.subsample_frac > 0.0, f"--subsample_frac={args.subsample_frac} invalido: debe ser > 0."
        n_before = len(df)
        df = df.sample(frac=args.subsample_frac, random_state=args.seed).reset_index(drop=True)
        print(f"Subsample aplicado: {len(df)} de {n_before} clips ({args.subsample_frac:.0%}).", flush=True)

    if args.output_used_data_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_used_data_csv)) or ".", exist_ok=True)
        used_cols = [df.columns[0]] + selected_classes
        df[used_cols].to_csv(args.output_used_data_csv, index=False)
        print(f"Subconjunto de datos usado guardado en: {args.output_used_data_csv} "
              f"({len(df)} filas, columnas: {used_cols})", flush=True)

    filename_to_row = None
    if args.cache_dir is not None:
        cache_filenames_csv = os.path.join(args.cache_dir, "train_filenames.csv")
        cache_array_path = os.path.join(args.cache_dir, "train_features.npy")
        assert os.path.exists(cache_filenames_csv) and os.path.exists(cache_array_path), (
            f"--cache_dir={args.cache_dir} no contiene train_features.npy / train_filenames.csv. "
            f"Correr precompute_features.py primero."
        )
        filename_to_row = build_filename_to_row(cache_filenames_csv)
        missing = [fn for fn in df.iloc[:, 0].tolist() if fn not in filename_to_row]
        assert not missing, (
            f"{len(missing)} archivos del CSV de entrenamiento no estan en el cache "
            f"(ej. {missing[:3]}). Volver a correr precompute_features.py sobre el CSV completo."
        )
        print(f"Usando cache de features precomputadas: {args.cache_dir}", flush=True)

    print(f"Dataset: {len(df)} clips. Clases totales: {len(all_class_names)}. "
          f"Clases usadas en esta corrida: {len(selected_classes)}.", flush=True)
    if len(selected_classes) < len(all_class_names):
        print(f"Subconjunto de clases: {selected_classes}", flush=True)

    if args.system in ("cmlc", "hps", "nddr") and args.system == "nddr" and len(selected_classes) > 15:
        print(f"ADVERTENCIA: NDDR MTL con {len(selected_classes)} clases puede ser muy lento/pesado "
              f"en memoria. Si el job se cuelga o hace OOM, reintentar con --top_n_classes 10-15.", flush=True)
    if args.system == "br":
        total_trainings = args.folds * len(selected_classes)
        print(f"ADVERTENCIA: BR entrenara {total_trainings} modelos independientes "
              f"({args.folds} folds x {len(selected_classes)} clases). Esto puede tardar "
              f"~{len(selected_classes)}x mas que CMLC en tiempo total.", flush=True)

    epochs_fieldnames = ["fold", "epoch", "class", "train_loss", "val_loss",
                         "val_f1_macro", "val_f1_micro", "val_hamming_loss", "lr", "epoch_time_sec"]
    final_fieldnames = ["fold", "class", "precision", "recall", "f1", "support"]

    epochs_file, epochs_writer = open_csv_writer(args.output_csv_epochs, epochs_fieldnames)
    final_file, final_writer = open_csv_writer(args.output_csv_final, final_fieldnames)

    start_time = time.time()

    if args.system == "br":
        train_br(args, df, selected_classes, device, epochs_writer, epochs_file, final_writer, final_file, start_time,
                 filename_to_row=filename_to_row)
    else:
        train_joint(args, df, selected_classes, device, epochs_writer, epochs_file, final_writer, final_file, start_time,
                    filename_to_row=filename_to_row)

    epochs_file.close()
    final_file.close()

    total_time = time.time() - start_time
    print(f"\nEntrenamiento {args.system.upper()} completo. Tiempo total: {total_time/3600:.2f} h", flush=True)
    print(f"Resultados por epoca: {args.output_csv_epochs}", flush=True)
    print(f"Resultados finales por clase/fold: {args.output_csv_final}", flush=True)


if __name__ == "__main__":
    main()
