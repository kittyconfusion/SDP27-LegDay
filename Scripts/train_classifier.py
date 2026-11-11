#!/usr/bin/env python3
"""Train the LegDay motion classifier on a Mac and export it for the ESP32-S3.

Typical use, after copying the SD card's category folders to the Mac:

    python3 Scripts/train_classifier.py train /Volumes/SDCARD runs/first --width 32
    python3 Scripts/train_classifier.py train /Volumes/SDCARD runs/attn --width 32 --attention

Before real data exists, make a synthetic dataset to exercise everything:

    python3 Scripts/train_classifier.py synth runs/synthetic-data
    python3 Scripts/train_classifier.py train runs/synthetic-data runs/synthetic --install-demo

``--install-demo`` copies the trained model into the Rust crate
(``crates/legday-classifier/models/demo.ldm``) together with a test window, so
the next firmware build runs it on the board and ``cargo test`` checks the
Rust inference against PyTorch.

``bench-models`` writes untrained models of increasing size for the
on-board timing benchmark (diagnostics built with ``--features model-bench``),
each with a golden window so the Rust inference can be checked against
PyTorch for every layer type, attention included.

What ``train`` produces in the output folder:

* ``model.ldm``         the model, in the format the S3 reads
* ``report.txt``        sizes, accuracy, per-class precision / recall
* ``confusion.png``     which classes get mistaken for which
* ``embedding.png``     validation windows plotted by the model's internal
                        features (t-SNE): separated clusters = separable classes
* ``training.png``      loss and accuracy per epoch
* ``golden.bin``        one window and its expected outputs, for the Rust test
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import legday_ml as ml

# Matplotlib wants a writable config folder; keep it out of the home directory.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
# scikit-learn's worker pool cannot count cores in some sandboxes; tell it.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

REPO = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO / "crates" / "legday-classifier" / "models"


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------


def command_synth(args) -> None:
    rng = np.random.default_rng(args.seed)
    for category in ml.CATEGORIES:
        folder = args.output / category
        folder.mkdir(parents=True, exist_ok=True)
        for session in range(args.files):
            frames = ml.synthetic_recording(category, args.seconds, rng)
            path = folder / f"S{session:07d}.Q6B"
            frames.tofile(path)
    print(f"wrote {len(ml.CATEGORIES)} categories x {args.files} files x {args.seconds:.0f} s to {args.output}")


# ---------------------------------------------------------------------------
# train: data
# ---------------------------------------------------------------------------


def load_dataset(root: Path, hop: int, val_fraction: float):
    """Window every recording and split each one by time into train / val.

    The last ``val_fraction`` of every file is held out, with a gap of one
    window between the parts. Splitting by time rather than picking random
    windows matters: neighbouring windows overlap almost completely, so a
    random split would put near-copies of training data into validation and
    report an accuracy the model has not earned.
    """
    recordings = ml.find_recordings(root)
    if not recordings:
        raise SystemExit(f"no .Q6B/.BIN files under {root}/<CATEGORY>/")

    present = {category for category, _ in recordings}
    class_names = [c for c in ml.CATEGORIES if c in present] + sorted(present - set(ml.CATEGORIES))
    gap = -(-ml.WINDOW_FRAMES // hop)  # windows that would overlap the split point

    split = {"train": ([], []), "val": ([], [])}
    val_sources = []  # (frames, start) of each validation window, for the golden file

    for category, path in recordings:
        frames = ml.load_recording(path)
        if len(frames) < ml.WINDOW_FRAMES:
            print(f"  skipping {path} ({len(frames)} frames, shorter than one window)")
            continue

        inputs = ml.windows(ml.frame_channels(frames), hop)
        label = class_names.index(category)
        boundary = int(len(inputs) * (1 - val_fraction))

        split["train"][0].append(inputs[: max(0, boundary - gap)])
        split["train"][1].append(np.full(max(0, boundary - gap), label))
        split["val"][0].append(inputs[boundary:])
        split["val"][1].append(np.full(len(inputs) - boundary, label))
        val_sources.extend((frames, index * hop) for index in range(boundary, len(inputs)))

    def stack(part):
        x, y = split[part]
        return np.concatenate(x).astype(np.float32), np.concatenate(y).astype(np.int64)

    return class_names, stack("train"), stack("val"), val_sources, len(recordings)


# ---------------------------------------------------------------------------
# train: model
# ---------------------------------------------------------------------------


def train_model(args, class_names, train, val):
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    (x_train, y_train), (x_val, y_val) = train, val

    # Standardize every channel with statistics from the training set only.
    # The same mean/std travel inside the model file to the S3.
    mean = x_train.mean(axis=(0, 2)).astype(np.float32)
    std = np.maximum(x_train.std(axis=(0, 2)), 1e-3).astype(np.float32)

    def normalize(x):
        return (x - mean[None, :, None]) / std[None, :, None]

    x_train_n = torch.from_numpy(normalize(x_train))
    x_val_n = torch.from_numpy(normalize(x_val))
    y_train_t = torch.from_numpy(y_train)
    y_val_t = torch.from_numpy(y_val)

    model = ml.build_model(len(class_names), args.width, args.attention)

    # Weight rarer classes more, so a dataset with lots of walking does not
    # teach the model to always answer "walk".
    counts = np.bincount(y_train, minlength=len(class_names)).astype(np.float32)
    class_weights = torch.from_numpy(counts.sum() / np.maximum(counts, 1) / len(class_names))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = (float("inf"), None, 0)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(x_train_n))

        for start in range(0, len(order), args.batch):
            batch = order[start : start + args.batch]
            x = x_train_n[batch]

            # Augmentation: small noise and a random gain per window, so the
            # model does not memorise exact amplitudes of one recording.
            x = x * torch.empty(len(batch), 1, 1).uniform_(0.9, 1.1) + 0.05 * torch.randn_like(x)

            optimizer.zero_grad()
            loss = loss_fn(model(x), y_train_t[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x_train_n)
            val_logits = model(x_val_n)
            record = {
                "epoch": epoch + 1,
                "train_loss": loss_fn(train_logits, y_train_t).item(),
                "val_loss": loss_fn(val_logits, y_val_t).item(),
                "train_acc": (train_logits.argmax(1) == y_train_t).float().mean().item(),
                "val_acc": (val_logits.argmax(1) == y_val_t).float().mean().item(),
            }
        history.append(record)

        # Keep the epoch that did best on validation (early stopping).
        if record["val_loss"] < best[0]:
            best = (record["val_loss"], {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1:>3}: train loss {record['train_loss']:.3f} acc {record['train_acc']:.3f} | "
                f"val loss {record['val_loss']:.3f} acc {record['val_acc']:.3f}"
            )

    model.load_state_dict(best[1])
    model.eval()
    print(f"  kept epoch {best[2]} (lowest validation loss)")
    return model, mean, std, normalize, history


# ---------------------------------------------------------------------------
# train: report
# ---------------------------------------------------------------------------


def write_report(args, model, class_names, normalize, train, val, history, file_count):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from sklearn.manifold import TSNE
    from sklearn.metrics import classification_report, confusion_matrix

    (x_val, y_val) = val
    with torch.no_grad():
        x = torch.from_numpy(normalize(x_val))
        predicted = model(x).argmax(1).numpy()
        embedding = model.embed(x).numpy()

    labels = list(range(len(class_names)))
    report = classification_report(y_val, predicted, labels=labels, target_names=class_names, digits=3, zero_division=0)
    matrix = confusion_matrix(y_val, predicted, labels=labels)
    accuracy = float((predicted == y_val).mean())

    params = ml.count_parameters(model)
    macs = ml.count_macs(model)
    summary = [
        f"data:        {args.data} ({file_count} files)",
        f"windows:     {len(train[1])} train, {len(y_val)} validation (hop {args.hop} frames)",
        f"model:       width {args.width}{' + self-attention' if args.attention else ''}",
        f"parameters:  {params:,} ({params * 4 / 1024:.0f} kB as float32)",
        f"compute:     {macs:,} multiply-accumulates per inference",
        f"validation:  {accuracy:.1%} accuracy",
        "",
        report,
        "confusion matrix (rows = true class, columns = predicted):",
        "          " + " ".join(f"{name:>7}" for name in class_names),
    ]
    summary += [f"{name:>8}  " + " ".join(f"{v:>7}" for v in row) for name, row in zip(class_names, matrix)]
    (args.output / "report.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))

    # Confusion matrix as an image.
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(labels, class_names, rotation=45)
    axis.set_yticks(labels, class_names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    for row in labels:
        for column in labels:
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=8)
    axis.set_title(f"validation confusion matrix ({accuracy:.1%})")
    figure.tight_layout()
    figure.savefig(args.output / "confusion.png", dpi=120)

    # Embedding: each validation window as a point, placed by the model's
    # internal features. Tight, separate clusters are the visual evidence
    # that the classes are separable in what the model learned.
    if len(embedding) > 5:
        # float64 plus errstate: NumPy 2 on macOS raises spurious matmul
        # overflow warnings inside t-SNE's PCA start-up; the result is fine.
        with np.errstate(all="ignore"):
            points = TSNE(
                n_components=2, perplexity=min(30, len(embedding) - 1), random_state=0
            ).fit_transform(embedding.astype(np.float64))
        figure, axis = plt.subplots(figsize=(7, 6))
        for label, name in enumerate(class_names):
            chosen = y_val == label
            axis.scatter(points[chosen, 0], points[chosen, 1], s=8, label=name)
        axis.legend(markerscale=2)
        axis.set_title("validation windows in the model's embedding (t-SNE)")
        figure.tight_layout()
        figure.savefig(args.output / "embedding.png", dpi=120)

    # Training curves.
    epochs = [h["epoch"] for h in history]
    figure, (loss_axis, acc_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, [h["train_loss"] for h in history], label="train")
    loss_axis.plot(epochs, [h["val_loss"] for h in history], label="validation")
    loss_axis.set_title("loss")
    acc_axis.plot(epochs, [h["train_acc"] for h in history], label="train")
    acc_axis.plot(epochs, [h["val_acc"] for h in history], label="validation")
    acc_axis.set_title("accuracy")
    for axis in (loss_axis, acc_axis):
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output / "training.png", dpi=120)


def command_train(args) -> None:
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    class_names, train, val, val_sources, file_count = load_dataset(args.data, args.hop, args.val_fraction)
    print(f"classes: {class_names}; {len(train[1])} train / {len(val[1])} validation windows")
    if len(val[1]) == 0:
        raise SystemExit("no validation windows; record longer files or lower --val-fraction")

    model, mean, std, normalize, history = train_model(args, class_names, train, val)
    write_report(args, model, class_names, normalize, train, val, history, file_count)

    model_path = args.output / "model.ldm"
    ml.export_model(model_path, model, mean, std, class_names)

    # One validation window, start to finish, for the Rust parity test.
    frames, start = val_sources[0]
    window = frames[start : start + ml.WINDOW_FRAMES]
    with torch.no_grad():
        x = torch.from_numpy(normalize(ml.windows(ml.frame_channels(window), ml.WINDOW_FRAMES)))
        logits = model(x)[0].numpy()
    golden_path = args.output / "golden.bin"
    ml.export_golden(golden_path, window, logits)

    print(f"\nwrote {model_path} ({model_path.stat().st_size:,} bytes) and the plots in {args.output}")

    if args.install_demo:
        shutil.copyfile(model_path, DEMO_DIR / "demo.ldm")
        shutil.copyfile(golden_path, DEMO_DIR / "demo_golden.bin")
        print(f"installed as the firmware's demo model in {DEMO_DIR}")


# ---------------------------------------------------------------------------
# bench-models
# ---------------------------------------------------------------------------

BENCH_MODELS = (("tiny", 16, False), ("small", 32, False), ("small_attn", 32, True), ("medium", 80, False), ("large", 190, False))


def command_bench_models(args) -> None:
    """Untrained models of increasing size, only for timing on the S3."""
    import torch

    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    mean = np.zeros(ml.CHANNELS, dtype=np.float32)
    std = np.ones(ml.CHANNELS, dtype=np.float32)

    # One synthetic second of walking, replayed through every model.
    window = ml.synthetic_recording("WALK", 2.0, np.random.default_rng(0))[: ml.WINDOW_FRAMES]
    x = torch.from_numpy(ml.windows(ml.frame_channels(window), ml.WINDOW_FRAMES))

    for name, width, attention in BENCH_MODELS:
        model = ml.build_model(len(ml.CATEGORIES), width, attention).eval()
        path = args.output / f"{name}.ldm"
        ml.export_model(path, model, mean, std, list(ml.CATEGORIES))
        with torch.no_grad():
            ml.export_golden(args.output / f"{name}_golden.bin", window, model(x)[0].numpy())
        print(
            f"{name:<11} width {width:>3}{' +attn' if attention else '      '} "
            f"{ml.count_parameters(model):>9,} params {ml.count_macs(model):>12,} MACs  {path.stat().st_size:>10,} bytes"
        )


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    synth = commands.add_parser("synth", help="write a synthetic dataset")
    synth.add_argument("output", type=Path)
    synth.add_argument("--files", type=int, default=3, help="recordings per category")
    synth.add_argument("--seconds", type=float, default=40.0, help="length of each recording")
    synth.add_argument("--seed", type=int, default=1)

    train = commands.add_parser("train", help="train, evaluate, and export a model")
    train.add_argument("data", type=Path, help="folder containing <CATEGORY>/ subfolders of recordings")
    train.add_argument("output", type=Path)
    train.add_argument("--width", type=int, default=32, help="model size: 16, 32, 80, 190 ...")
    train.add_argument("--attention", action="store_true", help="add a self-attention layer")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hop", type=int, default=56, help="frames between windows (56 = 0.25 s)")
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--install-demo", action="store_true", help="copy the model into the firmware")

    bench = commands.add_parser("bench-models", help="untrained models for the S3 timing benchmark")
    bench.add_argument("--output", type=Path, default=DEMO_DIR / "bench")

    args = parser.parse_args()
    {"synth": command_synth, "train": command_train, "bench-models": command_bench_models}[args.command](args)


if __name__ == "__main__":
    main()
