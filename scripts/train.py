"""
scripts/train.py

Fine-tuning von RF-DETR-Seg-Nano auf dem Dokument-Datensatz.

Modell : RFDETRSegNano (Apache 2.0, ~33MB, 312×312 Eingabe)
Aufgabe: Instance Segmentation — gibt BBox + Segmentierungsmaske zurück

Voraussetzung:
    python scripts/prepare_dataset.py   ← einmalig ausführen

Verwendung:
    python scripts/train.py
    python scripts/train.py --run-name v1 --epochs 50 --batch 4

Optionale Argumente:
    --dataset-dir   Pfad zum aufbereiteten Datensatz  (Standard: data/)
    --run-name      Name des Ausgabeordners            (Standard: rf_detr_nano)
    --epochs        Trainingsepochen                   (Standard: 50)
    --batch         Batch-Größe                        (Standard: 4)
    --lr            Lernrate                           (Standard: 1e-4)
    --workers       DataLoader-Threads                 (Standard: 2)
    --device        cpu | cuda | mps | auto            (Standard: auto)
    --img-size      Eingabegröße (muss durch 32 teilbar sein) (Standard: 312)
"""

import argparse
import datetime
import io
import json
import re
import sys
import threading
import time
from pathlib import Path


class EpochTimer:
    """
    Interceptiert sys.stdout, erkennt abgeschlossene Epochen in der PTL-Ausgabe
    und gibt nach jeder Epoche eine aktualisierte Zeitschätzung aus.

    PTL schreibt typisch: "Epoch 5/49: 100%|..."
    Wir erkennen den Abschluss wenn diese Zeile mit 100% endet.
    """

    _EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+):.*100%")

    def __init__(self, total_epochs: int):
        self._total = total_epochs
        self._start = time.monotonic()
        self._epoch_times: list[float] = []
        self._last_epoch = 0
        self._orig_stdout = sys.stdout
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        self._orig_stdout.write(text)
        self._orig_stdout.flush()
        with self._lock:
            self._buf += text
            # Suche nach abgeschlossenen Epochen in gepuffertem Text
            for line in self._buf.splitlines():
                m = self._EPOCH_RE.search(line)
                if m:
                    epoch = int(m.group(1)) + 1  # PTL zählt ab 0
                    if epoch > self._last_epoch:
                        self._last_epoch = epoch
                        self._record_epoch(epoch)
            # Nur die letzte unvollständige Zeile im Puffer behalten
            if "\n" in self._buf:
                self._buf = self._buf.rsplit("\n", 1)[-1]
        return len(text)

    def flush(self) -> None:
        self._orig_stdout.flush()

    def _record_epoch(self, epoch: int) -> None:
        elapsed = time.monotonic() - self._start
        self._epoch_times.append(elapsed / epoch)
        avg = sum(self._epoch_times[-5:]) / len(self._epoch_times[-5:])
        remaining_secs = avg * (self._total - epoch)
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_secs)
        rem_min = int(remaining_secs // 60)
        rem_sec = int(remaining_secs % 60)
        self._orig_stdout.write(
            f"\n  ⏱  Epoche {epoch}/{self._total} fertig"
            f"  |  ~{avg/60:.1f} min/Epoche"
            f"  |  noch ~{rem_min}m {rem_sec:02d}s"
            f"  |  Fertig ca. {eta:%H:%M:%S}\n\n"
        )
        self._orig_stdout.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig_stdout


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR-Seg-Nano Fine-Tuning")
    parser.add_argument("--dataset-dir", default="data")
    parser.add_argument("--run-name",    default="rf_detr_nano")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch",       type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--workers",     type=int,   default=2)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--img-size",    type=int,   default=312)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        print(
            "\n[FEHLER] Datensatz nicht gefunden."
            "\nBitte zuerst ausführen:"
            "\n    python scripts/prepare_dataset.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    device = detect_device() if args.device == "auto" else args.device
    print(f"  Gerät: {device}")

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rfdetr import RFDETRSegNano
    except ImportError:
        print(
            "\n[FEHLER] rfdetr nicht installiert."
            "\nBitte ausführen:"
            '\n    pip install "rfdetr[train,onnx]"\n',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n  Starte Training: {args.epochs} Epochen")
    print(f"  Batch-Größe:     {args.batch}")
    print(f"  Lernrate:        {args.lr}")
    print(f"  Img-Größe:       {args.img_size}×{args.img_size}")
    print(f"  Ausgabe:         {run_dir.resolve()}\n")

    model = RFDETRSegNano(
        device=device,
        resolution=args.img_size,
    )

    with EpochTimer(args.epochs):
        model.train(
            dataset_dir=str(dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            num_workers=args.workers,
            output_dir=str(run_dir),
        )

    # Trainingsparameter speichern
    params = {
        "run_name":    args.run_name,
        "date":        datetime.datetime.now().isoformat(timespec="seconds"),
        "model":       "RFDETRSegNano",
        "task":        "instance_segmentation",
        "epochs":      args.epochs,
        "batch_size":  args.batch,
        "lr":          args.lr,
        "img_size":    args.img_size,
        "workers":     args.workers,
        "device":      device,
        "dataset_dir": str(dataset_dir.resolve()),
    }
    params_path = run_dir / "training_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n  Parameter gespeichert: {params_path}")
    print(f"  Checkpoint:             {run_dir}/checkpoint_best_total.pth")


if __name__ == "__main__":
    main()
