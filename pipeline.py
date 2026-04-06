"""
pipeline.py

End-to-End Pipeline:
  1. Datensatz aufbereiten   (scripts/prepare_dataset.py)
  2. Modell trainieren       (scripts/train.py)
  3. ONNX exportieren        (scripts/export_onnx.py)
  4. Quantisieren            (scripts/quantize_onnx.py)
  5. Evaluieren              (scripts/evaluate_onnx.py)
  6. HTML generieren         (scripts/generate_html.py)
  7. HTML im Browser öffnen

Jeder Schritt kann einzeln übersprungen werden (--skip-*).
Bereits erledigte Schritte werden automatisch erkannt und übersprungen,
wenn die erwarteten Ausgaben bereits vorhanden sind.

Verwendung:
    python pipeline.py
    python pipeline.py --run-name v1 --epochs 50 --batch 4
    python pipeline.py --skip-prepare --skip-train   # nur ab ONNX-Export

Optionale Argumente:
    --run-name          Name des Ausgabeordners            (Standard: rf_detr_nano)
    --roboflow-dir      Pfad zum Roboflow-Datensatz        (Standard: s.u.)
    --epochs            Trainingsepochen                   (Standard: 50)
    --batch             Batch-Größe                        (Standard: 4)
    --lr                Lernrate                           (Standard: 1e-4)
    --workers           DataLoader-Threads                 (Standard: 2)
    --img-size          Eingabegröße                       (Standard: 312)
    --val-split         Anteil Validierung                 (Standard: 0.2)
    --eval-samples      Beispielbilder für HTML            (Standard: 20)
    --skip-prepare      Datensatz-Vorbereitung überspringen
    --skip-train        Training überspringen
    --skip-export       ONNX-Export überspringen
    --skip-quantize     Quantisierung überspringen
    --skip-eval         Evaluierung überspringen
    --skip-html         HTML-Generierung überspringen
    --no-open           Browser nicht öffnen
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> bool:
    """Führt einen Pipeline-Schritt aus. Gibt True zurück bei Erfolg."""
    print(f"\n{'='*60}")
    print(f"  Schritt: {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n[FEHLER] Schritt '{name}' fehlgeschlagen (exit {result.returncode})")
        return False
    return True


def already_done(path: Path, label: str) -> bool:
    if path.exists():
        print(f"  [Übersprungen] {label} — bereits vorhanden: {path}")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR End-to-End Pipeline")
    parser.add_argument("--run-name",     default="rf_detr_nano")
    parser.add_argument("--roboflow-dir", default="/Users/agredo/Doument Detection/train")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch",        type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--workers",      type=int,   default=2)
    parser.add_argument("--img-size",     type=int,   default=312)
    parser.add_argument("--val-split",    type=float, default=0.2)
    parser.add_argument("--eval-samples", type=int,   default=20)
    parser.add_argument("--skip-prepare",  action="store_true")
    parser.add_argument("--skip-train",    action="store_true")
    parser.add_argument("--skip-export",   action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-eval",     action="store_true")
    parser.add_argument("--skip-html",     action="store_true")
    parser.add_argument("--no-open",       action="store_true")
    args = parser.parse_args()

    run_dir   = Path("runs") / args.run_name
    data_dir  = Path("data")
    html_path = run_dir / "comparison.html"

    print(f"\nRF-DETR-Seg-Nano Pipeline")
    print(f"  Run:     {args.run_name}")
    print(f"  Ausgabe: {run_dir.resolve()}")

    # ──────────────────────────────────────────────────────────────
    # 1. Datensatz aufbereiten
    # ──────────────────────────────────────────────────────────────
    if not args.skip_prepare:
        train_ann = data_dir / "train" / "_annotations.coco.json"
        if not already_done(train_ann, "Datensatz-Vorbereitung"):
            ok = run_step("Datensatz aufbereiten", [
                "scripts/prepare_dataset.py",
                "--roboflow-dir", args.roboflow_dir,
                "--val-split",    str(args.val_split),
            ])
            if not ok:
                sys.exit(1)
    else:
        print("\n  [Übersprungen] Datensatz-Vorbereitung (--skip-prepare)")

    # ──────────────────────────────────────────────────────────────
    # 2. Training
    # ──────────────────────────────────────────────────────────────
    if not args.skip_train:
        ckpt = run_dir / "checkpoint_best_total.pth"
        if not already_done(ckpt, "Training"):
            ok = run_step("Training", [
                "scripts/train.py",
                "--run-name",  args.run_name,
                "--epochs",    str(args.epochs),
                "--batch",     str(args.batch),
                "--lr",        str(args.lr),
                "--workers",   str(args.workers),
                "--img-size",  str(args.img_size),
            ])
            if not ok:
                sys.exit(1)
    else:
        print("\n  [Übersprungen] Training (--skip-train)")

    # ──────────────────────────────────────────────────────────────
    # 3. ONNX Export (FP32)
    # ──────────────────────────────────────────────────────────────
    if not args.skip_export:
        fp32_path = run_dir / "models" / "model_fp32.onnx"
        if not already_done(fp32_path, "ONNX Export"):
            ok = run_step("ONNX Export (FP32)", [
                "scripts/export_onnx.py",
                "--run-name",  args.run_name,
                "--img-size",  str(args.img_size),
            ])
            if not ok:
                sys.exit(1)
    else:
        print("\n  [Übersprungen] ONNX Export (--skip-export)")

    # ──────────────────────────────────────────────────────────────
    # 4. Quantisierung (FP16, INT8, INT4)
    # ──────────────────────────────────────────────────────────────
    if not args.skip_quantize:
        int4_path = run_dir / "models" / "model_int4.onnx"
        if not already_done(int4_path, "Quantisierung"):
            ok = run_step("Quantisierung (FP16 / INT8 / INT4)", [
                "scripts/quantize_onnx.py",
                "--run-name", args.run_name,
            ])
            if not ok:
                sys.exit(1)
    else:
        print("\n  [Übersprungen] Quantisierung (--skip-quantize)")

    # ──────────────────────────────────────────────────────────────
    # 5. Evaluierung
    # ──────────────────────────────────────────────────────────────
    if not args.skip_eval:
        results_path = run_dir / "models" / "results.json"
        if not already_done(results_path, "Evaluierung"):
            ok = run_step("Evaluierung aller ONNX-Modelle", [
                "scripts/evaluate_onnx.py",
                "--run-name", args.run_name,
                "--samples",  str(args.eval_samples),
            ])
            if not ok:
                sys.exit(1)
    else:
        print("\n  [Übersprungen] Evaluierung (--skip-eval)")

    # ──────────────────────────────────────────────────────────────
    # 6. HTML generieren
    # ──────────────────────────────────────────────────────────────
    if not args.skip_html:
        ok = run_step("HTML Vergleich generieren", [
            "scripts/generate_html.py",
            "--run-name", args.run_name,
        ])
        if not ok:
            sys.exit(1)
    else:
        print("\n  [Übersprungen] HTML (--skip-html)")

    # ──────────────────────────────────────────────────────────────
    # Fertig
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline abgeschlossen!")
    print(f"  HTML: {html_path.resolve()}")
    print(f"{'='*60}\n")

    if not args.no_open and html_path.exists():
        webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
