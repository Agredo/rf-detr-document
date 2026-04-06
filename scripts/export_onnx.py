"""
scripts/export_onnx.py

Exportiert ein trainiertes RF-DETR-Seg-Nano-Checkpoint als ONNX (FP32).

Der ONNX-Export ist die Basis für alle weiteren Quantisierungsschritte.

ONNX-Ausgaben des Segmentierungsmodells:
    boxes   : [1, N, 4]   — Bounding Boxes (x1, y1, x2, y2), normalisiert 0..1
    scores  : [1, N]      — Konfidenzwerte
    labels  : [1, N]      — Klassenindex
    masks   : [1, N, H, W] — Segmentierungsmasken (binär, 0 oder 1)

Verwendung:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --run-name rf_detr_nano
    python scripts/export_onnx.py --checkpoint runs/rf_detr_nano/checkpoint_best_total.pth
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR-Seg-Nano → ONNX exportieren")
    parser.add_argument("--run-name",   default="rf_detr_nano")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Expliziter Checkpoint-Pfad (überschreibt --run-name)",
    )
    parser.add_argument("--img-size",   type=int, default=312)
    parser.add_argument("--device",     default="cpu",
                        help="cpu empfohlen für ONNX-Export (stabiler als MPS)")
    args = parser.parse_args()

    # Checkpoint bestimmen
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path("runs") / args.run_name / "checkpoint_best_total.pth"

    if not ckpt_path.exists():
        print(f"[FEHLER] Checkpoint nicht gefunden: {ckpt_path}", file=sys.stderr)
        print("Bitte zuerst trainieren: python scripts/train.py", file=sys.stderr)
        sys.exit(1)

    out_dir = ckpt_path.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_fp32.onnx"

    try:
        from rfdetr import RFDETRSegNano
    except ImportError:
        print('[FEHLER] rfdetr nicht installiert: pip install "rfdetr[train,onnx]"', file=sys.stderr)
        sys.exit(1)

    print(f"  Lade Checkpoint: {ckpt_path}")
    print(f"  Exportiere nach: {out_path}")

    model = RFDETRSegNano(
        pretrain_weights=str(ckpt_path),
        device=args.device,
        resolution=args.img_size,
    )

    model.export(output_dir=str(out_dir), filename="model_fp32.onnx")

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"\n✓ ONNX (FP32) exportiert: {out_path}  ({size_mb:.1f} MB)")
    else:
        # RF-DETR kann den Namen leicht variieren — suche ONNX-Datei
        onnx_files = list(out_dir.glob("*.onnx"))
        if onnx_files:
            f = onnx_files[0]
            f.rename(out_path)
            size_mb = out_path.stat().st_size / 1_048_576
            print(f"\n✓ ONNX (FP32) exportiert: {out_path}  ({size_mb:.1f} MB)")
        else:
            print("[FEHLER] Keine ONNX-Datei gefunden nach Export.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
