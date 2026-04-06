"""
scripts/quantize_onnx.py

Erstellt FP16, INT8 und INT4 Varianten aus dem FP32 ONNX-Modell.

Methoden:
    FP16  : Alle float32-Gewichte → float16           (onnxconverter-common)
    INT8  : Static Quantization mit Kalibrierungsbildern (onnxruntime)
    INT4  : Weight-only INT4 (MatMulNBits)             (onnxruntime)

Ausgabe (im models/-Ordner des Runs):
    model_fp32.onnx   ← Eingabe (muss bereits existieren)
    model_fp16.onnx
    model_int8.onnx
    model_int4.onnx

Verwendung:
    python scripts/quantize_onnx.py
    python scripts/quantize_onnx.py --run-name rf_detr_nano
    python scripts/quantize_onnx.py --fp16 --int8 --int4   # alle drei
"""

import argparse
import shutil
import sys
from pathlib import Path


def export_fp16(fp32_path: Path, out_path: Path) -> None:
    """Konvertiert FP32 ONNX → FP16."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("[FEHLER] onnxconverter-common fehlt: pip install onnxconverter-common", file=sys.stderr)
        sys.exit(1)

    print(f"  FP16: Konvertiere {fp32_path.name} → {out_path.name}")
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(out_path))
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"       ✓ {out_path.name}  ({size_mb:.1f} MB)")


def export_int8(fp32_path: Path, out_path: Path, calibration_images: list[Path]) -> None:
    """
    INT8 Static Quantization mit Kalibrierungsbildern.
    Nutzt onnxruntime.quantization mit einem einfachen Bild-Datenreader.
    """
    try:
        import numpy as np
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from PIL import Image
    except ImportError:
        print("[FEHLER] onnxruntime fehlt: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    import onnxruntime as ort

    # Eingabegröße aus dem Modell bestimmen
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    _, _, H, W = input_info.shape   # [batch, channel, H, W]
    input_name = input_info.name

    class ImageReader(CalibrationDataReader):
        def __init__(self, images: list[Path], h: int, w: int, name: str):
            self.images = images[:100]   # max. 100 Kalibrierungsbilder
            self.idx = 0
            self.h, self.w = h, w
            self.name = name

        def get_next(self):
            if self.idx >= len(self.images):
                return None
            img = Image.open(self.images[self.idx]).convert("RGB").resize((self.w, self.h))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)[np.newaxis]     # [1, 3, H, W]
            self.idx += 1
            return {self.name: arr}

    print(f"  INT8: Kalibriere mit {min(len(calibration_images), 100)} Bildern → {out_path.name}")
    reader = ImageReader(calibration_images, H, W, input_name)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"       ✓ {out_path.name}  ({size_mb:.1f} MB)")


def export_int4(fp32_path: Path, out_path: Path) -> None:
    """
    INT4 Weight-only Quantization (MatMulNBits).
    Konvertiert nur Gewichtsmatrizen (keine Aktivierungen) → sicher für Transformer.
    """
    try:
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    except ImportError:
        print("[FEHLER] onnxruntime fehlt oder zu alt: pip install onnxruntime>=1.18.0", file=sys.stderr)
        sys.exit(1)

    print(f"  INT4: Weight-only Quantization → {out_path.name}")
    model = onnx.load(str(fp32_path))
    quantizer = MatMulNBitsQuantizer(
        model=model,
        block_size=32,
        is_symmetric=True,
        bits=4,
    )
    quantizer.process()
    onnx.save(quantizer.model.model, str(out_path))
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"       ✓ {out_path.name}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX FP32 → FP16/INT8/INT4 quantisieren")
    parser.add_argument("--run-name",  default="rf_detr_nano")
    parser.add_argument("--fp32-path", default=None, help="Expliziter Pfad zur FP32 ONNX-Datei")
    parser.add_argument("--fp16",  action="store_true", default=True)
    parser.add_argument("--int8",  action="store_true", default=True)
    parser.add_argument("--int4",  action="store_true", default=True)
    parser.add_argument(
        "--calib-dir",
        default=None,
        help="Kalibrierungsbilder für INT8 (Standard: data/valid/)",
    )
    args = parser.parse_args()

    # FP32 ONNX-Pfad
    if args.fp32_path:
        fp32_path = Path(args.fp32_path)
    else:
        fp32_path = Path("runs") / args.run_name / "models" / "model_fp32.onnx"

    if not fp32_path.exists():
        print(f"[FEHLER] FP32 ONNX nicht gefunden: {fp32_path}", file=sys.stderr)
        print("Bitte zuerst exportieren: python scripts/export_onnx.py", file=sys.stderr)
        sys.exit(1)

    models_dir = fp32_path.parent
    print(f"\nQuantisierung von: {fp32_path}  ({fp32_path.stat().st_size / 1_048_576:.1f} MB)\n")

    # FP16
    if args.fp16:
        export_fp16(fp32_path, models_dir / "model_fp16.onnx")

    # INT8 — Kalibrierungsbilder sammeln
    if args.int8:
        calib_dir = Path(args.calib_dir) if args.calib_dir else Path("data/valid")
        calib_imgs = sorted(calib_dir.glob("*.jpg")) + sorted(calib_dir.glob("*.png"))
        if not calib_imgs:
            print(f"  [WARNUNG] Keine Kalibrierungsbilder in {calib_dir} gefunden, überspringe INT8.")
        else:
            export_int8(fp32_path, models_dir / "model_int8.onnx", calib_imgs)

    # INT4
    if args.int4:
        export_int4(fp32_path, models_dir / "model_int4.onnx")

    print(f"\n✓ Alle Modelle gespeichert in: {models_dir.resolve()}")
    for f in sorted(models_dir.glob("model_*.onnx")):
        print(f"   {f.name:<22}  {f.stat().st_size / 1_048_576:6.1f} MB")


if __name__ == "__main__":
    main()
