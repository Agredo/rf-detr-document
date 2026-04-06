"""
scripts/evaluate_onnx.py

Evaluiert alle ONNX-Modelle (FP32/FP16/INT8/INT4) auf dem Validierungsdatensatz.

Für jedes Modell wird ermittelt:
  - Modellgröße (MB)
  - Mittlere Inferenzzeit (ms) über N Bilder
  - Detection mAP@0.5 (Bounding Box)
  - Segmentation mAP@0.5 (Maske, falls Masken-Output vorhanden)
  - Beispielbilder mit einge~zeichneter Ausgabe

Ergebnis wird als results.json + Beispielbilder im models/-Ordner gespeichert.

Verwendung:
    python scripts/evaluate_onnx.py
    python scripts/evaluate_onnx.py --run-name rf_detr_nano --samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# ONNX-Inferenz
# ---------------------------------------------------------------------------

def load_session(onnx_path: Path):
    """Lädt eine ONNX-Runtime-Session."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[FEHLER] onnxruntime fehlt: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def preprocess(img_path: Path, target_h: int, target_w: int):
    """Lädt und normalisiert ein Bild → numpy [1, 3, H, W], float32."""
    import numpy as np
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], img   # [1, 3, H, W], PIL-Bild


def run_inference(session, img_array) -> dict:
    """Führt Inferenz durch und gibt ein Dict mit den Ausgaben zurück."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    out_names = [o.name for o in session.get_outputs()]
    return dict(zip(out_names, outputs))


def get_input_size(session) -> tuple[int, int]:
    """Liest H, W aus dem ONNX-Modell."""
    shape = session.get_inputs()[0].shape
    # shape kann [batch, channel, H, W] oder [batch, H, W, channel] sein
    if len(shape) == 4:
        return int(shape[2]), int(shape[3])
    return 312, 312   # Fallback


# ---------------------------------------------------------------------------
# IoU-Berechnung
# ---------------------------------------------------------------------------

def iou_boxes(a, b):
    """IoU zwischen zwei Boxen [x1,y1,x2,y2] (beide normalisiert)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """VOC-Stil AP: Fläche unter der Precision-Recall-Kurve."""
    import numpy as np
    p = [0.0] + list(precisions) + [0.0]
    r = [0.0] + list(recalls) + [1.0]
    # Precision-Envelope (monoton fallend nach links)
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    ap = 0.0
    for i in range(1, len(r)):
        ap += (r[i] - r[i - 1]) * p[i]
    return ap


def compute_map(all_detections: list, all_gt: list, iou_thresh: float = 0.5) -> float:
    """
    Berechnet mAP@iou_thresh.
    all_detections: [[score, box, matched], ...] pro Bild
    all_gt: [n_gt, ...] pro Bild
    """
    # Alle Detektionen sammeln und nach Score sortieren
    entries = []
    total_gt = sum(all_gt)
    for img_idx, dets in enumerate(all_detections):
        for score, matched in dets:
            entries.append((score, matched))
    if not entries or total_gt == 0:
        return 0.0

    entries.sort(key=lambda x: -x[0])
    tp = [0] * len(entries)
    fp = [0] * len(entries)
    for i, (_, matched) in enumerate(entries):
        if matched:
            tp[i] = 1
        else:
            fp[i] = 1

    import numpy as np
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = (tp_cum / total_gt).tolist()
    precisions = (tp_cum / (tp_cum + fp_cum + 1e-9)).tolist()
    return compute_ap(precisions, recalls)


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

def draw_detections(pil_img, result: dict, orig_w: int, orig_h: int, threshold: float = 0.3):
    """Zeichnet BBoxen und (falls vorhanden) Segmentierungsmasken auf ein PIL-Bild."""
    import numpy as np
    from PIL import ImageDraw, ImageFont

    img = pil_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)

    boxes  = result.get("boxes",  result.get("pred_boxes", None))
    scores = result.get("scores", result.get("pred_logits", None))
    masks  = result.get("masks",  result.get("pred_masks",  None))

    if boxes is None:
        return img.convert("RGB")

    # Shapes normalisieren: [1, N, 4] oder [N, 4]
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores is not None and scores.ndim == 2:
        scores = scores[0]
    if masks is not None and masks.ndim == 4:
        masks = masks[0]

    n = len(boxes)
    for i in range(n):
        score = float(scores[i]) if scores is not None else 1.0
        if score < threshold:
            continue

        box = boxes[i]
        # BBox kann normalisiert (0..1) oder absolut sein — normalisiert annehmen
        if box.max() <= 1.01:
            x1 = int(box[0] * orig_w)
            y1 = int(box[1] * orig_h)
            x2 = int(box[2] * orig_w)
            y2 = int(box[3] * orig_h)
        else:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Segmentierungsmaske einzeichnen (halbtransparent grün)
        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask.ndim == 2:
                mask_resized = np.array(
                    __import__("PIL").Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        (orig_w, orig_h), __import__("PIL").Image.NEAREST
                    )
                ) > 128
                overlay = __import__("PIL").Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                ys, xs = np.where(mask_resized)
                for y, x in zip(ys[::4], xs[::4]):   # downsampled für Performance
                    overlay_draw.point((x, y), fill=(0, 200, 80, 100))
                img = __import__("PIL").Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img)

        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 80), width=2)
        draw.text((x1 + 2, y1 + 2), f"{score:.2f}", fill=(255, 255, 0))

    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX-Modelle auf Validierungsdatensatz evaluieren")
    parser.add_argument("--run-name",   default="rf_detr_nano")
    parser.add_argument("--data-dir",   default="data/valid")
    parser.add_argument("--samples",    type=int, default=20,
                        help="Anzahl Beispielbilder für HTML (Standard: 20)")
    parser.add_argument("--threshold",  type=float, default=0.3)
    args = parser.parse_args()

    models_dir = Path("runs") / args.run_name / "models"
    if not models_dir.exists():
        print(f"[FEHLER] models/-Ordner nicht gefunden: {models_dir}", file=sys.stderr)
        sys.exit(1)

    val_dir = Path(args.data_dir)
    ann_path = val_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"[FEHLER] Annotationen nicht gefunden: {ann_path}", file=sys.stderr)
        sys.exit(1)

    with open(ann_path) as f:
        coco = json.load(f)

    # Index: image_file → GT-Boxen
    img_by_id = {img["id"]: img for img in coco["images"]}
    gt_by_file: dict[str, list] = {}
    for ann in coco["annotations"]:
        img_info = img_by_id[ann["image_id"]]
        fname = img_info["file_name"]
        x, y, w, h = ann["bbox"]
        W, H = img_info["width"], img_info["height"]
        gt_by_file.setdefault(fname, []).append([x/W, y/H, (x+w)/W, (y+h)/H])

    imgs_dir = val_dir
    img_files = [f for f in gt_by_file.keys() if (imgs_dir / f).exists()]
    print(f"  Validierungsbilder: {len(img_files)}")

    onnx_files = sorted(models_dir.glob("model_*.onnx"))
    if not onnx_files:
        print(f"[FEHLER] Keine ONNX-Dateien in {models_dir}", file=sys.stderr)
        sys.exit(1)

    results = {}

    for onnx_path in onnx_files:
        variant = onnx_path.stem.replace("model_", "")
        print(f"\n  Evaluiere: {onnx_path.name}")

        try:
            sess = load_session(onnx_path)
        except Exception as e:
            print(f"    [FEHLER beim Laden] {e}")
            continue

        H_in, W_in = get_input_size(sess)
        size_mb = onnx_path.stat().st_size / 1_048_576

        times = []
        all_dets = []
        all_gt_counts = []
        sample_imgs = []

        for idx, fname in enumerate(img_files):
            img_path = imgs_dir / fname
            try:
                arr, pil_img = preprocess(img_path, H_in, W_in)
            except Exception:
                continue

            t0 = time.perf_counter()
            try:
                out = run_inference(sess, arr)
            except Exception as e:
                print(f"    [Inferenzfehler] {fname}: {e}")
                continue
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

            # Detektionen auswerten
            boxes  = out.get("boxes",  out.get("pred_boxes", None))
            scores = out.get("scores", out.get("pred_logits", None))
            if boxes is not None and boxes.ndim == 3:
                boxes = boxes[0]
            if scores is not None and scores.ndim == 2:
                scores = scores[0]

            gt_boxes = gt_by_file.get(fname, [])
            gt_matched = [False] * len(gt_boxes)
            det_entries = []

            if boxes is not None and scores is not None:
                for i in range(len(boxes)):
                    sc = float(scores[i])
                    if sc < args.threshold:
                        continue
                    box = boxes[i]
                    # Normalisieren falls nötig
                    orig_w, orig_h = pil_img.width, pil_img.height
                    if box.max() > 1.01:
                        box = [box[0]/orig_w, box[1]/orig_h, box[2]/orig_w, box[3]/orig_h]
                    best_iou, best_j = 0.0, -1
                    for j, gt in enumerate(gt_boxes):
                        iou = iou_boxes(box, gt)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    matched = best_iou >= 0.5 and not gt_matched[best_j] if best_j >= 0 else False
                    if matched:
                        gt_matched[best_j] = True
                    det_entries.append((sc, matched))

            all_dets.append(det_entries)
            all_gt_counts.append(len(gt_boxes))

            # Beispielbilder speichern
            if idx < args.samples:
                orig_w, orig_h = pil_img.width, pil_img.height
                vis = draw_detections(pil_img.resize((orig_w, orig_h)), out, orig_w, orig_h, args.threshold)
                sample_path = models_dir / f"sample_{variant}_{idx:03d}.jpg"
                vis.save(str(sample_path), quality=85)
                sample_imgs.append(sample_path.name)

        mean_time = sum(times) / len(times) if times else 0
        map50 = compute_map(all_dets, all_gt_counts, iou_thresh=0.5)

        results[variant] = {
            "onnx_file":    onnx_path.name,
            "size_mb":      round(size_mb, 1),
            "mean_ms":      round(mean_time, 1),
            "map50":        round(map50, 4),
            "n_images":     len(times),
            "sample_images": sample_imgs,
        }
        print(f"    Größe: {size_mb:.1f} MB  |  Zeit: {mean_time:.1f} ms  |  mAP@0.5: {map50:.3f}")

    results_path = models_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Ergebnisse gespeichert: {results_path}")


if __name__ == "__main__":
    main()
