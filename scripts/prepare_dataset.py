"""
scripts/prepare_dataset.py

Bereitet den Roboflow-Datensatz für RF-DETR-Seg-Training vor.

Was dieses Script macht:
  1. Liest den Roboflow COCO-Datensatz (nur "train"-Split vorhanden).
  2. Normalisiert Kategorien: cat_id 1 und 2 (beide "documents") → cat_id 1 "document".
  3. Teilt in 80% train / 20% valid auf.
  4. Erstellt die von RF-DETR erwartete Struktur:
       data/
         train/
           _annotations.coco.json
           image1.jpg  (Symlink)
           ...
         valid/
           _annotations.coco.json
           image1.jpg  (Symlink)
           ...

Verwendung:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --roboflow-dir "/Users/agredo/Doument Detection/train"
    python scripts/prepare_dataset.py --val-split 0.2 --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


CATEGORY = {"id": 1, "name": "document", "supercategory": "object"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboflow-Datensatz für RF-DETR aufbereiten")
    parser.add_argument(
        "--roboflow-dir",
        default="/Users/agredo/Doument Detection/train",
        help="Pfad zum Roboflow-train-Ordner mit _annotations.coco.json",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Ausgabeverzeichnis (Standard: data/)",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Anteil Validierung (Standard: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_dir = Path(args.roboflow_dir)
    out_dir = Path(args.out_dir)
    ann_path = src_dir / "_annotations.coco.json"

    if not ann_path.exists():
        print(f"FEHLER: {ann_path} nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    print(f"Lese Annotationen: {ann_path}")
    with open(ann_path, encoding="utf-8") as f:
        raw = json.load(f)

    images = raw["images"]
    annotations = raw["annotations"]

    # Kategorien normalisieren: cat_id 1 und 2 → 1
    remapped_ids = {c["id"] for c in raw["categories"] if c["name"] in ("documents", "Document Scanner - v1 Document Dataset - v2_1")}
    print(f"  Kategorien werden zu 'document' (id=1) normalisiert: {remapped_ids}")

    # Index: image_id → Liste von Annotationen (erste nutzen, falls mehrere)
    ann_by_img: dict[int, list] = {}
    for ann in annotations:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Bilder mit nutzbarer Annotation
    usable = [img for img in images if img["id"] in ann_by_img]
    print(f"  Bilder gesamt: {len(images)}  davon mit Annotation: {len(usable)}")

    # Shuffle + Split
    random.seed(args.seed)
    random.shuffle(usable)
    n_val = max(1, int(len(usable) * args.val_split))
    val_imgs   = usable[:n_val]
    train_imgs = usable[n_val:]
    print(f"  Split → train: {len(train_imgs)}  valid: {len(val_imgs)}")

    for split_name, split_imgs in [("train", train_imgs), ("valid", val_imgs)]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_images, coco_annotations = [], []
        img_id_map: dict[int, int] = {}   # old_id → new sequential id
        ann_id = 1

        for new_img_id, img_obj in enumerate(split_imgs, start=1):
            img_id_map[img_obj["id"]] = new_img_id
            coco_images.append({
                "id": new_img_id,
                "file_name": img_obj["file_name"],
                "width": img_obj["width"],
                "height": img_obj["height"],
            })

            # Symlink anlegen
            src_img = src_dir / img_obj["file_name"]
            dst_img = split_dir / img_obj["file_name"]
            if not dst_img.exists() and not dst_img.is_symlink():
                if src_img.exists():
                    os.symlink(src_img.resolve(), dst_img)

            for ann in ann_by_img.get(img_obj["id"], []):
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": new_img_id,
                    "category_id": 1,          # normalisiert
                    "bbox": ann["bbox"],
                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "segmentation": seg,
                    "iscrowd": 0,
                })
                ann_id += 1

        coco_out = {
            "info": {"description": "Document Detection – RF-DETR Segmentation Dataset", "version": "1.0"},
            "licenses": [],
            "categories": [CATEGORY],
            "images": coco_images,
            "annotations": coco_annotations,
        }
        ann_out = split_dir / "_annotations.coco.json"
        with open(ann_out, "w", encoding="utf-8") as f:
            json.dump(coco_out, f, separators=(",", ":"))

        print(f"  [{split_name}] {len(coco_images)} Bilder, {len(coco_annotations)} Annotationen → {ann_out}")

    print("\n✓ Datensatz bereit unter:", out_dir.resolve())


if __name__ == "__main__":
    main()
