# RF-DETR-Seg-Nano — Dokumenterkennung & ONNX-Pipeline

Dieses Repository enthält die komplette Pipeline zum Fine-Tuning, Export und zur Evaluation des **RF-DETR-Seg-Nano**-Modells für die Erkennung und Segmentierung von Dokumenten in Fotos.

---

## Idee & Ziel

Das Ziel ist eine **Echtzeit-fähige Dokumentensegmentierung** auf mobilen Geräten (iOS/Android via .NET MAUI). Das Modell soll aus einem beliebigen Foto zuverlässig den Bereich eines Dokuments (Ausweis, Vertrag, Brief, ...) erkennen und dessen genaue Kontur als Maske liefern — als Grundlage für automatisches Zuschneiden und Entzerren.

**RF-DETR-Seg-Nano** wurde gewählt, weil:
- Sehr kleine Modellgröße (~33 MB FP32) → mobilfreundlich
- End-to-End-Architektur (kein NMS nötig)
- ONNX-Export unterstützt
- Gute Qualität bei einfachen Klassen (1 Klasse: `document`)

---

## Projektstruktur

```
rf-detr-document/
├── data/
│   ├── train/
│   │   └── _annotations.coco.json
│   └── valid/
│       └── _annotations.coco.json
├── runs/
│   └── v1_fast/
│       ├── checkpoint_best_ema.pth        ← bester Checkpoint (EMA, Epoch 2)
│       ├── checkpoint_best_regular.pth    ← bester Checkpoint (regulär, Epoch 1)
│       ├── models/                        ← Epoch-1-Modelle
│       ├── models_epoch2/                 ← Epoch-2-Modelle (alle Quantisierungen)
│       └── comparison.html               ← interaktiver Vergleich
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── export_onnx.py
│   ├── quantize_onnx.py
│   ├── evaluate_onnx.py
│   └── generate_html.py
├── rf-detr-seg-nano.pt                    ← Basis-Pretrained-Weights
└── requirements.txt
```

---

## Voraussetzungen

```bash
pip install -r requirements.txt
# rfdetr[train,onnx]>=1.6.3
# onnxruntime, onnxconverter-common, Pillow, opencv-python, numpy, tqdm
```

---

## Schritt-für-Schritt-Workflow

### 1. Datensatz vorbereiten

Der Rohdatensatz wurde aus Roboflow exportiert (COCO-Format). Das Script normalisiert die Kategorien (Kategorie 1 und 2 wurden beide als `document` behandelt → zusammengeführt zu einer einzigen Klasse) und teilt in 80 % Train / 20 % Validierung auf.

```bash
python scripts/prepare_dataset.py \
    --roboflow-dir "/Pfad/zum/Roboflow-Export/train" \
    --val-split 0.2
```

Ausgabe: `data/train/` und `data/valid/` mit COCO-JSON und Bild-Symlinks.

---

### 2. Training

Fine-Tuning auf Basis der offiziellen RF-DETR-Seg-Nano-Weights:

```bash
python scripts/train.py \
    --run-name v1_fast \
    --epochs 50 \
    --batch 4 \
    --lr 1e-4 \
    --device auto
```

Das Modell trainiert mit PyTorch Lightning. Nach jeder Epoche erscheint eine Zeitschätzung. Checkpoints werden unter `runs/<run-name>/` gespeichert:
- `checkpoint_best_ema.pth` — bestes EMA-Modell
- `checkpoint_best_regular.pth` — bestes reguläres Modell

**Erzielte Metriken (Run `v1_fast`):**

| Epoche | mAP 50:95 | mAP 50 | F1   |
|--------|-----------|--------|------|
| 0      | 0.981     | 0.995  | —    |
| 1      | 0.985     | 0.995  | 0.996|
| 2      | 0.990     | 0.995  | 0.996|

---

### 3. ONNX-Export (FP32)

```bash
python scripts/export_onnx.py \
    --run-name v1_fast \
    --checkpoint runs/v1_fast/checkpoint_best_ema.pth \
    --out-dir runs/v1_fast/models_epoch2
```

Erzeugt `model_fp32.onnx` (~125 MB) im angegebenen Ausgabeordner.

---

### 4. Quantisierung

Aus dem FP32-Modell werden weitere Varianten erzeugt:

```bash
python scripts/quantize_onnx.py \
    --run-name v1_fast \
    --fp32-path runs/v1_fast/models_epoch2/model_fp32.onnx \
    --fp16 --int8 --int4
```

| Variante | Größe  | Inferenz (CPU) | Kompatibilität         |
|----------|--------|----------------|------------------------|
| FP32     | 125 MB | ~350 ms        | Universal              |
| FP16     | 67 MB  | —              | GPU only (gemischte Conv-Typen) |
| INT8     | 39 MB  | ~190 ms        | CPU ✓                  |
| INT4     | 34 MB  | ~280 ms        | CPU ✓ (Weight-only)    |

**Methoden:**
- **FP16**: `onnxconverter-common` → alle float32-Gewichte → float16, I/O bleibt float32
- **INT8**: Static Quantization mit Kalibrierungsbildern aus dem Validierungsset (`onnxruntime`)
- **INT4**: Weight-only MatMulNBits-Quantisierung (`MatMulNBitsQuantizer`, block_size=32, symmetric)

---

### 5. Evaluation & HTML-Bericht

```bash
# Evaluation aller Modelle (erzeugt results.json + Beispielbilder)
python scripts/evaluate_onnx.py \
    --run-name v1_fast \
    --models-dir runs/v1_fast/models_epoch2 \
    --max-images 25 \
    --threshold 0.95

# Interaktiven HTML-Bericht generieren
python scripts/generate_html.py \
    --run-name v1_fast \
    --epoch-dirs models models_epoch2 \
    --epoch-labels "Epoch 1" "Epoch 2"
```

Der HTML-Bericht (`comparison.html`) enthält:
- **Epoch-Tabs** — Umschalten zwischen verschiedenen Trainings-Epochen
- **Live-Canvas-Rendering** — Originalbilder werden per JavaScript überlagert (kein Pre-Rendering)
- **Threshold-Slider** — Konfidenzschwelle in Echtzeit anpassen (Standard: 0,95)
- **Varianten-Chips** (FP32 / FP16 / INT8 / INT4) — zum Ein-/Ausblenden
- **Metriktabelle** — mAP@0.5, Modellgröße, Inferenzzeit

---

## ONNX-Modell: Input & Output

### Input

| Name    | Shape              | Dtype   | Beschreibung                                 |
|---------|--------------------|---------|----------------------------------------------|
| `input` | `[1, 3, 312, 312]` | float32 | NCHW, RGB, Pixelwerte / 255.0 (keine ImageNet-Normalisierung) |

Das Bild muss auf **312 × 312** skaliert werden. Die Skalierungsmethode ist bilinear.

### Outputs

| Name     | Shape               | Dtype   | Beschreibung                                                  |
|----------|---------------------|---------|---------------------------------------------------------------|
| `dets`   | `[1, 100, 4]`       | float32 | Bounding Boxes als **cxcywh normiert** (Mittelpunkt + Breite/Höhe, 0–1) |
| `labels` | `[1, 100, 2]`       | float32 | Rohlogits pro Query: `[logit_document, logit_no-object]`     |
| `masks`  | `[1, 100, 78, 78]`  | float32 | Rohlogits für die Segmentierungsmaske (78 × 78 pro Query)    |

**100 Queries** werden immer ausgegeben (RF-DETR-Architektur). Die meisten haben sehr niedrige Scores und müssen gefiltert werden.

### Wichtige Eigenheiten

- **Klasse 1 = kein Objekt** (No-Object-Token). Der Score der Dokumentklasse ist:
  ```
  score = softmax([logit_document, logit_no-object])[0]
  ```
  Nicht einfach `sigmoid(labels[:,0])` — das führt zu falschen Ergebnissen.

- **Masken sind Logits**, kein Sigmoid. Vor der Verwendung:
  ```
  probability = sigmoid(mask_logit)
  binary_mask = probability > 0.5
  ```

- **Masken haben Größe 78 × 78** und müssen auf Originalgröße hochskaliert werden.

- **BBoxen sind cxcywh** (Mittelpunkt-Format), nicht xyxy. Umrechnung:
  ```
  x1 = (cx - w/2) * orig_width
  y1 = (cy - h/2) * orig_height
  ```

---

## Datenformat der Evaluationsergebnisse

`results.json` pro models-Ordner:

```json
{
  "fp32": {
    "size_mb": 125.3,
    "mean_ms": 347.2,
    "map50": 0.994,
    "sample_dets": [
      {
        "boxes": [[x1, y1, x2, y2], ...],
        "scores": [0.997, ...],
        "masks": ["base64-encoded-PNG", ...]
      }
    ]
  }
}
```

---

## Zugehörige .NET-Bibliothek

Die .NET-Preprocessing-Bibliothek für den Einsatz in .NET MAUI-Apps befindet sich in einem separaten Repository:

➜ [github.com/Agredo/RfDetr.Preprocessing](https://github.com/Agredo/RfDetr.Preprocessing)

Sie übernimmt Pre- und Postprocessing des ONNX-Modells (ohne selbst Inferenz auszuführen).
