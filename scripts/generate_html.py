"""
scripts/generate_html.py

Erstellt eine standalone HTML-Vergleichsseite für alle ONNX-Varianten.

Zeigt für jeden Run:
  - Metriktabelle (Modellgröße, Inferenzzeit, mAP@0.5)
  - Beispielbilder mit Segmentierungsmasken/BBoxen
  - Alle Bilder als Base64 eingebettet → vollständig offline nutzbar

Verwendung:
    python scripts/generate_html.py
    python scripts/generate_html.py --run-name rf_detr_nano --out runs/comparison.html
"""

import argparse
import base64
import json
import sys
from pathlib import Path


VARIANT_LABELS = {
    "fp32": ("FP32", "#4A90D9", "Volle Präzision"),
    "fp16": ("FP16", "#7B68EE", "Halbpräzision"),
    "int8": ("INT8", "#F5A623", "8-Bit Integer"),
    "int4": ("INT4", "#E74C3C", "4-Bit Integer"),
}


def img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_html(run_name: str, results: dict, models_dir: Path, training_params: dict | None) -> str:
    # Varianten in der richtigen Reihenfolge
    order = ["fp32", "fp16", "int8", "int4"]
    variants = [v for v in order if v in results]

    # --- Metriktabelle ---
    table_rows = ""
    for v in variants:
        r = results[v]
        label, color, desc = VARIANT_LABELS.get(v, (v.upper(), "#888", ""))
        table_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{label}</span></td>
          <td>{desc}</td>
          <td>{r['size_mb']} MB</td>
          <td>{r['mean_ms']} ms</td>
          <td>{r['map50']:.3f}</td>
          <td>{r['n_images']}</td>
        </tr>"""

    # --- Beispielbilder ---
    sample_sections = ""
    n_samples = max((len(results[v].get("sample_images", [])) for v in variants), default=0)

    for idx in range(n_samples):
        cards = ""
        for v in variants:
            r = results[v]
            imgs = r.get("sample_images", [])
            if idx >= len(imgs):
                continue
            label, color, _ = VARIANT_LABELS.get(v, (v.upper(), "#888", ""))
            img_path = models_dir / imgs[idx]
            if not img_path.exists():
                continue
            b64 = img_to_base64(img_path)
            cards += f"""
            <div class="card">
              <div class="card-header" style="background:{color}">{label}</div>
              <img src="data:image/jpeg;base64,{b64}" alt="{label} Sample {idx+1}">
            </div>"""
        if cards:
            sample_sections += f"""
        <div class="sample-group">
          <h3>Bild {idx + 1}</h3>
          <div class="card-row">{cards}
          </div>
        </div>"""

    # --- Trainingsparameter ---
    params_html = ""
    if training_params:
        params_html = "<h2>Trainingsparameter</h2><table class='params-table'>"
        for k, v in training_params.items():
            params_html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        params_html += "</table>"

    # --- Varianten Chips für Filter ---
    filter_chips = "".join(
        f'<button class="chip active" data-variant="{v}" '
        f'style="--chip-color:{VARIANT_LABELS.get(v, (v, "#888", ""))[1]}" '
        f'onclick="toggleVariant(this)">{VARIANT_LABELS.get(v, (v, "#888", ""))[0]}</button>'
        for v in variants
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RF-DETR Seg Nano — ONNX Vergleich: {run_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e0e0e0;
      padding: 24px;
    }}
    h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
    h2 {{ font-size: 1.2rem; margin: 32px 0 12px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }}
    h3 {{ font-size: 1rem; color: #888; margin: 20px 0 8px; }}
    .subtitle {{ color: #666; margin-bottom: 28px; font-size: 0.9rem; }}
    .metrics-table {{
      width: 100%; border-collapse: collapse; margin-bottom: 24px;
    }}
    .metrics-table th {{
      background: #1e2130; padding: 10px 14px; text-align: left;
      font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .metrics-table td {{ padding: 10px 14px; border-bottom: 1px solid #1e2130; }}
    .metrics-table tr:hover td {{ background: #1a1f2e; }}
    .badge {{
      display: inline-block; padding: 3px 10px; border-radius: 4px;
      font-size: 0.75rem; font-weight: 700; color: #fff; letter-spacing: 0.5px;
    }}
    .params-table {{ border-collapse: collapse; font-size: 0.85rem; }}
    .params-table td {{ padding: 5px 16px 5px 0; border-bottom: 1px solid #1e2130; }}
    .params-table b {{ color: #aaa; }}
    .filters {{ margin: 16px 0; display: flex; gap: 8px; flex-wrap: wrap; }}
    .chip {{
      padding: 6px 14px; border-radius: 20px; border: 2px solid var(--chip-color);
      background: transparent; color: var(--chip-color); cursor: pointer;
      font-size: 0.8rem; font-weight: 600; transition: all 0.15s;
    }}
    .chip.active {{ background: var(--chip-color); color: #fff; }}
    .sample-group {{ margin-bottom: 28px; }}
    .card-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: #1a1f2e; border-radius: 8px; overflow: hidden;
      transition: transform 0.15s;
    }}
    .card:hover {{ transform: scale(1.02); }}
    .card-header {{
      padding: 6px 12px; font-size: 0.75rem; font-weight: 700;
      color: #fff; letter-spacing: 0.5px;
    }}
    .card img {{ width: 100%; display: block; }}
  </style>
</head>
<body>
  <h1>RF-DETR Seg Nano</h1>
  <p class="subtitle">Run: <b>{run_name}</b> &nbsp;·&nbsp; Modell: RFDETRSegNano &nbsp;·&nbsp; Aufgabe: Instance Segmentation</p>

  <h2>Metriken</h2>
  <table class="metrics-table">
    <thead>
      <tr>
        <th>Variante</th><th>Beschreibung</th><th>Modellgröße</th>
        <th>Ø Inferenz</th><th>mAP@0.5</th><th>Bilder</th>
      </tr>
    </thead>
    <tbody>{table_rows}
    </tbody>
  </table>

  {params_html}

  <h2>Beispielbilder</h2>
  <div class="filters">
    <span style="color:#666; font-size:0.8rem; line-height:2">Anzeigen:</span>
    {filter_chips}
  </div>

  <div id="samples">
    {sample_sections}
  </div>

  <script>
    function toggleVariant(btn) {{
      btn.classList.toggle('active');
      const variant = btn.dataset.variant;
      document.querySelectorAll('.card[data-variant="' + variant + '"]').forEach(c => {{
        c.style.display = btn.classList.contains('active') ? '' : 'none';
      }});
    }}
    // Varianten-Attribute an Cards setzen
    const variantClasses = {json.dumps({v: VARIANT_LABELS.get(v, (v,"",""))[0] for v in variants})};
    document.querySelectorAll('.card').forEach((card, i) => {{
      const header = card.querySelector('.card-header');
      if (header) {{
        for (const [variant, label] of Object.entries(variantClasses)) {{
          if (header.textContent.trim() === label) {{
            card.dataset.variant = variant;
          }}
        }}
      }}
    }});
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Vergleichs-HTML generieren")
    parser.add_argument("--run-name", default="rf_detr_nano")
    parser.add_argument("--out",      default=None,
                        help="Ausgabepfad (Standard: runs/<run-name>/comparison.html)")
    args = parser.parse_args()

    models_dir = Path("runs") / args.run_name / "models"
    results_path = models_dir / "results.json"

    if not results_path.exists():
        print(f"[FEHLER] results.json nicht gefunden: {results_path}", file=sys.stderr)
        print("Bitte zuerst: python scripts/evaluate_onnx.py", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    params_path = Path("runs") / args.run_name / "training_params.json"
    training_params = json.loads(params_path.read_text()) if params_path.exists() else None

    html = build_html(args.run_name, results, models_dir, training_params)

    out_path = Path(args.out) if args.out else Path("runs") / args.run_name / "comparison.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ HTML gespeichert: {out_path.resolve()}")


if __name__ == "__main__":
    main()
