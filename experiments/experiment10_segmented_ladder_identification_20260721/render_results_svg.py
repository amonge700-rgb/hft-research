"""Render a dependency-free SVG summary from Experiment 10 CSV/JSON files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def heat_color(value: float) -> str:
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        red = 255
        green = int(245 - 150 * value)
        blue = int(245 - 150 * value)
    else:
        red = int(245 + 150 * value)
        green = int(245 + 150 * value)
        blue = 255
    return f"rgb({red},{green},{blue})"


def main() -> None:
    validation = json.loads((RESULTS / "validation_summary.json").read_text(encoding="utf-8"))
    with (RESULTS / "cps_sensitivity_correlation.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        correlation_rows = list(csv.DictReader(handle))
    with (RESULTS / "local_inversion_summary.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        inversion_rows = list(csv.DictReader(handle))

    noise_values = sorted({float(row["noise_pct"]) for row in inversion_rows})
    by_noise = []
    for noise in noise_values:
        selected = [row for row in inversion_rows if float(row["noise_pct"]) == noise]
        unit = sum(float(row["unit_location_accuracy_pct"]) for row in selected) / len(selected)
        group = sum(float(row["group_location_accuracy_pct"]) for row in selected) / len(selected)
        by_noise.append((noise, unit, group))

    width, height = 1280, 560
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#1f2937}.title{font-size:22px;font-weight:700}.sub{font-size:16px;font-weight:700}.tick{font-size:12px}.note{font-size:13px}</style>',
        '<text x="35" y="38" class="title">Experiment 10: segmented HFT matrix model and spatial identifiability</text>',
    ]

    # Panel A: correlation heatmap.
    x0, y0, cell = 55, 105, 72
    svg.append(f'<text x="{x0}" y="78" class="sub">A. Log-sensitivity correlation</text>')
    labels = ["Cps1", "Cps2", "Cps3", "Cps4"]
    for j, label in enumerate(labels):
        svg.append(f'<text x="{x0 + j*cell + cell/2}" y="{y0-12}" text-anchor="middle" class="tick">{label}</text>')
    for i, row in enumerate(correlation_rows):
        svg.append(f'<text x="{x0-10}" y="{y0+i*cell+cell/2+4}" text-anchor="end" class="tick">{labels[i]}</text>')
        for j, label in enumerate(labels):
            value = float(row[label])
            color = heat_color(value)
            tx = x0 + j * cell
            ty = y0 + i * cell
            svg.append(f'<rect x="{tx}" y="{ty}" width="{cell}" height="{cell}" fill="{color}" stroke="white"/>')
            svg.append(f'<text x="{tx+cell/2}" y="{ty+cell/2+5}" text-anchor="middle" class="note">{value:.3f}</text>')
    svg.append(f'<text x="{x0}" y="{y0+4*cell+30}" class="note">Interior sections Cps2-Cps4 are nearly collinear.</text>')

    # Panel B: Fisher spectrum.
    bx, by, bw, bh = 455, 105, 300, 288
    eig = sorted(validation["fisher_eigenvalues"], reverse=True)
    svg.append(f'<text x="{bx}" y="78" class="sub">B. Fisher eigenvalue spectrum</text>')
    svg.append(f'<line x1="{bx}" y1="{by+bh}" x2="{bx+bw}" y2="{by+bh}" stroke="#374151"/>')
    svg.append(f'<line x1="{bx}" y1="{by}" x2="{bx}" y2="{by+bh}" stroke="#374151"/>')
    log_min = math.floor(math.log10(min(eig)))
    log_max = math.ceil(math.log10(max(eig)))
    for power in range(log_min, log_max + 1):
        yy = by + bh * (log_max - power) / (log_max - log_min)
        svg.append(f'<line x1="{bx}" y1="{yy}" x2="{bx+bw}" y2="{yy}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{bx-8}" y="{yy+4}" text-anchor="end" class="tick">1e{power}</text>')
    bar_w = 42
    for i, value in enumerate(eig):
        xv = bx + 35 + i * 65
        yv = by + bh * (log_max - math.log10(value)) / (log_max - log_min)
        svg.append(f'<rect x="{xv}" y="{yv}" width="{bar_w}" height="{by+bh-yv}" fill="#2563eb"/>')
        svg.append(f'<text x="{xv+bar_w/2}" y="{by+bh+18}" text-anchor="middle" class="tick">{i+1}</text>')
    svg.append(f'<text x="{bx}" y="{by+bh+46}" class="note">Fisher condition number = {validation["fisher_condition_number"]:.2e}</text>')

    # Panel C: location accuracy by noise.
    cx, cy, cw, ch = 850, 105, 355, 288
    svg.append(f'<text x="{cx}" y="78" class="sub">C. Localization accuracy vs. complex noise</text>')
    svg.append(f'<line x1="{cx}" y1="{cy+ch}" x2="{cx+cw}" y2="{cy+ch}" stroke="#374151"/>')
    svg.append(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy+ch}" stroke="#374151"/>')
    for pct in [0, 25, 50, 75, 100]:
        yy = cy + ch * (1.0 - pct / 100.0)
        svg.append(f'<line x1="{cx}" y1="{yy}" x2="{cx+cw}" y2="{yy}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{cx-8}" y="{yy+4}" text-anchor="end" class="tick">{pct}%</text>')
    for idx, (noise, unit, group) in enumerate(by_noise):
        center = cx + 65 + idx * 105
        for offset, value, color in [(-24, unit, "#2563eb"), (8, group, "#f97316")]:
            bar_h = ch * value / 100.0
            svg.append(f'<rect x="{center+offset}" y="{cy+ch-bar_h}" width="26" height="{bar_h}" fill="{color}"/>')
        svg.append(f'<text x="{center}" y="{cy+ch+19}" text-anchor="middle" class="tick">{noise:.2f}%</text>')
    svg.append(f'<rect x="{cx+35}" y="{cy+ch+42}" width="14" height="14" fill="#2563eb"/><text x="{cx+55}" y="{cy+ch+54}" class="note">4-unit</text>')
    svg.append(f'<rect x="{cx+125}" y="{cy+ch+42}" width="14" height="14" fill="#f97316"/><text x="{cx+145}" y="{cy+ch+54}" class="note">terminal/interior group</text>')

    svg.append('<text x="35" y="535" class="note">Conclusion: the matrix model is physically consistent, but internal unit-level localization is noise-sensitive; observation design must precede a final inverse solver.</text>')
    svg.append("</svg>")
    (RESULTS / "experiment10_summary.svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
