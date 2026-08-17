"""Generate publication-ready vector figures without a matplotlib dependency."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors


ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "results" / "segmentation_metrics.csv").open(encoding="utf-8-sig")))
strategies = ["uniform", "physical", "information_guided"]
palette = [colors.HexColor("#2E75B6"), colors.HexColor("#C76A20"), colors.HexColor("#2E7D5B")]


def chart(drawing: Drawing, x: float, y: float, title: str, column: str, target: float) -> None:
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = x, y, 285, 175
    series = []
    for strategy in strategies:
        selected = sorted(
            (row for row in rows if row["strategy"] == strategy),
            key=lambda row: int(row["coarse_sections_per_winding"]),
        )
        series.append(
            [
                (int(row["coarse_sections_per_winding"]), math.log10(max(float(row[column]), 1e-9)))
                for row in selected
            ]
        )
    plot.data = series + [[(2, math.log10(target)), (12, math.log10(target))]]
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 2, 12
    plot.xValueAxis.valueSteps = [2, 4, 6, 8, 12]
    all_y = [point[1] for values in series for point in values] + [math.log10(target)]
    plot.yValueAxis.valueMin = math.floor(min(all_y))
    plot.yValueAxis.valueMax = math.ceil(max(all_y))
    plot.yValueAxis.labelTextFormat = lambda value: f"1e{int(value)}"
    for i, color in enumerate(palette):
        plot.lines[i].strokeColor = color
        plot.lines[i].strokeWidth = 1.8
        plot.lines[i].symbol = None
    plot.lines[3].strokeColor = colors.black
    plot.lines[3].strokeDashArray = [4, 3]
    drawing.add(plot)
    drawing.add(String(x + 142, y + 194, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=10))
    drawing.add(String(x + 142, y - 18, "Sections per winding", textAnchor="middle", fontSize=8))
    drawing.add(String(x - 38, y + 82, "Error (%)", textAnchor="middle", fontSize=8, angle=90))


drawing = Drawing(650, 500)
chart(drawing, 55, 275, "Port admittance", "port_error_percent", 0.2)
chart(drawing, 355, 275, "Key modal frequencies", "resonance_error_percent", 0.01)
chart(drawing, 55, 35, "Internal peak voltage", "internal_peak_error_percent", 0.1)
chart(drawing, 355, 35, "Adjacent-voltage stress", "adjacent_voltage_stress_error_percent", 1.0)

legend_y = 488
for i, (strategy, color) in enumerate(zip(strategies, palette)):
    x = 120 + i * 170
    drawing.add(String(x, legend_y, strategy.replace("_", " "), fillColor=color, fontName="Helvetica-Bold", fontSize=9))

(ROOT / "figures").mkdir(exist_ok=True)
renderPDF.drawToFile(drawing, str(ROOT / "figures" / "segmentation_comparison.pdf"))
