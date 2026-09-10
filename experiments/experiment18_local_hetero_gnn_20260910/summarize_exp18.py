import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = ["falcon_homogeneous", "hft_typed", "edge_centric_hetero"]
TASKS = ["graph_response_mae", "node_voltage_mae", "cross_edge_log_current_mae"]
LABELS = ["FALCON homogeneous", "HFT typed", "Edge-centric hetero"]


def load(mode):
    return json.loads((ROOT / "results" / mode / "metrics.json").read_text(encoding="utf-8"))


def main():
    absolute = load("absolute")
    residual = load("residual")
    rows = []
    for mode, data in [("absolute", absolute), ("residual", residual)]:
        for model in MODELS:
            rows.append({"target_mode": mode, "model": model, **{task: data[model][task] for task in TASKS},
                         "peak_node_hit_rate": data[model]["peak_node_hit_rate"],
                         "parameters": data[model]["parameters"],
                         "train_seconds": data[model]["train_seconds"]})
    out = ROOT / "results"
    with (out / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    base = residual["falcon_homogeneous"]
    hetero = residual["edge_centric_hetero"]
    improvements = {task: 100.0 * (base[task] - hetero[task]) / base[task] for task in TASKS}
    summary = {
        "residual_edge_centric_vs_falcon_improvement_percent": improvements,
        "interpretation": {
            "graph_and_node": "small positive advantage under local residual targets",
            "cross_edge_current": "no advantage; edge-centric model is slightly worse",
            "localization": "all models remain weak, especially for larger topologies",
            "absolute_target": "heterogeneous models do not outperform the homogeneous baseline"
        }
    }
    (out / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    x = np.arange(len(MODELS))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, task, title in zip(axes, TASKS, ["Port response MAE", "Internal voltage MAE", "Cross-edge current log-MAE"]):
        vals_abs = [absolute[m][task] for m in MODELS]
        vals_res = [residual[m][task] for m in MODELS]
        w = 0.36
        ax.bar(x - w / 2, vals_abs, width=w, color=colors, alpha=0.45, hatch="//", label="absolute")
        ax.bar(x + w / 2, vals_res, width=w, color=colors, alpha=0.95, label="residual")
        ax.set_title(title)
        ax.set_xticks(x, ["FALCON", "Typed", "Edge-centric"])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("MAE (lower is better)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#777", alpha=.45, hatch="//"),
               plt.Rectangle((0, 0), 1, 1, facecolor="#777", alpha=.95)]
    axes[0].legend(handles, ["Absolute", "Residual"], loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ablation_absolute_vs_residual.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
