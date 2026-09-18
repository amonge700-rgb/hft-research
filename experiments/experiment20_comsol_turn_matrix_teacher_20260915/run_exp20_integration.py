"""Integrate EXP-020 COMSOL turn matrices with the EXP-015 physics solver.

The COMSOL electrostatic conductors P1..P4,S1..S4 are mapped to turn-potential
nodes p0..p3,s0..s3.  The COMSOL L/M order maps to the corresponding series
turn branches.  Two capacitance models are compared:

* full: every COMSOL mutual capacitance is retained;
* literature_base: turn-to-reference, all interwinding (WW), and adjacent
  same-layer interturn (IT) capacitances are retained.  Non-adjacent
  same-layer capacitances are removed and the Maxwell diagonal is restamped.

This is a quasi-static integration test.  R is computed from copper geometry;
it is not a frequency-dependent COMSOL AC-loss result.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "integration"
FIG = ROOT / "figures"
N = 4


def read_square(name: str) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(RAW / name, index_col=0)
    return list(frame.index), frame.to_numpy(dtype=float)


def branch_incidence() -> np.ndarray:
    a = np.zeros((2 * N, 2 * N))
    for offset in (0, N):
        for k in range(N):
            a[offset + k, offset + k] = 1.0
            if k + 1 < N:
                a[offset + k + 1, offset + k] = -1.0
    return a


def copper_resistance(metadata: pd.DataFrame) -> np.ndarray:
    rho = 1.0 / 5.8e7
    length = 2 * math.pi * metadata["r_center_m"].to_numpy(float)
    area = (metadata["radial_thickness_m"] * metadata["axial_width_m"]).to_numpy(float)
    return rho * length / area


def literature_base_capacitance(c_full: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    mutual = -c_full.copy()
    np.fill_diagonal(mutual, 0.0)
    c_ref = c_full.sum(axis=1)
    kept = np.zeros_like(mutual, dtype=bool)
    classes: list[dict] = []
    for i in range(2 * N):
        for j in range(i + 1, 2 * N):
            same = (i < N) == (j < N)
            adjacent = same and abs((i % N) - (j % N)) == 1
            if not same:
                edge_class, retain = "WW", True
            elif adjacent:
                edge_class, retain = "IT", True
            else:
                edge_class, retain = "same_layer_nonadjacent", False
            kept[i, j] = kept[j, i] = retain
            classes.append({"node_i": i, "node_j": j, "edge_class": edge_class,
                            "capacitance_F": mutual[i, j], "retained": int(retain)})
    m = mutual * kept
    c = -m
    np.fill_diagonal(c, c_ref + m.sum(axis=1))
    return c, classes


def bundle(c: np.ndarray, l: np.ndarray, r: np.ndarray, source: str) -> dict:
    return {"A": branch_incidence(), "R": np.diag(r), "L": l, "C": c,
            "G": np.zeros_like(c), "external": np.array([0, N]), "source": source}


def solve_sweep_numpy(frequency_hz: np.ndarray, b: dict) -> dict:
    """Exact NumPy transcription of EXP-015 physics/kron_solver.py."""
    yn = []
    for f in frequency_hz:
        z = b["R"] + 1j * 2 * np.pi * f * b["L"]
        yn.append(b["A"] @ np.linalg.solve(z, b["A"].T) + b["G"] + 1j * 2 * np.pi * f * b["C"])
    yn = np.stack(yn)
    p = b["external"]
    q = np.array([i for i in range(yn.shape[-1]) if i not in p])
    ypp = yn[:, p][:, :, p]
    ypi = yn[:, p][:, :, q]
    yip = yn[:, q][:, :, p]
    yii = yn[:, q][:, :, q]
    recovery = -np.linalg.solve(yii, yip)
    return {"nodal": yn, "port": ypp + ypi @ recovery, "recovery": recovery, "p": p, "q": q}


def recover_internal_states_numpy(solution: dict, port_voltage: np.ndarray) -> dict:
    vp = np.broadcast_to(port_voltage, (len(solution["nodal"]), len(port_voltage)))
    vi = np.einsum("fip,fp->fi", solution["recovery"], vp)
    v = np.zeros((len(vp), solution["nodal"].shape[-1]), dtype=complex)
    v[:, solution["p"]] = vp; v[:, solution["q"]] = vi
    current = np.einsum("fij,fj->fi", solution["nodal"], v)
    return {"node_voltage": v, "node_current": current,
            "internal_kcl_residual": current[:, solution["q"]]}


def open_input_admittance(y: np.ndarray) -> np.ndarray:
    return y[:, 0, 0] - y[:, 0, 1] * y[:, 1, 0] / y[:, 1, 1]


def cross_current(c: np.ndarray, v: np.ndarray, freq: np.ndarray) -> np.ndarray:
    cps = -c[:N, N:]
    dv = v[:, :N, None] - v[:, None, N:]
    return (1j * 2 * np.pi * freq[:, None, None] * cps[None] * dv).sum((1, 2))


def relnorm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    labels_c, c_full = read_square("capacitance_maxwell.csv")
    labels_l, l_full = read_square("inductance_matrix.csv")
    expected = [f"P{k}" for k in range(1, N + 1)] + [f"S{k}" for k in range(1, N + 1)]
    if labels_c != expected or labels_l != expected:
        raise ValueError(f"unexpected COMSOL node order: C={labels_c}, L={labels_l}")
    meta = pd.read_csv(RAW / "turn_metadata.csv")
    r_dc = copper_resistance(meta)
    c_base, classes = literature_base_capacitance(c_full)

    freq = np.logspace(3, 7, 241)
    full_b = bundle(c_full, l_full, r_dc, "EXP-020 COMSOL full C/L")
    base_b = bundle(c_base, l_full, r_dc, "EXP-020 literature TC/WW/IT base C")
    full = solve_sweep_numpy(freq, full_b)
    base = solve_sweep_numpy(freq, base_b)
    excitation = np.array([1 + 0j, 0 + 0j])
    full_state = recover_internal_states_numpy(full, excitation)
    base_state = recover_internal_states_numpy(base, excitation)
    yin_full, yin_base = open_input_admittance(full["port"]), open_input_admittance(base["port"])
    z_full, z_base = 1 / yin_full, 1 / yin_base
    ic_full = cross_current(c_full, full_state["node_voltage"], freq)
    ic_base = cross_current(c_base, base_state["node_voltage"], freq)

    mag_full = 20 * np.log10(np.maximum(np.abs(full["port"]), 1e-30))
    mag_base = 20 * np.log10(np.maximum(np.abs(base["port"]), 1e-30))
    idx_peak_full = int(np.argmax(np.abs(z_full)))
    idx_peak_base = int(np.argmax(np.abs(z_base)))
    removed = c_full - c_base
    metrics = {
        "scope": "quasi-static C and linear-core L/M; DC geometric R",
        "frequency_points": len(freq),
        "frequency_min_hz": float(freq[0]),
        "frequency_max_hz": float(freq[-1]),
        "full_capacitance_edges": 28,
        "literature_retained_edges": int(sum(x["retained"] for x in classes)),
        "literature_edge_retention": float(sum(x["retained"] for x in classes) / len(classes)),
        "removed_mutual_capacitance_pf": float(-np.triu(removed, 1).sum() * 1e12),
        "removed_fraction_of_mutual_capacitance_percent": float(
            -np.triu(removed, 1).sum() / -np.triu(c_full, 1).sum() * 100),
        "port_admittance_relative_frobenius": relnorm(base["port"], full["port"]),
        "port_admittance_max_magnitude_delta_db": float(np.max(np.abs(mag_base - mag_full))),
        "internal_voltage_relative_l2": relnorm(base_state["node_voltage"], full_state["node_voltage"]),
        "cross_winding_current_relative_l2": relnorm(ic_base, ic_full),
        "full_open_input_impedance_peak_hz": float(freq[idx_peak_full]),
        "literature_open_input_impedance_peak_hz": float(freq[idx_peak_base]),
        "open_input_peak_shift_percent": float((freq[idx_peak_base] / freq[idx_peak_full] - 1) * 100),
        "full_internal_kcl_max_a": float(np.abs(full_state["internal_kcl_residual"]).max()),
        "literature_internal_kcl_max_a": float(np.abs(base_state["internal_kcl_residual"]).max()),
        "dc_resistance_ohm": {expected[i]: float(r_dc[i]) for i in range(2 * N)},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "capacitance_edge_classes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=classes[0].keys())
        writer.writeheader()
        for row in classes:
            row = dict(row); row["node_i"] = expected[row["node_i"]]; row["node_j"] = expected[row["node_j"]]
            writer.writerow(row)
    pd.DataFrame(c_base, index=expected, columns=expected).to_csv(OUT / "capacitance_literature_base.csv")
    pd.DataFrame({
        "frequency_hz": freq,
        "abs_zin_full_ohm": np.abs(z_full),
        "abs_zin_literature_ohm": np.abs(z_base),
        "abs_ips_full_a": np.abs(ic_full),
        "abs_ips_literature_a": np.abs(ic_base),
    }).to_csv(OUT / "frequency_comparison.csv", index=False)

    def color(value, lo, hi, scheme="blue"):
        t = 0.0 if hi == lo else float(np.clip((value - lo) / (hi - lo), 0, 1))
        if scheme == "blue": return f"rgb({int(240-190*t)},{int(248-115*t)},{int(255-35*t)})"
        return f"rgb({int(255-65*t)},{int(244-185*t)},{int(225-185*t)})"

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="430">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#17202a}.title{font-size:17px;font-weight:bold}.lab{font-size:11px}</style>']
    for panel, matrix, title, scheme in ((0, -c_full * 1e12, "Full mutual C (pF)", "blue"),
                                          (1, l_full * 1e9, "COMSOL L/M (nH)", "red")):
        x0 = 45 + panel * 390; y0 = 65; cell = 36
        vals = matrix[~np.eye(8, dtype=bool)] if panel == 0 else matrix.ravel()
        lo, hi = float(vals.min()), float(vals.max())
        svg.append(f'<text class="title" x="{x0}" y="30">{title}</text>')
        for i in range(8):
            svg.append(f'<text class="lab" x="{x0-30}" y="{y0+i*cell+23}">{expected[i]}</text>')
            svg.append(f'<text class="lab" x="{x0+i*cell+8}" y="{y0-8}">{expected[i]}</text>')
            for j in range(8):
                value = matrix[i, j]
                fill = "#eeeeee" if panel == 0 and i == j else color(value, lo, hi, scheme)
                svg.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="35" height="35" fill="{fill}" stroke="white"/>')
    x0, y0, width, height = 825, 65, 445, 288
    lf, lz = np.log10(freq), np.log10(np.maximum(np.abs(z_full), 1e-30))
    lb = np.log10(np.maximum(np.abs(z_base), 1e-30))
    xmin, xmax = lf.min(), lf.max(); ymin, ymax = min(lz.min(), lb.min()), max(lz.max(), lb.max())
    def points(values):
        return " ".join(f"{x0+(x-xmin)/(xmax-xmin)*width:.1f},{y0+height-(y-ymin)/(ymax-ymin)*height:.1f}" for x,y in zip(lf, values))
    svg += [f'<text class="title" x="{x0}" y="30">Open-secondary input impedance</text>',
            f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#333"/>',
            f'<polyline points="{points(lz)}" fill="none" stroke="#1565c0" stroke-width="2"/>',
            f'<polyline points="{points(lb)}" fill="none" stroke="#d84315" stroke-width="2" stroke-dasharray="7,5"/>',
            f'<text class="lab" x="{x0}" y="385">1 kHz</text><text class="lab" x="{x0+width-45}" y="385">10 MHz</text>',
            f'<text class="lab" x="{x0+80}" y="410" fill="#1565c0">full</text>',
            f'<text class="lab" x="{x0+170}" y="410" fill="#d84315">TC/WW/IT base</text>', '</svg>']
    (FIG / "exp20_matrix_and_sparse_comparison.svg").write_text("".join(svg), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
