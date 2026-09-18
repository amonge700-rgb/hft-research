"""Generate the 12-sample EXP-021 COMSOL turn-matrix teacher set."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import mph
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP20 = ROOT.parent / "experiment20_comsol_turn_matrix_teacher_20260915"
SOURCE = EXP20 / "model" / "revisions" / "exp20_teacher_4p4_axisym_C_LM_converged_terminal_verified.mph"
OUT = ROOT / "data" / "comsol_teachers"
LOG = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
LABELS = ["P1", "P2", "P3", "P4", "S1", "S2", "S3", "S4"]
GRID = list(itertools.product((1.0, 1.5, 2.0), (0.15, 0.25), (0.0, 0.625)))


def dataset(model, tag):
    return next(x for x in model / "datasets" if x.tag() == tag)


def scalar(model, expr, data):
    return float(np.asarray(model.evaluate(expr, dataset=data)).squeeze())


def write_matrix(path, matrix):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["node", *LABELS])
        for label, row in zip(LABELS, matrix):
            w.writerow([label, *[f"{float(x):.15e}" for x in row]])


def assemble(energies):
    matrix = np.zeros((8, 8))
    for i in range(8): matrix[i, i] = 2 * energies[i]
    case = 8
    for i in range(8):
        for j in range(i + 1, 8):
            matrix[i, j] = matrix[j, i] = energies[case] - energies[i] - energies[j]
            case += 1
    return matrix


def rectangle_positions(gps_mm, gturn_mm, zoff_mm):
    w, t, r0 = 1.0, 0.5, 11.0
    primary, secondary = [], []
    for k in range(4):
        z = (k - 1.5) * (w + gturn_mm) - w / 2
        primary.append((r0, z))
        secondary.append((r0 + t + gps_mm, z + zoff_mm))
    return primary, secondary


def set_geometry(model, gps, gturn, zoff):
    java = model.java; geom = java.component("comp1").geom("geom1")
    model.parameter("g_ps", f"{gps}[mm]")
    model.parameter("g_turn", f"{gturn}[mm]")
    try: model.parameter("z_offset_s", f"{zoff}[mm]")
    except Exception: java.param().set("z_offset_s", f"{zoff}[mm]", "Secondary axial offset")
    p, s = rectangle_positions(gps, gturn, zoff)
    for tag, pos in zip(("P1", "P2", "P3", "P4"), p):
        geom.feature(tag).set("pos", [f"{pos[0]/1000:.12g}", f"{pos[1]/1000:.12g}"])
    for tag, pos in zip(("S1", "S2", "S3", "S4"), s):
        geom.feature(tag).set("pos", [f"{pos[0]/1000:.12g}", f"{pos[1]/1000:.12g}"])
    geom.run(); java.component("comp1").mesh("mesh1").feature("size").set("hauto", "3")
    java.component("comp1").mesh("mesh1").run()
    return p + s


def extract(model):
    java = model.java; dc, dl = dataset(model, "dset1"), dataset(model, "dset3")
    c = np.zeros((8, 8)); tc = 0.0
    for excitation in range(1, 9):
        model.parameter("case_id", str(excitation)); tic = time.perf_counter()
        java.study("stdC2").run(); tc += time.perf_counter() - tic
        for conductor in range(1, 9): c[conductor-1, excitation-1] = scalar(model, f"es.Q0_{conductor}", dc)
    energies, tl = [], 0.0
    for case in range(1, 37):
        model.parameter("case_id", str(case)); tic = time.perf_counter()
        java.study("stdLMstat").run(); tl += time.perf_counter() - tic
        energies.append(scalar(model, "mf.intWm", dl))
    return c, assemble(energies), tc, tl


def validate(c, l):
    return {
        "C_symmetry": float(np.linalg.norm(c-c.T)/np.linalg.norm(c)),
        "L_symmetry": float(np.linalg.norm(l-l.T)/np.linalg.norm(l)),
        "C_min_eigenvalue_F": float(np.linalg.eigvalsh((c+c.T)/2).min()),
        "L_min_eigenvalue_H": float(np.linalg.eigvalsh((l+l.T)/2).min()),
        "C_offdiag_max_F": float(np.max(c[~np.eye(8, dtype=bool)])),
    }


def run_one(model, idx, gps, gturn, zoff, save_model=False):
    sid = f"g{idx:02d}_gps{gps:g}_gt{gturn:g}_zo{zoff:g}".replace(".", "p")
    folder = OUT / sid; folder.mkdir(parents=True, exist_ok=True)
    positions = set_geometry(model, gps, gturn, zoff)
    c, l, tc, tl = extract(model); checks = validate(c, l)
    write_matrix(folder / "C.csv", c); write_matrix(folder / "L.csv", l)
    meta = {"sample_id": sid, "g_ps_mm": gps, "g_turn_mm": gturn,
            "z_offset_s_mm": zoff, "mesh_hauto": 3,
            "electrostatic_8case_time_s": tc, "magnetic_36case_time_s": tl,
            "turn_positions_mm": {k: list(v) for k,v in zip(LABELS, positions)}, **checks}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if save_model: model.save(folder / f"{sid}.mph")
    return meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--start", type=int, default=0); args = ap.parse_args()
    client = mph.start(cores=8, version="6.4"); model = client.load(SOURCE)
    grid = [(1.0, 0.15, 0.625)] if args.pilot else GRID
    rows=[]; log=[]
    for local_idx, values in enumerate(grid):
        idx = args.start + local_idx if args.pilot else local_idx
        try:
            meta = run_one(model, idx, *values, save_model=args.pilot)
            rows.append(meta); log.append(f"PASS {meta['sample_id']} C={meta['C_min_eigenvalue_F']:.4e} L={meta['L_min_eigenvalue_H']:.4e}")
            print(log[-1], flush=True)
        except Exception as exc:
            log.append(f"FAIL index={idx} values={values} error={exc!r}"); print(log[-1], flush=True)
            if args.pilot: raise
    (LOG / ("pilot.log" if args.pilot else "generation.log")).write_text("\n".join(log)+"\n", encoding="utf-8")
    if not args.pilot:
        with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            fields=[k for k in rows[0] if k != "turn_positions_mm"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r[k] for k in fields} for r in rows)


if __name__ == "__main__": main()
