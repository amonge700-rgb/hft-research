from __future__ import annotations

import csv
import json
import math
import time
import traceback
from pathlib import Path

import mph
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model" / "exp20_teacher_4p4_axisym_C_LM.mph"
OUT = ROOT / "data" / "convergence"
MODELS = ROOT / "model" / "revisions"
FIGS = ROOT / "figures" / "convergence"
LOGS = ROOT / "logs"
for folder in (OUT, MODELS, FIGS, LOGS):
    folder.mkdir(parents=True, exist_ok=True)

LABELS = ["P1", "P2", "P3", "P4", "S1", "S2", "S3", "S4"]
BOUNDARIES = [
    [9, 10, 11, 21], [12, 13, 14, 22], [15, 16, 17, 23], [18, 19, 20, 24],
    [25, 26, 27, 37], [28, 29, 30, 38], [31, 32, 33, 39], [34, 35, 36, 40],
]
LEVELS = [("coarse", 5), ("medium", 3), ("fine", 1)]


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", *LABELS])
        for label, row in zip(LABELS, matrix):
            w.writerow([label, *[f"{float(x):.15e}" for x in row]])


def assemble(energies: list[float]) -> np.ndarray:
    matrix = np.zeros((8, 8))
    for i in range(8):
        matrix[i, i] = 2.0 * energies[i]
    case = 8
    for i in range(8):
        for j in range(i + 1, 8):
            matrix[i, j] = matrix[j, i] = energies[case] - energies[i] - energies[j]
            case += 1
    return matrix


def dataset_by_tag(model, tag: str):
    for node in model / "datasets":
        if node.tag() == tag:
            return node
    raise RuntimeError(f"Dataset {tag} not found")


def scalar(model, expression: str, dataset) -> float:
    return float(np.asarray(model.evaluate(expression, dataset=dataset)).squeeze())


def solve_cases(model, study_tag: str, dataset, energy_expr: str) -> tuple[list[float], float]:
    energies = []
    total = 0.0
    for case_id in range(1, 37):
        model.parameter("case_id", str(case_id))
        t0 = time.perf_counter()
        model.java.study(study_tag).run()
        total += time.perf_counter() - t0
        energies.append(scalar(model, energy_expr, dataset))
    return energies, total


def mesh_stats(mesh) -> tuple[int, int]:
    elements = int(mesh.getNumElem())
    vertices = int(mesh.getNumVertex())
    return elements, vertices


def solution_dof(java_model, tag: str) -> int:
    sol = java_model.sol(tag)
    for method in ("getSize", "getNU"):
        try:
            value = getattr(sol, method)()
            return int(np.asarray(value).max())
        except Exception:
            pass
    return -1


def export_plot(java_model, tag: str, label: str, dataset_tag: str, feature_type: str,
                expression: str | None, filename: Path) -> None:
    result = java_model.result()
    if tag in list(result.tags()):
        result.remove(tag)
    pg = result.create(tag, "PlotGroup2D")
    pg.label(label)
    pg.set("data", dataset_tag)
    feat = pg.create(tag + "_f", feature_type)
    if expression is not None:
        feat.set("expr", expression)
    pg.run()
    etag = "img_" + tag
    if etag in list(result.export().tags()):
        result.export().remove(etag)
    image = result.export().create(etag, "Image2D")
    image.set("plotgroup", tag)
    image.set("pngfilename", str(filename))
    image.set("width", "1800")
    image.set("height", "1200")
    image.run()


def main() -> None:
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [f"started={started}", f"source={SOURCE}"]
    client = mph.start(cores=8, version="6.4")
    model = client.load(SOURCE)
    java = model.java
    comp = java.component("comp1")
    mesh = comp.mesh("mesh1")
    dset_c = dataset_by_tag(model, "dset1")
    dset_l = dataset_by_tag(model, "dset3")

    members = [[i + 1] for i in range(8)]
    case_id = 9
    for i in range(8):
        for j in range(i + 1, 8):
            members[i].append(case_id)
            members[j].append(case_id)
            case_id += 1
    for i, cases in enumerate(members, start=1):
        cond = "||".join(f"case_id=={case}" for case in cases)
        model.parameter(f"V{i}", f"if({cond},1[V],0[V])")
        model.parameter(f"I{i}", f"if({cond},1[A],0[A])")

    results: dict[str, dict] = {}
    status_rows = []
    for level, hauto in LEVELS:
        mesh.feature("size").set("hauto", str(hauto))
        t_mesh = time.perf_counter()
        mesh.run()
        mesh_time = time.perf_counter() - t_mesh
        elements, vertices = mesh_stats(mesh)

        e_c, t_c = solve_cases(model, "stdC2", dset_c, "es.intWe")
        C = assemble(e_c)
        e_l, t_l = solve_cases(model, "stdLMstat", dset_l, "mf.intWm")
        L = assemble(e_l)
        results[level] = {"C": C, "L": L}
        write_matrix(OUT / f"capacitance_maxwell_{level}.csv", C)
        write_matrix(OUT / f"inductance_matrix_{level}.csv", L)
        with (OUT / f"energy_cases_{level}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "electrostatic_energy_J", "magnetic_energy_J"])
            for k, (wc, wl) in enumerate(zip(e_c, e_l), start=1):
                w.writerow([k, f"{wc:.15e}", f"{wl:.15e}"])
        dof_c = solution_dof(java, "sol1")
        dof_l = solution_dof(java, "sol3")
        status_rows.append({
            "mesh": level, "hauto": hauto, "elements": elements, "vertices": vertices,
            "dof_electrostatic": dof_c, "dof_magnetic": dof_l,
            "mesh_time_s": mesh_time, "electrostatic_36case_time_s": t_c,
            "magnetic_36case_time_s": t_l, "converged": True,
        })
        model.save(MODELS / f"exp20_teacher_4p4_axisym_C_LM_{level}.mph")
        log_lines.append(
            f"{level}: elements={elements}, vertices={vertices}, dofC={dof_c}, "
            f"dofL={dof_l}, mesh_s={mesh_time:.3f}, C_s={t_c:.3f}, L_s={t_l:.3f}"
        )

    with (OUT / "mesh_solver_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(status_rows[0]))
        w.writeheader()
        w.writerows(status_rows)

    comparisons = []
    for a, b in (("coarse", "medium"), ("medium", "fine")):
        for kind in ("C", "L"):
            old, new = results[a][kind], results[b][kind]
            scale = np.maximum(np.maximum(np.abs(old), np.abs(new)), np.finfo(float).tiny)
            rel = np.abs(new - old) / scale
            write_matrix(OUT / f"{kind}_relative_change_{a}_to_{b}.csv", rel)
            off = np.abs(new.copy())
            np.fill_diagonal(off, 0)
            important = off >= 0.01 * off.max()
            diag_max = float(np.diag(rel).max())
            mutual_max = float(rel[important].max()) if important.any() else 0.0
            comparisons.append({
                "matrix": kind, "from": a, "to": b,
                "max_diagonal_relative_change": diag_max,
                "max_important_mutual_relative_change": mutual_max,
                "accepted_lt_1pct": diag_max < 0.01 and mutual_max < 0.01,
            })
    with (OUT / "mesh_convergence_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        w.writeheader()
        w.writerows(comparisons)

    # Independent conductor-charge extraction on the fine mesh.
    for tag, boundaries in zip((f"intQ{i}" for i in range(1, 9)), BOUNDARIES):
        if tag in list(comp.cpl().tags()):
            comp.cpl().remove(tag)
        op = comp.cpl().create(tag, "Integration")
        op.selection().geom("geom1", 1)
        op.selection().set(boundaries)
    Q_raw = np.zeros((8, 8))
    for excitation in range(1, 9):
        model.parameter("case_id", str(excitation))
        java.study("stdC2").run()
        for conductor in range(1, 9):
            Q_raw[conductor - 1, excitation - 1] = scalar(
                # Explicit 2*pi*r weighting is required for a boundary
                # integration coupling in a 2D axisymmetric component.
                model, f"intQ{conductor}(-es.nD*2*pi*r)", dset_c
            )
    C_energy = results["fine"]["C"]
    candidates = [Q_raw, -Q_raw]
    C_charge = min(candidates, key=lambda x: np.linalg.norm(x - C_energy))
    diff = C_charge - C_energy
    fro = float(np.linalg.norm(diff) / np.linalg.norm(C_energy))
    element_scale = np.maximum(np.abs(C_energy), np.finfo(float).tiny)
    worst = float(np.max(np.abs(diff) / element_scale))
    write_matrix(OUT / "capacitance_charge_fine.csv", C_charge)
    write_matrix(OUT / "capacitance_charge_minus_energy.csv", diff)
    with (OUT / "charge_method_validation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["relative_frobenius_error", f"{fro:.15e}"])
        w.writerow(["worst_elementwise_relative_error", f"{worst:.15e}"])
        w.writerow(["charge_sign_selected", "raw" if C_charge is Q_raw else "negated_raw"])

    # Representative final-mesh plots.
    model.parameter("case_id", "1")
    java.study("stdC2").run()
    export_plot(java, "pg_geom", "EXP-020 geometry", "dset1", "Surface", "1",
                FIGS / "geometry.png")
    export_plot(java, "pg_mesh", "EXP-020 final mesh", "dset1", "Mesh", None,
                FIGS / "final_mesh.png")
    export_plot(java, "pg_E", "P1 electric field", "dset1", "Surface", "es.normE",
                FIGS / "P1_electric_field.png")
    java.study("stdLMstat").run()
    export_plot(java, "pg_B", "P1 magnetic flux density", "dset3", "Surface", "mf.normB",
                FIGS / "P1_magnetic_flux_density.png")
    model.parameter("case_id", "12")  # P1 + S1 in the established case map.
    java.study("stdC2").run()
    export_plot(java, "pg_PS", "P1 plus S1 coupled electric field", "dset1", "Surface", "es.normE",
                FIGS / "P1_S1_coupled_field.png")

    model.save(MODELS / "exp20_teacher_4p4_axisym_C_LM_converged_charge_verified.mph")
    summary = {
        "source_model": str(SOURCE),
        "mesh_metrics": status_rows,
        "convergence": comparisons,
        "charge_validation": {
            "relative_frobenius_error": fro,
            "worst_elementwise_relative_error": worst,
        },
        "restrictions": {
            "DAB_added": False, "thermal_added": False, "GNN_added": False,
            "nonlinear_core_added": False,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_lines.append(f"charge_relative_frobenius_error={fro:.6e}")
    log_lines.append(f"charge_worst_elementwise_relative_error={worst:.6e}")
    log_lines.append(f"finished={time.strftime('%Y-%m-%d %H:%M:%S')}")
    (LOGS / "convergence_charge_solver_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        (LOGS / "convergence_charge_error.log").write_text(error, encoding="utf-8")
        print(error)
        raise
