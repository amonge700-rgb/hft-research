from pathlib import Path
import csv
import json
import mph
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model" / "revisions" / "exp20_teacher_4p4_axisym_C_LM_converged_charge_verified.mph"
OUT = ROOT / "data" / "convergence"
REV = ROOT / "model" / "revisions" / "exp20_teacher_4p4_axisym_C_LM_converged_charge_verified_v2.mph"
LABELS = ["P1", "P2", "P3", "P4", "S1", "S2", "S3", "S4"]

def dataset_by_tag(model, tag):
    return next(node for node in model / "datasets" if node.tag() == tag)

def scalar(model, expression, dataset):
    return float(np.asarray(model.evaluate(expression, dataset=dataset)).squeeze())

def write_matrix(path, matrix):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", *LABELS])
        for label, row in zip(LABELS, matrix):
            w.writerow([label, *[f"{float(x):.15e}" for x in row]])

client = mph.start(cores=8, version="6.4")
model = client.load(MODEL)
dset = dataset_by_tag(model, "dset1")
C_energy = np.loadtxt(OUT / "capacitance_maxwell_fine.csv", delimiter=",", skiprows=1, usecols=range(1, 9))
Q = np.zeros((8, 8))
for excitation in range(1, 9):
    model.parameter("case_id", str(excitation))
    model.java.study("stdC2").run()
    for conductor in range(1, 9):
        Q[conductor - 1, excitation - 1] = scalar(
            model, f"intQ{conductor}(-es.nD*2*pi*r)", dset
        )
if np.linalg.norm(-Q - C_energy) < np.linalg.norm(Q - C_energy):
    Q = -Q
diff = Q - C_energy
fro = float(np.linalg.norm(diff) / np.linalg.norm(C_energy))
scale = np.maximum(np.abs(C_energy), np.finfo(float).tiny)
worst = float(np.max(np.abs(diff) / scale))
write_matrix(OUT / "capacitance_charge_fine.csv", Q)
write_matrix(OUT / "capacitance_charge_minus_energy.csv", diff)
with (OUT / "charge_method_validation.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["relative_frobenius_error", f"{fro:.15e}"])
    w.writerow(["worst_elementwise_relative_error", f"{worst:.15e}"])
    w.writerow(["axisymmetric_boundary_weight", "2*pi*r"])
summary_path = OUT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["charge_validation"] = {
    "relative_frobenius_error": fro,
    "worst_elementwise_relative_error": worst,
    "axisymmetric_boundary_weight": "2*pi*r",
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
model.save(REV)
with (ROOT / "logs" / "charge_axisym_repair_log.txt").open("w", encoding="utf-8") as f:
    f.write(f"source={MODEL}\nrevision={REV}\nrelative_frobenius_error={fro:.15e}\n")
    f.write(f"worst_elementwise_relative_error={worst:.15e}\naxisymmetric_boundary_weight=2*pi*r\n")
print(json.dumps(summary["charge_validation"]))
