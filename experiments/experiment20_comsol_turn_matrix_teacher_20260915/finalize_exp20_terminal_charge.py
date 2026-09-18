from pathlib import Path
import csv
import json
import mph
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model" / "revisions" / "terminal_charge_api_test.mph"
FINAL = ROOT / "model" / "revisions" / "exp20_teacher_4p4_axisym_C_LM_converged_terminal_verified.mph"
OUT = ROOT / "data" / "convergence"
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
model = client.load(SOURCE)
dset = dataset_by_tag(model, "dset1")
C_energy = np.loadtxt(OUT / "capacitance_maxwell_fine.csv", delimiter=",", skiprows=1, usecols=range(1, 9))
C_charge = np.zeros((8, 8))
for excitation in range(1, 9):
    model.parameter("case_id", str(excitation))
    model.java.study("stdC2").run()
    for conductor in range(1, 9):
        C_charge[conductor - 1, excitation - 1] = scalar(
            model, f"es.Q0_{conductor}", dset
        )
diff = C_charge - C_energy
fro = float(np.linalg.norm(diff) / np.linalg.norm(C_energy))
scale = np.maximum(np.abs(C_energy), np.finfo(float).tiny)
worst = float(np.max(np.abs(diff) / scale))
write_matrix(OUT / "capacitance_charge_fine.csv", C_charge)
write_matrix(OUT / "capacitance_charge_minus_energy.csv", diff)
with (OUT / "charge_method_validation.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["relative_frobenius_error", f"{fro:.15e}"])
    w.writerow(["worst_elementwise_relative_error", f"{worst:.15e}"])
    w.writerow(["method", "COMSOL Voltage Terminal reaction charge es.Q0_i"])
summary_path = OUT / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["source_model"] = "model/exp20_teacher_4p4_axisym_C_LM.mph"
summary["charge_validation"] = {
    "relative_frobenius_error": fro,
    "worst_elementwise_relative_error": worst,
    "method": "COMSOL Voltage Terminal reaction charge es.Q0_i",
    "accepted": fro < 1e-6 and worst < 1e-5,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
model.save(FINAL)
with (ROOT / "logs" / "terminal_charge_validation_log.txt").open("w", encoding="utf-8") as f:
    f.write("source=model/revisions/terminal_charge_api_test.mph\n")
    f.write("final_revision=model/revisions/exp20_teacher_4p4_axisym_C_LM_converged_terminal_verified.mph\n")
    f.write(f"relative_frobenius_error={fro:.15e}\n")
    f.write(f"worst_elementwise_relative_error={worst:.15e}\n")
    f.write("method=COMSOL Voltage Terminal reaction charge es.Q0_i\n")
print(json.dumps(summary["charge_validation"]))
