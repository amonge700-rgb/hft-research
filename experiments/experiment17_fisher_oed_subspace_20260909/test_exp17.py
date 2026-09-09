from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
m = json.loads((root / "results" / "metrics.json").read_text(encoding="utf-8"))
assert m["synthetic_only"] is True
assert m["dab_windows_used"] is False
assert len(m["oed_frequency_indices"]) == 6
assert len(set(m["oed_frequency_indices"])) == 6
assert len(m["fisher_selected_parameters"]) == 4
assert m["diagnostics"]["all_ports_all_frequencies"]["rank"] == 8
z = np.load(root / "results" / "fisher_intermediates.npz")
assert z["jacobian_raw"].shape == (72, 8)
assert z["lowrank_basis"].shape == (8, 4)
print("EXP-017 result checks passed")
