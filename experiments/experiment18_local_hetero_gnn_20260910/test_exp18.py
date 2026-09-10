from pathlib import Path
import json

root=Path(__file__).resolve().parent
m=json.loads((root/"results"/"residual"/"metrics.json").read_text(encoding="utf-8"))
assert m["synthetic_only"] is True
assert m["samples"]["test"]>0
assert m["topologies"]==["4+4","6+6","8+8"]
assert len(m["relations"])==4
assert m["teacher_checks"]["max_reciprocity_error"]<1e-10
assert m["teacher_checks"]["max_internal_kcl_residual"]<1e-8
for k in ("falcon_homogeneous","hft_typed","edge_centric_hetero"):
    assert 0<=m[k]["peak_node_hit_rate"]<=1
    assert len(m[k]["by_topology"])==3
print("EXP-018 checks passed")
