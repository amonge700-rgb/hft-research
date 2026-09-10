import sys, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from run_exp19 import EdgeGate, make_case, hard_gate, evaluate

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
gen=torch.Generator(device=device).manual_seed(19)
for n in (4,6,8):
    c=make_case(n,"test",gen,device); m=EdgeGate().to(device=device,dtype=torch.float64)
    g=hard_gate(m(c.features),.4); out=evaluate(c,g)
    assert g.shape==(n,n) and abs(float(g.mean())-.4)<.08
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())
print("EXP19 smoke tests passed on",device)
