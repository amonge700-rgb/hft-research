import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from run_exp26a import make_scalable_case, hard_gate, metrics

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_default_dtype(torch.float64)
    g=torch.Generator(device=device).manual_seed(7); c=make_scalable_case(4,g,device)
    full=metrics(c,torch.ones_like(c.c_edges),torch.ones_like(c.m_edges),1)
    assert full['port_error_pct']<1e-8 and full['node_voltage_error_pct']<1e-8
    assert len(c.c_edges)==16 and len(c.m_edges)==28
    x=hard_gate(c.c_importance,.5); assert int(x.sum())==8
    print({'device':str(device),'full':full,'C_edges':len(c.c_edges),'M_edges':len(c.m_edges)})

if __name__=='__main__': main()
