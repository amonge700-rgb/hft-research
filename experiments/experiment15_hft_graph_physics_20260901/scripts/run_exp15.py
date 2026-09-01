"""Unified EXP-015 regression, gradient, diagnostics and artifact generator."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; EXPERIMENTS=ROOT.parent
sys.path[:0]=[str(ROOT),str(EXPERIMENTS/"experiment10_segmented_ladder_identification_20260721")]
from hft_graph import graph_to_matrices
from hft_graph.reference_models import make_reference_graph
from physics import solve_sweep, recover_internal_states, physical_diagnostics
from hft_segmented_ladder import LadderParameters, solve_sweep as old_sweep

def old_params_from(bundle,n):
    C=bundle.C.detach().cpu().numpy(); R=np.diag(bundle.R.detach().cpu().numpy())
    L=bundle.L.detach().cpu().numpy(); G=np.diag(bundle.G.detach().cpu().numpy())
    cseries=np.r_[[72e-12]*n,[42e-12]*n]
    ground=np.r_[np.linspace(7,4,n),np.linspace(5,3.5,n)]*1e-12
    cps=np.full(n,28.36e-12/n)
    return LadderParameters(n,4.,R,L,cseries,ground,cps,G)

def relerr(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))

def main():
    out=ROOT/"results"; out.mkdir(exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    freq=np.logspace(3,7,161); summary={"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"models":{}}
    rows=[]
    for n in (4,8):
        graph=make_reference_graph(n); bundle=graph_to_matrices(graph)
        new=solve_sweep(torch.tensor(freq,dtype=torch.float64),bundle)
        old=old_sweep(freq,old_params_from(bundle,n))["port_admittance"]
        err=relerr(new.port_admittance.detach().numpy(),old)
        states=recover_internal_states(new,torch.tensor([1+0j,0+0j],dtype=torch.complex128))
        diag=physical_diagnostics(bundle,new.port_admittance)
        kcl=float(states["internal_kcl_residual"].abs().max())
        summary["models"][f"{n}+{n}"]={"relative_port_error":err,"max_internal_kcl_a":kcl,**diag}
        for i,f in enumerate(freq): rows.append([n,f,err,float(new.port_admittance[i,0,0].real),float(new.port_admittance[i,0,0].imag)])
    # Autograd sanity: interwinding C influences high-frequency Y12.
    graph=make_reference_graph(4); target=graph.capacitance_edges[-1].capacitance_f.clone().detach().requires_grad_(True)
    graph.capacitance_edges[-1].capacitance_f=target
    y=solve_sweep(torch.tensor([1e6],dtype=torch.float64),graph_to_matrices(graph)).port_admittance
    loss=y.abs().square().sum(); loss.backward()
    summary["gradient"]={"cps_loss_gradient":float(target.grad),"finite":bool(torch.isfinite(target.grad))}
    with open(out/"summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    with open(out/"sweep.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["sections_per_winding","frequency_hz","global_relative_error","Re_Y11_S","Im_Y11_S"]); w.writerows(rows)
    # Dependency-free SVG plot.
    W,H=820,360; pts=[]
    vals=np.log10(np.maximum(np.abs(np.array([r[3]+1j*r[4] for r in rows if r[0]==4])),1e-18)); lo,hi=vals.min(),vals.max()
    for i,v in enumerate(vals): pts.append(f"{50+i*(W-80)/(len(vals)-1):.1f},{H-40-(v-lo)*(H-70)/(hi-lo):.1f}")
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="25" y="25">EXP-015 4+4 port |Y11|, 1 kHz--10 MHz</text><polyline fill="none" stroke="#1565c0" stroke-width="2" points="{" ".join(pts)}"/></svg>'
    (out/"figures"/"port_y11.svg").write_text(svg,encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
