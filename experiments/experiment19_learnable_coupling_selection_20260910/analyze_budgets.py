"""Rate-distortion sweep for the trained EXP-019 edge scorer."""
import csv, json, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from run_exp19 import EdgeGate, make_case, fisher_proxy, hard_gate, evaluate, seed_all

def main():
    seed=1910; seed_all(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64); gen=torch.Generator(device=device).manual_seed(seed)
    train=[make_case((4,6,8)[i%3],"train",gen,device) for i in range(150)]
    test=[make_case((4,6,8)[i%3],"test",gen,device) for i in range(75)]
    model=EdgeGate().to(device=device,dtype=torch.float64)
    model.load_state_dict(torch.load(ROOT/"results"/"learned_edge_gate.pt",map_location=device,weights_only=True)); model.eval()
    fisher={n:fisher_proxy(next(c for c in train if c.n==n)) for n in (4,6,8)}
    rows=[]
    for keep in (.2,.4,.6,.8):
        for ci,c in enumerate(test):
            scores={"distance":-torch.abs(c.features[...,0]-c.features[...,1]),
                    "magnitude":c.cps,"fisher_proxy":fisher[c.n],
                    "oracle_edge_current":c.edge_importance,"learned_gate":model(c.features).detach()}
            for name,s in scores.items(): rows.append({"budget":keep,"case":ci,"n":c.n,"method":name,**evaluate(c,hard_gate(s,keep))})
    keys=["port_response_mae","node_voltage_mae_norm","displacement_current_error_percent","peak_voltage_error_percent","peak_location_distance_norm","edge_retention"]
    summary={}
    for keep in (.2,.4,.6,.8):
        summary[str(keep)]={}
        for method in sorted({r["method"] for r in rows}):
            rr=[r for r in rows if r["budget"]==keep and r["method"]==method]
            summary[str(keep)][method]={k:{"mean":float(np.mean([r[k] for r in rr])),"p95":float(np.percentile([r[k] for r in rr],95))} for k in keys}
    out=ROOT/"results"
    with (out/"budget_sweep.json").open("w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    with (out/"budget_sweep.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import matplotlib.pyplot as plt
    methods=["distance","magnitude","fisher_proxy","learned_gate","oracle_edge_current"]
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key,title in zip(axs,["port_response_mae","node_voltage_mae_norm","displacement_current_error_percent"],
                            ["Port response distortion","Internal-voltage distortion","Displacement-current error (%)"]):
        for m in methods:
            ax.plot([20,40,60,80],[summary[str(k)][m][key]["mean"] for k in (.2,.4,.6,.8)],marker="o",label=m)
        ax.set_xlabel("Retained Cps edges (%)"); ax.set_title(title); ax.grid(True,alpha=.3)
    axs[0].set_yscale("log"); axs[1].set_yscale("log"); axs[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/"rate_distortion_curve.png",dpi=190); plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
