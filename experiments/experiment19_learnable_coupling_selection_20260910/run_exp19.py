"""EXP-019A: physics-decoded learnable selection of turn-to-turn Cps edges.

This is a synthetic/model-level pilot.  It asks which cross-winding
capacitance edges can be retained under a fixed edge budget while preserving
port admittance, internal turn voltages and total interwinding displacement
current.  The exact EXP-015 matrix/Kron solver remains the decoder.
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
EXP18 = ROOT.parent / "experiment18_local_hetero_gnn_20260910"
EXP15 = ROOT.parent / "experiment15_hft_graph_physics_20260901"
sys.path.insert(0, str(EXP18)); sys.path.insert(0, str(EXP15))
from run_exp18 import base_matrices, stamp_pair
from hft_graph.converters import MatrixBundle
from physics.kron_solver import solve_sweep
from physics.internal_states import recover_internal_states

HARM = torch.tensor([1., 3., 5., 7., 9.])


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@dataclass
class Case:
    n: int
    base: MatrixBundle
    c_fixed: torch.Tensor
    cps: torch.Tensor
    freqs: torch.Tensor
    portv: torch.Tensor
    features: torch.Tensor
    full_y: torch.Tensor
    full_v: torch.Tensor
    full_i: torch.Tensor
    edge_importance: torch.Tensor
    meta: dict


def remove_reference_cross_cap(base: MatrixBundle, n: int):
    c = base.C.clone()
    cv = torch.tensor(28.36e-12 / n, dtype=c.dtype, device=c.device)
    for k in range(n): stamp_pair(c, k, n + k, -cv)
    return c


def make_bundle(case: Case, gates: torch.Tensor):
    c = case.c_fixed.clone()
    for i in range(case.n):
        for j in range(case.n): stamp_pair(c, i, case.n + j, case.cps[i, j] * gates[i, j])
    b = case.base
    return MatrixBundle(b.A, b.R, b.L, c, b.G, b.node_order, b.segment_order,
                        b.external_indices, {"source": "EXP-019 gated Cps"})


def solve_case(case: Case, gates: torch.Tensor):
    sol = solve_sweep(case.freqs, make_bundle(case, gates))
    states = recover_internal_states(sol, case.portv)
    v = states["node_voltage"]
    # Complex current crossing the isolation interface.  Orientation is p -> s.
    currents = []
    for h, f in enumerate(case.freqs):
        dv = v[h, :case.n, None] - v[h, case.n:][None, :]
        currents.append((1j * 2 * math.pi * f * case.cps * gates * dv).sum())
    irms = torch.sqrt(torch.mean(torch.abs(torch.stack(currents)) ** 2))
    return sol.port_admittance, v, irms


def make_case(n: int, split: str, gen: torch.Generator, device):
    dtype = torch.float64
    base = base_matrices(n, device)
    c_fixed = remove_reference_cross_cap(base, n)
    pos = torch.linspace(0, 1, n, dtype=dtype, device=device)
    ii, jj = torch.meshgrid(pos, pos, indexing="ij")
    decay = .25 + .35 * torch.rand((), generator=gen, device=device, dtype=dtype)
    kernel = torch.exp(-torch.abs(ii - jj) / decay)
    ci = torch.rand((), generator=gen, device=device, dtype=dtype)
    cj = torch.rand((), generator=gen, device=device, dtype=dtype)
    width = .13 + .16 * torch.rand((), generator=gen, device=device, dtype=dtype)
    patch = torch.exp(-((ii-ci)/width)**2-((jj-cj)/width)**2)
    amplitude = .55 * torch.randn((), generator=gen, device=device, dtype=dtype)
    tilt = .25 * torch.randn((), generator=gen, device=device, dtype=dtype) * (ii-jj)
    cps = kernel * torch.exp(amplitude * patch + tilt)
    total = 28.36e-12 * torch.exp(.18 * torch.randn((), generator=gen, device=device, dtype=dtype))
    cps = cps / cps.sum() * total

    if split == "test":
        phi_deg = 36 + 14 * torch.rand((), generator=gen, device=device, dtype=dtype)
        kappa = torch.where(torch.rand((), generator=gen, device=device) > .5,
                            .80+.09*torch.rand((), generator=gen, device=device, dtype=dtype),
                            1.11+.09*torch.rand((), generator=gen, device=device, dtype=dtype))
    else:
        phi_deg = 5 + 30 * torch.rand((), generator=gen, device=device, dtype=dtype)
        kappa = .90 + .20 * torch.rand((), generator=gen, device=device, dtype=dtype)
    fs = 15e3 + 15e3 * torch.rand((), generator=gen, device=device, dtype=dtype)
    tr = 50e-9 + 150e-9 * torch.rand((), generator=gen, device=device, dtype=dtype)
    h = HARM.to(device=device, dtype=dtype); freqs = fs * h
    env = torch.sinc(freqs * tr); phi = phi_deg * math.pi / 180
    vp = 4 * 400 / (math.pi * h) * env
    vs = 4 * (100*kappa) / (math.pi*h) * env * torch.exp(-1j*h*phi)
    portv = torch.stack((vp.to(torch.complex128), vs.to(torch.complex128)), 1)

    # Edge features: endpoint positions, distance, relative C, geometry kernel,
    # topology size and operating condition.  No response label is leaked.
    logrel = torch.log(cps / cps.mean())
    features = torch.stack((ii, jj, torch.abs(ii-jj), logrel, kernel,
                            torch.full_like(ii, n/8), torch.full_like(ii, phi/math.pi),
                            torch.full_like(ii, kappa-1), torch.full_like(ii, tr/200e-9),
                            torch.full_like(ii, torch.log10(fs/20e3))), -1)
    dummy = Case(n, base, c_fixed, cps, freqs, portv, features, None, None, None, None, {})
    y, v, cur = solve_case(dummy, torch.ones_like(cps))
    dummy.full_y, dummy.full_v, dummy.full_i = y.detach(), v.detach(), cur.detach()
    edge_h=[]
    for hi,f in enumerate(freqs):
        dv=v[hi,:n,None]-v[hi,n:][None,:]
        edge_h.append(torch.abs(1j*2*math.pi*f*cps*dv))
    dummy.edge_importance=torch.sqrt(torch.mean(torch.stack(edge_h)**2,0)).detach()
    dummy.meta = {"n": n, "fs_hz": float(fs), "phase_deg": float(phi_deg),
                  "kappa": float(kappa), "rise_time_s": float(tr),
                  "cps_total_pf": float(cps.sum()*1e12)}
    return dummy


class EdgeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 48), nn.SiLU(), nn.Linear(48, 48),
                                 nn.SiLU(), nn.Linear(48, 1))
    def forward(self, features): return self.net(features).squeeze(-1)


def soft_budget_gate(scores, keep, temp=.20):
    # Straight-through top-k: the physical decoder sees the same hard budget
    # during training and evaluation, while gradients use the smooth sigmoid.
    threshold = torch.quantile(scores.detach().reshape(-1), 1-keep)
    soft = torch.sigmoid((scores-threshold)/temp)
    hard = hard_gate(scores, keep)
    return hard.detach() - soft.detach() + soft


def hard_gate(scores, keep):
    flat = scores.reshape(-1); k = max(1, int(round(keep*flat.numel())))
    ids = torch.topk(flat, k).indices
    out = torch.zeros_like(flat); out[ids] = 1
    return out.reshape_as(scores)


def physics_loss(case, gates):
    y, v, cur = solve_case(case, gates)
    ly = torch.mean(torch.abs(y-case.full_y)**2/(torch.abs(case.full_y)**2+1e-12))
    lv = torch.mean(torch.abs(v-case.full_v)**2)/(torch.mean(torch.abs(case.full_v)**2)+1e-12)
    li = (torch.log(cur.clamp_min(1e-12))-torch.log(case.full_i.clamp_min(1e-12)))**2
    return ly + .7*lv + .35*li


def fisher_proxy(case: Case):
    """Diagonal Fisher/sensitivity proxy at the full graph.

    z is a log-scale on each Cps edge; the observable contains normalized
    complex port admittance and internal voltage at natural DAB harmonics.
    """
    z0 = torch.zeros_like(case.cps, requires_grad=True)
    def obs(z):
        y, v, _ = solve_case(case, torch.exp(z))
        yy = torch.view_as_real(y/case.full_y.abs().clamp_min(1e-10)).reshape(-1)
        vv = torch.view_as_real(v/400).reshape(-1)
        return torch.cat((yy, .7*vv))
    jac = torch.autograd.functional.jacobian(obs, z0, vectorize=True)
    return torch.sum(jac.reshape(jac.shape[0], -1)**2, 0).reshape(case.n, case.n).detach()


def evaluate(case, gates):
    y, v, cur = solve_case(case, gates)
    vals = torch.stack((y[:,0,0], y[:,1,1], y[:,0,1]), 1)
    ref = torch.stack((case.full_y[:,0,0], case.full_y[:,1,1], case.full_y[:,0,1]), 1)
    port = torch.cat((torch.log10(vals.abs().clamp_min(1e-18)), torch.angle(vals)), 1)
    port0 = torch.cat((torch.log10(ref.abs().clamp_min(1e-18)), torch.angle(ref)), 1)
    phase_delta = torch.atan2(torch.sin(port[:,3:]-port0[:,3:]), torch.cos(port[:,3:]-port0[:,3:]))
    port_mae = torch.cat(((port[:,:3]-port0[:,:3]).abs(), phase_delta.abs()),1).mean()
    node_mae = torch.mean(torch.abs(v-case.full_v))/400
    vrms = torch.sqrt(torch.mean(torch.abs(v)**2,0)); vrms0 = torch.sqrt(torch.mean(torch.abs(case.full_v)**2,0))
    internal = torch.ones(2*case.n, dtype=torch.bool, device=v.device); internal[[0,case.n]]=False
    idx = torch.arange(2*case.n,device=v.device)[internal]
    pidx = idx[torch.argmax(vrms[internal])]; tidx = idx[torch.argmax(vrms0[internal])]
    loc_distance = torch.abs((pidx % case.n)-(tidx % case.n))/max(1,case.n-1)
    peak_rel = torch.abs(vrms[pidx]-vrms0[tidx])/vrms0[tidx].clamp_min(1e-12)
    return {"port_response_mae":float(port_mae), "node_voltage_mae_norm":float(node_mae),
            "displacement_current_error_percent":float(torch.abs(cur-case.full_i)/case.full_i*100),
            "peak_voltage_error_percent":float(peak_rel*100),
            "peak_location_hit":float(pidx==tidx), "peak_location_distance_norm":float(loc_distance),
            "edge_retention":float(gates.mean())}


def aggregate(rows):
    groups = {}
    for r in rows:
        g = groups.setdefault(r["method"], {})
        for k,v in r.items():
            if k not in ("method","case","n"): g.setdefault(k,[]).append(v)
    return {m:{k:{"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)),
                    "p95":float(np.percentile(v,95))} for k,v in d.items()} for m,d in groups.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda")
    ap.add_argument("--train",type=int,default=150); ap.add_argument("--test",type=int,default=75)
    ap.add_argument("--epochs",type=int,default=45); ap.add_argument("--keep",type=float,default=.40)
    ap.add_argument("--seed",type=int,default=1910); ap.add_argument("--quick",action="store_true")
    args=ap.parse_args(); seed_all(args.seed)
    if args.quick: args.train,args.test,args.epochs=30,18,8
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    out=ROOT/"results"; out.mkdir(parents=True,exist_ok=True)
    gen=torch.Generator(device=device).manual_seed(args.seed)
    train=[make_case((4,6,8)[i%3],"train",gen,device) for i in range(args.train)]
    test=[make_case((4,6,8)[i%3],"test",gen,device) for i in range(args.test)]
    model=EdgeGate().to(device=device,dtype=torch.float64); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-5)
    history=[]; tic=time.time()
    for ep in range(args.epochs):
        random.shuffle(train); total=0
        temp=max(.20,.55-.35*ep/max(1,args.epochs-1))
        bce=nn.BCEWithLogitsLoss()
        for case in train:
            opt.zero_grad(); scores=model(case.features)
            # Teacher supervision answers the actual selection question: which
            # edges carry the largest full-model displacement-current burden?
            # A small differentiable physics term keeps response preservation
            # tied to the exact circuit decoder.
            target=hard_gate(case.edge_importance,args.keep)
            gates=soft_budget_gate(scores,args.keep,temp)
            loss=bce(scores,target)+.15*physics_loss(case,gates)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),3.0); opt.step(); total+=float(loss.detach())
        history.append({"epoch":ep+1,"loss":total/len(train),"temperature":temp})
        print(f"epoch {ep+1:03d} loss={history[-1]['loss']:.6g} temp={temp:.3f}")
    train_seconds=time.time()-tic

    # One nominal Fisher-diagonal proxy per topology; condition-aware learned
    # gates remain sample-specific.  All methods use the same hard edge budget.
    fisher={n:fisher_proxy(next(c for c in train if c.n==n)) for n in (4,6,8)}
    rows=[]
    for ci,case in enumerate(test):
        scores={
            "distance":-torch.abs(case.features[...,0]-case.features[...,1]),
            "magnitude":case.cps,
            "fisher_proxy":fisher[case.n],
            "oracle_edge_current":case.edge_importance,
            "learned_gate":model(case.features).detach(),
        }
        for name,score in scores.items():
            metrics=evaluate(case,hard_gate(score,args.keep))
            rows.append({"case":ci,"n":case.n,"method":name,**metrics})
    summary=aggregate(rows)
    teacher={"max_reciprocity":0.,"minimum_real_y_eigenvalue":float("inf")}
    for c in test:
        teacher["max_reciprocity"]=max(teacher["max_reciprocity"],float((c.full_y-c.full_y.transpose(-1,-2)).abs().max()))
        teacher["minimum_real_y_eigenvalue"]=min(teacher["minimum_real_y_eigenvalue"],
            min(float(torch.linalg.eigvalsh((q+q.conj().T).real/2).min()) for q in c.full_y))
    payload={"scope":{"synthetic_only":True,"teacher":"EXP-015 exact matrix/Kron solver",
             "selected_relation":"cross-winding capacitance only","online_inverse":False,
             "comsol_or_hardware":False},"config":vars(args)|{"device_used":str(device),"train_seconds":train_seconds},
             "teacher_checks":teacher,"metrics":summary}
    with (out/"metrics.json").open("w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)
    with (out/"per_case.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out/"history.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    torch.save(model.state_dict(),out/"learned_edge_gate.pt")

    import matplotlib.pyplot as plt
    methods=list(summary); tasks=["port_response_mae","node_voltage_mae_norm","displacement_current_error_percent","peak_location_distance_norm"]
    titles=["Port response MAE","Node voltage MAE / 400 V","Displacement-current error (%)","Peak-location distance"]
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for ax,key,title in zip(axs.flat,tasks,titles):
        vals=[summary[m][key]["mean"] for m in methods]; err=[summary[m][key]["std"] for m in methods]
        ax.bar(range(len(methods)),vals,yerr=err,capsize=3); ax.set_xticks(range(len(methods)),methods,rotation=20,ha="right"); ax.set_title(title); ax.grid(axis="y",alpha=.25)
    fig.suptitle(f"EXP-019A: equal {args.keep:.0%} cross-capacitance edge budget"); fig.tight_layout(); fig.savefig(out/"method_comparison.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4)); ax.plot([h["epoch"] for h in history],[h["loss"] for h in history]); ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Physics-decoded loss"); ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(out/"training_curve.png",dpi=180); plt.close(fig)
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
