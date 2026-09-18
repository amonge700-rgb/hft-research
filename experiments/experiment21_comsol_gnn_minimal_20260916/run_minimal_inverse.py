"""EXP-021: COMSOL-teacher, multi-window, physics-constrained GNN pilot.

The experiment is deliberately small.  Each COMSOL 4+4 turn model supplies a
nominal Maxwell C matrix and an L/M matrix.  Synthetic states alter five
physically grouped coordinates.  Four DAB-like operating windows are used to
reconstruct a noisy 2-port admittance.  An edge-centric GNN estimates the five
coordinates and is trained with both parameter and differentiable circuit
residual losses.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
TEACHERS = ROOT / "data" / "comsol_teachers"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N = 8
FREQ = np.array([17e3, 51e3, 85e3, 23e3, 69e3, 115e3], dtype=float)


def seed_all(seed: int = 21) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def incidence_np() -> np.ndarray:
    a = np.zeros((N, N))
    for o in (0, 4):
        for k in range(4):
            a[o + k, o + k] = 1
            if k < 3: a[o + k + 1, o + k] = -1
    return a


def restamp_c(c0: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Scale TC, IT, WW Maxwell-capacitance classes while preserving structure."""
    m = -c0.copy(); np.fill_diagonal(m, 0)
    cref = c0.sum(1) * np.exp(theta[2])
    out = np.zeros_like(m)
    for i in range(N):
        for j in range(i + 1, N):
            same = (i < 4) == (j < 4)
            if not same: scale = math.exp(theta[1])       # WW
            elif abs((i % 4) - (j % 4)) == 1: scale = math.exp(theta[0])  # IT
            else: scale = 1.0
            out[i, j] = out[j, i] = m[i, j] * scale
    c = -out; np.fill_diagonal(c, cref + out.sum(1))
    return c


def modal_l(l0: np.ndarray, theta: np.ndarray) -> np.ndarray:
    w, u = np.linalg.eigh((l0 + l0.T) / 2)
    main = w[-1] * np.outer(u[:, -1], u[:, -1])
    leak = (u[:, :-1] * w[:-1]) @ u[:, :-1].T
    return math.exp(theta[3]) * main + math.exp(theta[4]) * leak


def solve_port_np(c: np.ndarray, l: np.ndarray, freq: np.ndarray = FREQ) -> np.ndarray:
    a = incidence_np(); r = np.eye(N) * 2.2e-3
    ports = np.array([0, 4]); internal = np.array([1, 2, 3, 5, 6, 7])
    ans = []
    for f in freq:
        z = r + 1j * 2 * np.pi * f * l
        y = a @ np.linalg.solve(z, a.T) + 1j * 2 * np.pi * f * c
        pp = y[np.ix_(ports, ports)]; pi = y[np.ix_(ports, internal)]
        ip = y[np.ix_(internal, ports)]; ii = y[np.ix_(internal, internal)]
        ans.append(pp - pi @ np.linalg.solve(ii, ip))
    return np.stack(ans)


def noisy_multiwindow_y(y: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """Generate four DAB-like voltage directions, currents, then LS-recover Y."""
    phis = np.deg2rad([8, 17, 29, 41]); ratios = np.array([.86, .96, 1.05, 1.14])
    out = []
    for k, f in enumerate(FREQ):
        h = (1, 3, 5)[k % 3]
        v = np.vstack([np.ones(4), ratios * np.exp(-1j * h * phis)])
        i = y[k] @ v
        sigma = np.linalg.norm(i) / math.sqrt(i.size) * 10 ** (-snr_db / 20)
        i += sigma / math.sqrt(2) * (rng.normal(size=i.shape) + 1j * rng.normal(size=i.shape))
        out.append(i @ v.conj().T @ np.linalg.inv(v @ v.conj().T + 1e-10*np.eye(2)))
    return np.stack(out)


def load_teachers() -> list[dict]:
    manifest = pd.read_csv(TEACHERS / "manifest.csv")
    rows = []
    for _, r in manifest.iterrows():
        p = TEACHERS / r["sample_id"]
        c = pd.read_csv(p / "C.csv", index_col=0).to_numpy(float)
        l = pd.read_csv(p / "L.csv", index_col=0).to_numpy(float)
        gps, gt, zo = float(r.g_ps_mm), float(r.g_turn_mm), float(r.z_offset_s_mm)
        z = np.array([(k-1.5)*(1+gt) for k in range(4)] +
                     [(k-1.5)*(1+gt)+zo for k in range(4)])
        x = np.array([11.5]*4 + [12.0+gps]*4)
        node = np.stack([(x-x.mean())/3, z/3, np.r_[np.zeros(4), np.ones(4)]], 1)
        rows.append(dict(sample_id=r.sample_id, gps=gps, gt=gt, zo=zo, C=c, L=l, node=node))
    return rows


def audit(teachers: list[dict]) -> dict:
    table = []
    for t in teachers:
        c=t["C"]; l=t["L"]
        it=sum(-c[i,i+1] for i in [0,1,2,4,5,6])
        ww=-c[:4,4:].sum(); cref=c.sum(1).sum()
        ev=np.linalg.eigvalsh(l)
        table.append(dict(sample_id=t["sample_id"],g_ps_mm=t["gps"],g_turn_mm=t["gt"],z_offset_mm=t["zo"],
                          IT_total_pF=it*1e12,WW_total_pF=ww*1e12,TC_total_pF=cref*1e12,
                          L_min_uH=ev[0]*1e6,L_main_uH=ev[-1]*1e6,
                          C_sym=float(np.max(abs(c-c.T))),L_sym=float(np.max(abs(l-l.T)))))
    df=pd.DataFrame(table); (RESULTS/"teacher_audit.csv").parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(RESULTS/"teacher_audit.csv",index=False)
    return {"samples":len(df),"all_finite":bool(np.isfinite(df.select_dtypes(float)).all().all()),
            "WW_gap_correlation":float(df.WW_total_pF.corr(df.g_ps_mm)),
            "IT_gap_correlation":float(df.IT_total_pF.corr(df.g_turn_mm)),
            "max_C_symmetry_F":float(df.C_sym.max()),"max_L_symmetry_H":float(df.L_sym.max())}


def make_dataset(teachers: list[dict], states_per_graph: int=50) -> list[dict]:
    rng=np.random.default_rng(2109); data=[]
    for gi,t in enumerate(teachers):
        for s in range(states_per_graph):
            theta=rng.uniform(-.16,.16,5)
            c=restamp_c(t["C"],theta); l=modal_l(t["L"],theta)
            clean=solve_port_np(c,l); snr=float(rng.uniform(34,52)); obs=noisy_multiwindow_y(clean,rng,snr)
            feat=np.concatenate([np.log10(np.abs(obs).reshape(-1)+1e-12),np.angle(obs).reshape(-1)])
            data.append(dict(graph=gi,theta=theta.astype(np.float32),measurement=feat.astype(np.float32),
                             yobs=obs.astype(np.complex64),snr=snr))
    return data


def graph_edges(t: dict) -> tuple[np.ndarray,np.ndarray]:
    src=[]; dst=[]; attr=[]
    for i in range(N):
        for j in range(N):
            if i==j: continue
            src.append(i); dst.append(j); dx=t["node"][j,0]-t["node"][i,0]; dz=t["node"][j,1]-t["node"][i,1]
            same=float((i<4)==(j<4)); adj=float(same and abs(i%4-j%4)==1)
            attr.append([dx,dz,math.hypot(dx,dz),same,adj,
                         math.log10(max(-t["C"][i,j],1e-18))+15,
                         math.log10(max(abs(t["L"][i,j]),1e-12))+7])
    return np.array([src,dst]),np.array(attr,np.float32)


class EdgeGNN(nn.Module):
    def __init__(self, meas_dim: int, hidden: int=64):
        super().__init__()
        self.meas=nn.Sequential(nn.Linear(meas_dim,128),nn.SiLU(),nn.Linear(128,hidden))
        self.node=nn.Linear(3,hidden); self.edge=nn.Linear(7,hidden)
        self.msg=nn.Sequential(nn.Linear(hidden*3,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.upd=nn.GRUCell(hidden,hidden)
        self.head=nn.Sequential(nn.Linear(hidden*2,128),nn.SiLU(),nn.Linear(128,5))
    def forward(self,node,edge_index,edge_attr,measurement):
        # node [B,N,3], edges shared per batch only inside a same-geometry mini-batch
        h=self.node(node); e=self.edge(edge_attr).unsqueeze(0).expand(node.shape[0],-1,-1)
        src,dst=edge_index
        for _ in range(3):
            m=self.msg(torch.cat([h[:,src],h[:,dst],e],-1)); agg=torch.zeros_like(h)
            agg.index_add_(1,dst,m); h=self.upd(agg.reshape(-1,h.shape[-1]),h.reshape(-1,h.shape[-1])).reshape_as(h)
        g=h.mean(1); q=self.meas(measurement)
        return .22*torch.tanh(self.head(torch.cat([g,q],-1)))


def torch_matrices(c0,l0,theta):
    b=theta.shape[0]; dtype=theta.dtype; dev=theta.device
    m=(-c0).clone(); m.fill_diagonal_(0); m=m.unsqueeze(0).expand(b,-1,-1).clone()
    scale=torch.ones((b,N,N),device=dev,dtype=dtype)
    for i in range(N):
        for j in range(i+1,N):
            same=(i<4)==(j<4)
            q=1 if not same else (0 if abs(i%4-j%4)==1 else -1)
            if q>=0: scale[:,i,j]=scale[:,j,i]=torch.exp(theta[:,q])
    mm=m*scale; cref=c0.sum(1).unsqueeze(0)*torch.exp(theta[:,2:3])
    c=-mm; idx=torch.arange(N,device=dev); c[:,idx,idx]=cref+mm.sum(2)
    w,u=torch.linalg.eigh((l0+l0.T)/2); main=w[-1]*torch.outer(u[:,-1],u[:,-1]); leak=(u[:,:-1]*w[:-1])@u[:,:-1].T
    l=torch.exp(theta[:,3,None,None])*main+torch.exp(theta[:,4,None,None])*leak
    return c,l


def torch_port(c,l):
    b=c.shape[0]; dev=c.device; rdtype=c.dtype; cdtype=torch.complex64 if rdtype==torch.float32 else torch.complex128
    a=torch.tensor(incidence_np(),device=dev,dtype=rdtype); r=torch.eye(N,device=dev,dtype=rdtype)*2.2e-3
    p=torch.tensor([0,4],device=dev); q=torch.tensor([1,2,3,5,6,7],device=dev); out=[]
    for f in FREQ:
        z=(r.unsqueeze(0)+1j*2*math.pi*f*l).to(cdtype)
        y=(a.to(cdtype).unsqueeze(0)@torch.linalg.solve(z,a.T.to(cdtype).expand(b,-1,-1))+
           1j*2*math.pi*f*c.to(cdtype))
        pp=y[:,p][:,:,p]; pi=y[:,p][:,:,q]; ip=y[:,q][:,:,p]; ii=y[:,q][:,:,q]
        out.append(pp-pi@torch.linalg.solve(ii,ip))
    return torch.stack(out,1)


def physics_loss(pred,c0,l0,yobs):
    c,l=torch_matrices(c0,l0,pred); yp=torch_port(c,l)
    scale=torch.mean(torch.abs(yobs),dim=(1,2,3),keepdim=True).clamp_min(1e-8)
    return torch.mean(torch.abs((yp-yobs)/scale)**2)


def batches_by_graph(data, graph_ids, batch_size=32, shuffle=True):
    ids=[i for i,d in enumerate(data) if d["graph"] in graph_ids]
    if shuffle: random.shuffle(ids)
    groups={}
    for i in ids: groups.setdefault(data[i]["graph"],[]).append(i)
    keys=list(groups); random.shuffle(keys)
    for g in keys:
        z=groups[g]
        for k in range(0,len(z),batch_size): yield g,[data[i] for i in z[k:k+batch_size]]


def tensors(batch,device):
    return (torch.tensor(np.stack([x["measurement"] for x in batch]),device=device),
            torch.tensor(np.stack([x["theta"] for x in batch]),device=device),
            torch.tensor(np.stack([x["yobs"] for x in batch]),device=device))


@torch.no_grad()
def evaluate(model,data,gids,teachers,device):
    rows=[]; model.eval()
    for g,batch in batches_by_graph(data,gids,64,False):
        m,t,y=tensors(batch,device); ei,ea=graph_edges(teachers[g])
        node=torch.tensor(teachers[g]["node"],dtype=torch.float32,device=device).unsqueeze(0).expand(len(batch),-1,-1)
        p=model(node,torch.tensor(ei,dtype=torch.long,device=device),torch.tensor(ea,device=device),m)
        c0=torch.tensor(teachers[g]["C"],dtype=torch.float32,device=device); l0=torch.tensor(teachers[g]["L"],dtype=torch.float32,device=device)
        yp=torch_port(*torch_matrices(c0,l0,p)); base=torch.zeros_like(p); y0=torch_port(*torch_matrices(c0,l0,base))
        for n,x in enumerate(batch):
            rows.append(dict(graph=g,sample_id=teachers[g]["sample_id"],snr_db=x["snr"],
                theta_mae=float(torch.mean(abs(p[n]-t[n])).cpu()),
                response_error=float((torch.linalg.norm(yp[n]-y[n])/torch.linalg.norm(y[n])).cpu()),
                nominal_response_error=float((torch.linalg.norm(y0[n]-y[n])/torch.linalg.norm(y[n])).cpu())))
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=500); ap.add_argument("--states",type=int,default=50)
    args=ap.parse_args(); seed_all(); RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers=load_teachers(); audit_metrics=audit(teachers); data=make_dataset(teachers,args.states)
    # Geometry-held-out split: no state from a held-out COMSOL geometry leaks into training.
    train=set(range(8)); val={8,9}; test={10,11}
    model=EdgeGNN(len(data[0]["measurement"])).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    best=1e99; best_state=None; history=[]; tic=time.time()
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for g,batch in batches_by_graph(data,train,32,True):
            m,t,y=tensors(batch,device); ei,ea=graph_edges(teachers[g]); b=len(batch)
            node=torch.tensor(teachers[g]["node"],dtype=torch.float32,device=device).unsqueeze(0).expand(b,-1,-1)
            p=model(node,torch.tensor(ei,dtype=torch.long,device=device),torch.tensor(ea,device=device),m)
            c0=torch.tensor(teachers[g]["C"],dtype=torch.float32,device=device); l0=torch.tensor(teachers[g]["L"],dtype=torch.float32,device=device)
            lp=torch.mean((p-t)**2); ly=physics_loss(p,c0,l0,y); loss=lp+0.02*ly
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); losses.append(float(loss.detach()))
        if epoch%10==0 or epoch==args.epochs-1:
            v=evaluate(model,data,val,teachers,device).theta_mae.mean(); history.append((epoch,np.mean(losses),v))
            if v<best: best=float(v); best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save(model.state_dict(),RESULTS/"edge_gnn.pt")
    testdf=evaluate(model,data,test,teachers,device); testdf.to_csv(RESULTS/"test_cases.csv",index=False)
    pd.DataFrame(history,columns=["epoch","train_loss","val_theta_mae"]).to_csv(RESULTS/"training_history.csv",index=False)
    summary={"scope":"minimal simulation-only feasibility pilot","teacher_audit":audit_metrics,
      "cuda_available":torch.cuda.is_available(),"device":str(device),"torch_version":torch.__version__,
      "train_graphs":len(train),"validation_graphs":len(val),"test_graphs":len(test),"states_per_graph":args.states,
      "trainable_parameters":sum(p.numel() for p in model.parameters()),"best_val_theta_mae_log_scale":best,
      "test_theta_mae_log_scale":float(testdf.theta_mae.mean()),
      "test_response_relative_error":float(testdf.response_error.mean()),
      "nominal_response_relative_error":float(testdf.nominal_response_error.mean()),
      "response_error_reduction_percent":float((1-testdf.response_error.mean()/testdf.nominal_response_error.mean())*100),
      "training_seconds":time.time()-tic,
      "limitations":["12 COMSOL geometries only","simulation-to-simulation","quasi-static C and linear-core L/M","measurement-chain and hardware validation absent"]}
    (RESULTS/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    # Dependency-free SVG summary.
    vals=[summary["nominal_response_relative_error"],summary["test_response_relative_error"]]; labels=["Nominal","GNN + physics"]
    w,h=640,400; mx=max(vals)*1.15; bars=[]
    for i,(lab,v) in enumerate(zip(labels,vals)):
        x=120+i*230; bh=260*v/mx; bars.append(f'<rect x="{x}" y="{330-bh:.1f}" width="130" height="{bh:.1f}" fill="{["#8884d8","#2a9d8f"][i]}"/><text x="{x+65}" y="355" text-anchor="middle">{lab}</text><text x="{x+65}" y="{315-bh:.1f}" text-anchor="middle">{v*100:.2f}%</text>')
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="320" y="35" text-anchor="middle" font-size="20">Held-out geometry response error</text><line x1="80" y1="330" x2="580" y2="330" stroke="black"/>{"".join(bars)}</svg>'
    (FIGURES/"heldout_response_error.svg").write_text(svg,encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
