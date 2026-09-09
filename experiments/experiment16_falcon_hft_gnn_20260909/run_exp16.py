"""EXP-016: FALCON-style and HFT-typed GNN pilot with a physics closure.

Synthetic EXP-015 4+4/8+8 matrices are deliberately used as a method-chain
test.  They are not experimental transformer measurements.
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

EXPERIMENTS = Path(__file__).resolve().parents[1]
EXP15 = EXPERIMENTS / "experiment15_hft_graph_physics_20260901"
sys.path.insert(0, str(EXP15))
from hft_graph.converters import MatrixBundle, graph_to_matrices
from hft_graph.reference_models import make_reference_graph
from physics.kron_solver import solve_sweep


PARAM_NAMES = ["R_p", "R_s", "L_leak", "L_common", "C_long_p",
               "C_long_s", "C_ground", "C_ps"]


def seed_all(seed=1609):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@dataclass
class Template:
    n: int
    bundle: MatrixBundle
    leak: torch.Tensor
    common: torch.Tensor
    c_parts: list[torch.Tensor]
    node_static: torch.Tensor


def _cap_matrix(graph, kind, dtype, device):
    import copy
    g = copy.deepcopy(graph)
    g.capacitance_edges = [e for e in g.capacitance_edges if e.subtype == kind]
    return graph_to_matrices(g, dtype=dtype, device=device).C


def make_template(n, device, dtype=torch.float64):
    graph = make_reference_graph(n, dtype=dtype, device=device)
    b = graph_to_matrices(graph, dtype=dtype, device=device)
    common = b.L - torch.diag(torch.diag(b.L))
    # The reference factory adds the common diagonal to each self term.  Its
    # rank-one common matrix is reconstructed from the off-diagonal entries.
    ratio = 4.0
    flux = torch.cat((torch.ones(n, dtype=dtype, device=device),
                      -torch.ones(n, dtype=dtype, device=device) / ratio))
    common_full = (220e-6 / n**2) * torch.outer(flux, flux)
    leak = torch.diag(torch.diag(b.L - common_full))
    c_lp = _cap_matrix(graph, "longitudinal", dtype, device)
    # Split longitudinal capacitance by winding node block.
    maskp = torch.zeros_like(c_lp); maskp[:n, :n] = 1
    masks = torch.zeros_like(c_lp); masks[n:, n:] = 1
    c_ground = _cap_matrix(graph, "ground", dtype, device)
    c_ps = _cap_matrix(graph, "interwinding", dtype, device)
    pos = torch.arange(n, dtype=dtype, device=device) / max(n-1, 1)
    node_static = torch.stack((torch.cat((torch.zeros(n,device=device,dtype=dtype),
                                          torch.ones(n,device=device,dtype=dtype))),
                               torch.cat((pos,pos)),
                               torch.cat((torch.nn.functional.one_hot(torch.tensor(0),2).to(device=device,dtype=dtype).repeat(n,1)[:,0],
                                          torch.nn.functional.one_hot(torch.tensor(1),2).to(device=device,dtype=dtype).repeat(n,1)[:,1]))), dim=1)
    return Template(n,b,leak,common_full,[c_lp*maskp,c_lp*masks,c_ground,c_ps],node_static)


def scaled_bundle(t: Template, z: torch.Tensor):
    """Positive log-scales preserve R>0, L SPD and C PSD."""
    s = torch.exp(z)
    n=t.n; R=t.bundle.R.clone()
    R[:n,:n] *= s[0]; R[n:,n:] *= s[1]
    L = s[2]*t.leak + s[3]*t.common
    C = sum(si*ci for si,ci in zip(s[4:],t.c_parts))
    return MatrixBundle(t.bundle.A,R,L,C,t.bundle.G,t.bundle.node_order,
                        t.bundle.segment_order,t.bundle.external_indices,t.bundle.metadata)


def response(t, z, freqs):
    y=solve_sweep(freqs, scaled_bundle(t,z)).port_admittance
    vals=torch.stack((y[:,0,0],y[:,1,1],y[:,0,1]),1)
    return torch.cat((torch.log10(torch.abs(vals).clamp_min(1e-18)), torch.angle(vals)),1)


def graph_tensors(t, z):
    b=scaled_bundle(t,z); n=t.n; N=2*n
    r=torch.diag(b.R); ld=torch.diag(b.L)
    node=torch.cat((t.node_static, torch.log10(r[:,None]), torch.log10(ld[:,None])),1).float()
    # Three weighted relations: physical series adjacency, magnetic coupling,
    # and electric coupling. Values are normalized per graph for conditioning.
    series=torch.zeros((N,N),dtype=b.R.dtype,device=b.R.device)
    for off in (0,n):
        for k in range(n-1): series[off+k,off+k+1]=series[off+k+1,off+k]=1
    mag=torch.abs(b.L); mag.fill_diagonal_(0); mag/=mag.max().clamp_min(1e-30)
    cap=torch.abs(b.C); cap.fill_diagonal_(0); cap/=cap.max().clamp_min(1e-30)
    return node,torch.stack((series,mag,cap),0).float()


class RelationMP(nn.Module):
    def __init__(self,d=64,layers=3,hetero=True):
        super().__init__(); self.hetero=hetero
        self.inp=nn.Linear(5,d); self.selfs=nn.ModuleList([nn.Linear(d,d) for _ in range(layers)])
        self.rels=nn.ModuleList([nn.ModuleList([nn.Linear(d,d) for _ in range(3 if hetero else 1)]) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.freq=nn.Sequential(nn.Linear(1,d),nn.SiLU(),nn.Linear(d,d))
        pool_mult=5 if hetero else 1
        self.head=nn.Sequential(nn.Linear(pool_mult*d+d,128),nn.SiLU(),nn.Linear(128,6))
    def forward(self,node,adj,logf):
        h=torch.nn.functional.silu(self.inp(node))
        for li,(sl,norm) in enumerate(zip(self.selfs,self.norms)):
            msgs=[]
            if self.hetero:
                for r in range(3):
                    a=adj[:,r]; a=a/(a.sum(-1,keepdim=True)+1e-6)
                    msgs.append(self.rels[li][r](torch.bmm(a,h)))
            else:
                a=adj.sum(1); a=a/(a.sum(-1,keepdim=True)+1e-6)
                msgs.append(self.rels[li][0](torch.bmm(a,h)))
            h=norm(h+torch.nn.functional.silu(sl(h)+sum(msgs)))
        if self.hetero:
            n=h.shape[1]//2
            pooled=torch.cat((h.mean(1),h[:,:n].mean(1),h[:,n:].mean(1),h[:,0],h[:,n]),1)
        else: pooled=h.mean(1)
        B,K=logf.shape
        g=pooled[:,None,:].expand(-1,K,-1); fe=self.freq(logf[:,:,None])
        return self.head(torch.cat((g,fe),-1))


def make_dataset(templates, count, freqs, device, spread=.22):
    rows=[]
    for i in range(count):
        t=templates[i%len(templates)]
        z=torch.empty(8,dtype=torch.float64,device=device).uniform_(-spread,spread)
        node,adj=graph_tensors(t,z); y=response(t,z,freqs).float()
        rows.append((t.n,z.float(),node,adj,y))
    return rows


def collate(rows, device):
    # Train topology-homogeneous mini-batches; caller groups by n.
    return (torch.stack([r[2] for r in rows]).to(device),
            torch.stack([r[3] for r in rows]).to(device),
            torch.stack([r[4] for r in rows]).to(device))


def train(model, train_rows, val_rows, logf, epochs, batch, device):
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    best=None; best_loss=float("inf"); history=[]
    for ep in range(epochs):
        model.train(); random.shuffle(train_rows); losses=[]
        for n in (4,8):
            subset=[r for r in train_rows if r[0]==n]
            for k in range(0,len(subset),batch):
                x,a,y=collate(subset[k:k+batch],device); pred=model(x,a,logf.expand(len(x),-1))
                loss=((pred-y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); vl=[]
        with torch.no_grad():
            for n in (4,8):
                subset=[r for r in val_rows if r[0]==n]
                for k in range(0,len(subset),batch):
                    x,a,y=collate(subset[k:k+batch],device); vl.append(((model(x,a,logf.expand(len(x),-1))-y)**2).mean().item())
        vm=float(np.mean(vl)); history.append([ep+1,float(np.mean(losses)),vm])
        if vm<best_loss: best_loss=vm; best={k:v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(best); return history


def evaluate(model, rows, logf, device):
    model.eval(); errs=[]; rec=[]
    with torch.no_grad():
        for n in (4,8):
            subset=[r for r in rows if r[0]==n]
            if not subset: continue
            x,a,y=collate(subset,device); p=model(x,a,logf.expand(len(x),-1))
            e=(p-y).abs(); errs.append(e.flatten())
            rec.append({"n":n,"mae":e.mean().item(),"mag_mae_decade":e[:,:,:3].mean().item(),"phase_mae_rad":e[:,:,3:].mean().item()})
    return torch.cat(errs).mean().item(),rec


def exact_inverse(t,target,freqs,steps=240):
    z=nn.Parameter(torch.zeros(8,dtype=torch.float64,device=freqs.device)); opt=torch.optim.Adam([z],lr=.035)
    for _ in range(steps):
        pred=response(t,z,freqs); loss=((pred-target)**2).mean()+1e-5*(z*z).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return z.detach(),loss.item()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--quick",action="store_true"); ap.add_argument("--device",default="cuda")
    args=ap.parse_args(); seed_all()
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    out=Path(__file__).resolve().parent/"results"; out.mkdir(exist_ok=True)
    freqs=torch.logspace(4,7,12,dtype=torch.float64,device=device); logf=torch.log10(freqs).float()[None]
    templates=[make_template(4,device),make_template(8,device)]
    counts=(160,40,60) if args.quick else (900,180,240); epochs=12 if args.quick else 70
    data=make_dataset(templates,sum(counts),freqs,device)
    tr,va,te=data[:counts[0]],data[counts[0]:sum(counts[:2])],data[sum(counts[:2]):]
    metrics={"device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
             "synthetic_only":True,"samples":{"train":len(tr),"val":len(va),"test":len(te)},"frequencies_hz":freqs.cpu().tolist()}
    histories={}
    for name,hetero in (("falcon_homogeneous",False),("hft_typed",True)):
        m=RelationMP(hetero=hetero).to(device); t0=time.time()
        hist=train(m,tr,va,logf,epochs,32,device); mae,by=evaluate(m,te,logf,device)
        metrics[name]={"normalized_response_mae":mae,"by_topology":by,"train_seconds":time.time()-t0,"parameters":sum(p.numel() for p in m.parameters())}
        histories[name]=hist; torch.save(m.state_dict(),out/f"{name}.pt")
    # Exact structured inversion: held-out 8+8 sample and reciprocity/passivity diagnostics.
    sample=next(r for r in te if r[0]==8); true_z=sample[1].double(); target=sample[4].double()
    zh,loss=exact_inverse(templates[1],target,freqs)
    sol=solve_sweep(freqs,scaled_bundle(templates[1],zh)).port_admittance
    herm=(sol-sol.transpose(-1,-2)).abs().max().item()
    mineig=min(torch.linalg.eigvalsh((y+y.conj().T).real/2).min().item() for y in sol)
    jac=torch.autograd.functional.jacobian(lambda q: response(templates[1],q,freqs).reshape(-1),true_z)
    sv=torch.linalg.svdvals(jac); cond=(sv.max()/sv.min()).item()
    scale_rel=(torch.exp(zh-true_z)-1).abs()*100
    metrics["physics_inverse"]={"loss":loss,"mean_abs_log_scale_error":(zh-true_z).abs().mean().item(),
        "mean_absolute_scale_error_percent":scale_rel.mean().item(),
        "per_parameter_scale_error_percent":dict(zip(PARAM_NAMES,scale_rel.cpu().tolist())),
        "jacobian_singular_values":sv.cpu().tolist(),"jacobian_condition_number":cond,
        "true_scales":torch.exp(true_z).cpu().tolist(),"estimated_scales":torch.exp(zh).cpu().tolist(),
        "reciprocity_max_abs":herm,"minimum_real_admittance_eigenvalue":mineig}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    for name,hist in histories.items():
        with (out/f"history_{name}.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f); w.writerow(["epoch","train_mse","val_mse"]); w.writerows(hist)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4,4.0))
        for name,hist in histories.items():
            a=np.asarray(hist); plt.semilogy(a[:,0],a[:,2],label=name.replace("_"," "))
        plt.xlabel("Epoch"); plt.ylabel("Validation MSE"); plt.grid(True,which="both",alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(out/"training_curves.png",dpi=220); plt.close()
        plt.figure(figsize=(7.2,4.0)); x=np.arange(len(PARAM_NAMES)); width=.36
        plt.bar(x-width/2,torch.exp(true_z).cpu(),width,label="true")
        plt.bar(x+width/2,torch.exp(zh).cpu(),width,label="identified")
        plt.xticks(x,PARAM_NAMES,rotation=30,ha="right"); plt.ylabel("Scale factor"); plt.legend(); plt.tight_layout()
        plt.savefig(out/"physics_inverse_scales.png",dpi=220); plt.close()
        plt.figure(figsize=(6.4,4.0)); plt.semilogy(range(1,len(sv)+1),sv.cpu(),"o-")
        plt.xlabel("Singular-value index"); plt.ylabel("Jacobian singular value"); plt.grid(True,which="both",alpha=.25); plt.tight_layout()
        plt.savefig(out/"identifiability_spectrum.png",dpi=220); plt.close()
    except Exception as exc:
        metrics["plot_warning"]=str(exc)
        (out/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(metrics,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
