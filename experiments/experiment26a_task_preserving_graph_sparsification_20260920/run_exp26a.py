"""EXP-026A: task-preserving sparsification of turn-level C/M coupling graphs.

This experiment is deliberately separated from online inverse identification.
It asks how many capacitive and magnetic coupling edges can be removed while
preserving port admittance, internal turn voltages, and coupling-current stress.

The scalable cases are matrix-physics stress tests, not new COMSOL samples.
The existing EXP-020/021 4+4 COMSOL teachers are used as an independent anchor.
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
EXP15 = ROOT.parent / "experiment15_hft_graph_physics_20260901"
EXP18 = ROOT.parent / "experiment18_local_hetero_gnn_20260910"
EXP21 = ROOT.parent / "experiment21_comsol_gnn_minimal_20260916"
sys.path[:0] = [str(EXP15), str(EXP18), str(EXP21)]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

from hft_graph.converters import MatrixBundle
from physics.kron_solver import solve_sweep
from physics.internal_states import recover_internal_states
from run_exp18 import base_matrices, stamp_pair

HARM = torch.tensor([1., 3., 5., 7., 9.])


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@dataclass
class Case:
    n: int
    base: MatrixBundle
    c_fixed: torch.Tensor
    c_edges: torch.Tensor
    c_pairs: torch.Tensor
    m_edges: torch.Tensor
    m_pairs: torch.Tensor
    c_feat: torch.Tensor
    m_feat: torch.Tensor
    freqs: torch.Tensor
    portv: torch.Tensor
    full_y: torch.Tensor | None = None
    full_v: torch.Tensor | None = None
    full_branch_i: torch.Tensor | None = None
    c_importance: torch.Tensor | None = None
    m_importance: torch.Tensor | None = None
    source: str = "scalable_matrix_physics"


def pair_tensor(pairs, device):
    return torch.tensor(pairs, dtype=torch.long, device=device)


def make_scalable_case(n: int, gen: torch.Generator, device, split="train") -> Case:
    dtype=torch.float64; base=base_matrices(n, device); N=2*n
    z=torch.linspace(0,1,n,dtype=dtype,device=device)
    zp=torch.cat((z,z + .03*torch.randn((),generator=gen,device=device,dtype=dtype)))
    winding=torch.cat((torch.zeros(n,device=device,dtype=dtype),torch.ones(n,device=device,dtype=dtype)))

    # Cross-winding capacitance candidates. Remove the reference diagonal Cps
    # inserted by the synthetic base model, then stamp a dense, nonuniform map.
    c_fixed=base.C.clone(); cv=torch.tensor(28.36e-12/n,dtype=dtype,device=device)
    for k in range(n): stamp_pair(c_fixed,k,n+k,-cv)
    cpairs=[(i,n+j) for i in range(n) for j in range(n)]
    cpair=pair_tensor(cpairs,device); dz=(z[:,None]-z[None,:]).abs()
    decay=.12+.28*torch.rand((),generator=gen,device=device,dtype=dtype)
    cm=torch.exp(-dz/decay)
    ci=torch.rand((),generator=gen,device=device,dtype=dtype); cj=torch.rand((),generator=gen,device=device,dtype=dtype)
    width=.10+.22*torch.rand((),generator=gen,device=device,dtype=dtype)
    patch=torch.exp(-((z[:,None]-ci)/width)**2-((z[None,:]-cj)/width)**2)
    cm=cm*torch.exp(.65*torch.randn((),generator=gen,device=device,dtype=dtype)*patch)
    total=28.36e-12*torch.exp(.20*torch.randn((),generator=gen,device=device,dtype=dtype)); cm=cm/cm.sum()*total
    cedge=cm.reshape(-1)

    # All mutual-inductance pairs are candidates. Preserve diagonal self L.
    mpairs=[(i,j) for i in range(N) for j in range(i+1,N)]
    mpair=pair_tensor(mpairs,device); medge=base.L[mpair[:,0],mpair[:,1]].clone()

    def feats(pair, values, relation):
        i,j=pair[:,0],pair[:,1]; d=(zp[i]-zp[j]).abs(); same=(winding[i]==winding[j]).to(dtype)
        adj=(same*(torch.abs((i%n)-(j%n))==1).to(dtype)); cross=1-same
        rel=torch.log(values.abs().clamp_min(1e-18)/values.abs().mean().clamp_min(1e-18))
        return torch.stack((zp[i],zp[j],d,same,adj,cross,rel,
                            torch.full_like(d,n/32),torch.full_like(d,relation)),1)
    cfeat=feats(cpair,cedge,0.); mfeat=feats(mpair,medge,1.)

    fs=(16e3+12e3*torch.rand((),generator=gen,device=device,dtype=dtype)); h=HARM.to(device=device,dtype=dtype)
    freqs=fs*h; phi=(8+38*torch.rand((),generator=gen,device=device,dtype=dtype))*math.pi/180
    kappa=.85+.30*torch.rand((),generator=gen,device=device,dtype=dtype); tr=20e-9+130e-9*torch.rand((),generator=gen,device=device,dtype=dtype)
    env=torch.sinc(freqs*tr); vp=4*400/(math.pi*h)*env; vs=4*(100*kappa)/(math.pi*h)*env*torch.exp(-1j*h*phi)
    portv=torch.stack((vp.to(torch.complex128),vs.to(torch.complex128)),1)
    c=Case(n,base,c_fixed,cedge,cpair,medge,mpair,cfeat,mfeat,freqs,portv,source=f"scalable_{split}")
    solve_full(c)
    return c


def build_bundle(case: Case, gc: torch.Tensor, gm: torch.Tensor) -> MatrixBundle:
    c=case.c_fixed.clone()
    for e,(i,j) in enumerate(case.c_pairs.tolist()): stamp_pair(c,i,j,case.c_edges[e]*gc[e])
    l=torch.diag(torch.diagonal(case.base.L)).clone()
    for e,(i,j) in enumerate(case.m_pairs.tolist()): l[i,j]=l[j,i]=case.m_edges[e]*gm[e]
    # A tiny diagonal correction preserves positive definiteness after arbitrary
    # edge deletion. This correction is reported as part of the approximation.
    mineig=torch.linalg.eigvalsh((l+l.T)/2)[0]
    l=l+torch.relu(torch.tensor(1e-12,device=l.device,dtype=l.dtype)-mineig)*torch.eye(l.shape[0],device=l.device,dtype=l.dtype)
    b=case.base
    return MatrixBundle(b.A,b.R,l,c,b.G,b.node_order,b.segment_order,b.external_indices,{"source":"EXP26A gated C/M"})


def solve_case(case: Case, gc: torch.Tensor, gm: torch.Tensor):
    bundle=build_bundle(case,gc,gm); sol=solve_sweep(case.freqs,bundle); st=recover_internal_states(sol,case.portv)
    cdtype=sol.port_admittance.dtype
    rhs=bundle.A.T.to(cdtype)[None]@st["node_voltage"].unsqueeze(-1)
    branch_i=torch.linalg.solve(sol.branch_impedance,rhs).squeeze(-1)
    return sol.port_admittance,st["node_voltage"],branch_i


def capacitive_edge_current(case: Case, node_v: torch.Tensor, gc: torch.Tensor):
    """Return complex displacement current on every candidate C edge.

    Deleted edges remain in the returned tensor with zero current.  Keeping a
    common edge index is essential: a small port-response error must not hide
    the loss or relocation of local interwinding displacement current.
    """
    dv=node_v[:,case.c_pairs[:,0]]-node_v[:,case.c_pairs[:,1]]
    omega=(2*math.pi*case.freqs).to(node_v.real.dtype)[:,None]
    return 1j*omega*case.c_edges[None,:]*gc[None,:]*dv


def solve_full(case: Case):
    gc=torch.ones_like(case.c_edges); gm=torch.ones_like(case.m_edges)
    y,v,bi=solve_case(case,gc,gm); case.full_y=y.detach(); case.full_v=v.detach(); case.full_branch_i=bi.detach()
    ci=[]; mi=[]
    for k,f in enumerate(case.freqs):
        dv=v[k,case.c_pairs[:,0]]-v[k,case.c_pairs[:,1]]
        ci.append(torch.abs(1j*2*math.pi*f*case.c_edges*dv))
        # Mutual voltage contribution is used as a physically interpretable
        # magnetic-edge burden proxy.
        ii=bi[k,case.m_pairs[:,0]]; ij=bi[k,case.m_pairs[:,1]]
        mi.append((2*math.pi*f*case.m_edges.abs())*torch.sqrt((ii.abs()**2+ij.abs()**2)/2))
    case.c_importance=torch.sqrt(torch.mean(torch.stack(ci)**2,0)).detach()
    case.m_importance=torch.sqrt(torch.mean(torch.stack(mi)**2,0)).detach()


class DualGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)


def hard_gate(score,keep):
    k=max(1,round(float(keep)*score.numel())); ids=torch.topk(score,k).indices; g=torch.zeros_like(score); g[ids]=1; return g


def metrics(case,gc,gm,repeats=2):
    t=[]
    for _ in range(repeats):
        t0=time.perf_counter(); y,v,bi=solve_case(case,gc,gm)
        if y.is_cuda: torch.cuda.synchronize()
        t.append(time.perf_counter()-t0)
    ey=float(torch.linalg.norm(y-case.full_y)/torch.linalg.norm(case.full_y)*100)
    ev=float(torch.linalg.norm(v-case.full_v)/torch.linalg.norm(case.full_v)*100)
    ei=float(torch.linalg.norm(bi-case.full_branch_i)/torch.linalg.norm(case.full_branch_i)*100)
    ic=capacitive_edge_current(case,v,gc)
    ic_full=capacitive_edge_current(case,case.full_v,torch.ones_like(case.c_edges))
    eic=float(torch.linalg.norm(ic-ic_full)/torch.linalg.norm(ic_full).clamp_min(1e-30)*100)
    return dict(port_error_pct=ey,node_voltage_error_pct=ev,branch_current_error_pct=ei,
                displacement_current_error_pct=eic,
                solve_ms=1000*float(np.median(t)),c_keep=float(gc.mean()),m_keep=float(gm.mean()))


def train_gate(cases,relation,epochs,device):
    model=DualGate().to(device=device,dtype=torch.float64); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-5)
    hist=[]
    for ep in range(epochs):
        random.shuffle(cases); loss_sum=0
        for c in cases:
            x=c.c_feat if relation=='C' else c.m_feat; imp=c.c_importance if relation=='C' else c.m_importance
            target=torch.log(imp/imp.median().clamp_min(1e-18)+1e-12)
            pred=model(x); loss=nn.functional.smooth_l1_loss(pred,target)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); opt.step(); loss_sum+=float(loss.detach())
        hist.append(loss_sum/len(cases))
    return model,hist


def rank_scores(case,method,mc,mm):
    if method=='distance': return -case.c_feat[:,2],-case.m_feat[:,2]
    if method=='magnitude': return case.c_edges.abs(),case.m_edges.abs()
    if method=='physics_proxy': return case.c_importance,case.m_importance
    if method=='learned_gate': return mc(case.c_feat).detach(),mm(case.m_feat).detach()
    raise KeyError(method)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--train-per-size',type=int,default=5); ap.add_argument('--test-per-size',type=int,default=3); ap.add_argument('--quick',action='store_true'); ap.add_argument('--seed',type=int,default=2601)
    a=ap.parse_args(); seed_all(a.seed)
    if a.quick: a.epochs=3; a.train_per_size=1; a.test_per_size=1
    device=torch.device(a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu'); torch.set_default_dtype(torch.float64)
    gen=torch.Generator(device=device).manual_seed(a.seed); train_sizes=[4,8,16]; test_sizes=[4,8,16,32]
    train=[make_scalable_case(n,gen,device,'train') for n in train_sizes for _ in range(a.train_per_size)]
    test=[make_scalable_case(n,gen,device,'test') for n in test_sizes for _ in range(a.test_per_size)]
    tic=time.time(); mc,hc=train_gate(train,'C',a.epochs,device); mm,hm=train_gate(train,'M',a.epochs,device); train_s=time.time()-tic
    budgets=[.4,.6,.8,.9,.95,1.0]; methods=['distance','magnitude','physics_proxy','learned_gate']; rows=[]
    for ci,c in enumerate(test):
        for method in methods:
            sc,sm=rank_scores(c,method,mc,mm)
            for b in budgets:
                sparse_c=hard_gate(sc,b); sparse_m=hard_gate(sm,b)
                scenarios={'C_only':(sparse_c,torch.ones_like(sparse_m)),
                           'M_only':(torch.ones_like(sparse_c),sparse_m),
                           'joint':(sparse_c,sparse_m)}
                for scenario,(gc,gm) in scenarios.items():
                    rows.append(dict(case=ci,n=c.n,method=method,scenario=scenario,budget=b,**metrics(c,gc,gm)))
                    print(f'n={c.n:02d} case={ci:02d} {method:13s} {scenario:6s} budget={b:.2f} port={rows[-1]["port_error_pct"]:.3g}%')
    out=ROOT/'results'; figdir=ROOT/'figures'; out.mkdir(parents=True,exist_ok=True); figdir.mkdir(exist_ok=True)
    with (out/'per_case_budget.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Aggregate with transparent mean/std/count fields.
    agg=[]
    for n in test_sizes:
      for m in methods:
       for scenario in ('C_only','M_only','joint'):
        for b in budgets:
         rr=[r for r in rows if r['n']==n and r['method']==m and r['scenario']==scenario and r['budget']==b]
         d={'n':n,'method':m,'scenario':scenario,'budget':b,'count':len(rr)}
         for k in ('port_error_pct','node_voltage_error_pct','branch_current_error_pct','displacement_current_error_pct','solve_ms'):
             d[k+'_mean']=float(np.mean([r[k] for r in rr])); d[k+'_std']=float(np.std([r[k] for r in rr]))
         agg.append(d)
    with (out/'aggregate.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)
    torch.save({'C':mc.state_dict(),'M':mm.state_dict()},out/'dual_gate.pt')
    summary={'scope':{'online_inverse':False,'new_large_comsol':False,'scalable_teacher':'matrix-physics stress test','relations':['cross-winding C','all mutual M']},
             'config':vars(a)|{'device_used':str(device),'train_seconds':train_s,'train_sizes':train_sizes,'test_sizes':test_sizes,'budgets':budgets},
             'best_at_80pct':{}}
    for n in test_sizes:
        summary['best_at_80pct'][str(n)]={}
        for scenario in ('C_only','M_only','joint'):
            cand=[x for x in agg if x['n']==n and x['budget']==.8 and x['scenario']==scenario]
            best=min(cand,key=lambda x:x['port_error_pct_mean']+x['node_voltage_error_pct_mean']+x['branch_current_error_pct_mean']+x['displacement_current_error_pct_mean'])
            summary['best_at_80pct'][str(n)][scenario]={k:best[k] for k in best}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    import matplotlib.pyplot as plt
    for metric,title in [('port_error_pct_mean','Port-admittance error (%)'),('node_voltage_error_pct_mean','Internal-voltage error (%)'),('branch_current_error_pct_mean','Inductive branch-current error (%)'),('displacement_current_error_pct_mean','Interwinding displacement-current error (%)')]:
      for scenario in ('C_only','M_only','joint'):
        fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
        for ax,n in zip(axs.flat,test_sizes):
            for m in methods:
                q=[x for x in agg if x['n']==n and x['method']==m and x['scenario']==scenario]; ax.plot([x['budget']*100 for x in q],[x[metric] for x in q],marker='o',label=m)
            ax.set_title(f'{n}+{n} turns'); ax.set_yscale('symlog',linthresh=.05); ax.grid(True,alpha=.25); ax.set_xlabel('Retained candidate edges (%)'); ax.set_ylabel(title)
        axs[0,0].legend(fontsize=8); fig.suptitle(scenario); fig.tight_layout(); fig.savefig(figdir/(metric+'_'+scenario+'.png'),dpi=190); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(hc,label='C gate'); ax.plot(hm,label='M gate'); ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Smooth-L1 ranking loss'); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'training_curve.png',dpi=190); plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
