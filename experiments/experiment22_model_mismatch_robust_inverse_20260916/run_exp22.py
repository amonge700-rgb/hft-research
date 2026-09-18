"""EXP-022: does AI help when the inverse physics model is incomplete?"""
from __future__ import annotations
import json, math, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
EXP21=ROOT.parent/"experiment21_comsol_gnn_minimal_20260916"
sys.path.insert(0,str(EXP21))
import run_minimal_inverse as base

OUT=ROOT/"results"; FIG=ROOT/"figures"
FREQ=np.array([20e3,60e3,100e3,200e3,500e3,1e6],float)
base.FREQ=FREQ

def rich_port(c,l,nuis):
    """Forward plant unavailable to the inverse estimator."""
    a=base.incidence_np(); p=np.array([0,4]); q=np.array([1,2,3,5,6,7]); ans=[]
    # Unmodelled asymmetric ground paths.
    cc=c.copy(); cc[0,0]+=nuis["cgp_pf"]*1e-12; cc[4,4]+=nuis["cgs_pf"]*1e-12
    for f in FREQ:
        r0=2.2e-3*(1+0.0039*(nuis["temp_c"]-25))
        rac=r0*(1+nuis["skin"]*math.sqrt(f/100e3))
        z=np.eye(8)*rac+1j*2*np.pi*f*l
        y=a@np.linalg.solve(z,a.T)+1j*2*np.pi*f*cc
        pp=y[np.ix_(p,p)]; pi=y[np.ix_(p,q)]; ip=y[np.ix_(q,p)]; ii=y[np.ix_(q,q)]
        ans.append(pp-pi@np.linalg.solve(ii,ip))
    return np.stack(ans)

def observe(y,nuis,rng):
    phis=np.deg2rad([7,18,31,43]); ratios=np.array([.84,.95,1.06,1.16]); out=[]
    for k,f in enumerate(FREQ):
        h=[1,3,5,9,19,39][k]
        v=np.vstack([np.ones(4),ratios*np.exp(-1j*h*phis)])
        i=y[k]@v
        # Independent voltage/current acquisition chains; inverse code ignores them.
        dv=np.diag([(1+nuis["vg1"])*np.exp(-1j*2*np.pi*f*nuis["vt1_ns"]*1e-9),
                    (1+nuis["vg2"])*np.exp(-1j*2*np.pi*f*nuis["vt2_ns"]*1e-9)])
        di=np.diag([(1+nuis["ig1"])*np.exp(-1j*2*np.pi*f*nuis["it1_ns"]*1e-9),
                    (1+nuis["ig2"])*np.exp(-1j*2*np.pi*f*nuis["it2_ns"]*1e-9)])
        vm=dv@v; im=di@i
        sigma=np.linalg.norm(im)/math.sqrt(im.size)*10**(-nuis["snr_db"]/20)
        im+=sigma/math.sqrt(2)*(rng.normal(size=im.shape)+1j*rng.normal(size=im.shape))
        out.append(im@vm.conj().T@np.linalg.inv(vm@vm.conj().T+1e-10*np.eye(2)))
    return np.stack(out)

def nuisance(rng):
    d=dict(temp_c=float(rng.uniform(25,85)),skin=float(rng.uniform(.05,.28)),
      cgp_pf=float(rng.uniform(0,4)),cgs_pf=float(rng.uniform(0,4)),snr_db=float(rng.uniform(34,50)))
    for k in ["vg1","vg2","ig1","ig2"]: d[k]=float(rng.uniform(-.02,.02))
    for k in ["vt1_ns","vt2_ns","it1_ns","it2_ns"]: d[k]=float(rng.uniform(-45,45))
    return d

def dataset(teachers,states=80):
    rng=np.random.default_rng(2209); data=[]
    for g,t in enumerate(teachers):
        for _ in range(states):
            theta=rng.uniform(-.16,.16,5).astype(np.float32); nu=nuisance(rng)
            y=rich_port(base.restamp_c(t["C"],theta),base.modal_l(t["L"],theta),nu)
            obs=observe(y,nu,rng).astype(np.complex64)
            feat=np.concatenate([np.log10(abs(obs).reshape(-1)+1e-12),np.angle(obs).reshape(-1)]).astype(np.float32)
            data.append(dict(graph=g,theta=theta,measurement=feat,yobs=obs,nuis=nu))
    return data

def tensors(batch,dev):
    return (torch.tensor(np.stack([x["measurement"] for x in batch]),device=dev),
      torch.tensor(np.stack([x["theta"] for x in batch]),device=dev),
      torch.tensor(np.stack([x["yobs"] for x in batch]),device=dev))

@torch.no_grad()
def predict(model,data,gids,teachers,dev):
    ans={}
    model.eval()
    for g,b in base.batches_by_graph(data,gids,96,False):
        m,t,y=tensors(b,dev); ei,ea=base.graph_edges(teachers[g]); node=torch.tensor(teachers[g]["node"],dtype=torch.float32,device=dev)[None].expand(len(b),-1,-1)
        p=model(node,torch.tensor(ei,dtype=torch.long,device=dev),torch.tensor(ea,device=dev),m)
        ans[g]=(b,p,t,y)
    return ans

def simple_metrics(theta,true,c0,l0,y):
    yp=base.torch_port(*base.torch_matrices(c0,l0,theta)); er=(yp-y).reshape(len(theta),-1); yr=y.reshape(len(theta),-1)
    return float(abs(theta-true).mean().cpu()),float((torch.linalg.vector_norm(er,dim=1)/torch.linalg.vector_norm(yr,dim=1)).mean().cpu())

def refine(init,c0,l0,y,anchor=None,steps=140):
    raw=torch.nn.Parameter(torch.atanh(torch.clamp(init/.22,-.999,.999))); opt=torch.optim.Adam([raw],lr=.04); tic=time.time()
    for _ in range(steps):
        th=.22*torch.tanh(raw); loss=base.physics_loss(th,c0,l0,y)
        if anchor is not None: loss=loss+0.12*torch.mean((th-anchor)**2)
        opt.zero_grad(); loss.backward(); opt.step()
    return (.22*torch.tanh(raw)).detach(),time.time()-tic

def main():
    base.seed_all(22); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); teachers=base.load_teachers(); data=dataset(teachers,80)
    train=set(range(8)); val={8,9}; test={10,11}; model=base.EdgeGNN(len(data[0]["measurement"])).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5); best=1e9; state=None; hist=[]; tic=time.time()
    for ep in range(700):
        model.train(); ls=[]
        for g,b in base.batches_by_graph(data,train,32,True):
            m,t,y=tensors(b,dev); ei,ea=base.graph_edges(teachers[g]); node=torch.tensor(teachers[g]["node"],dtype=torch.float32,device=dev)[None].expand(len(b),-1,-1)
            p=model(node,torch.tensor(ei,dtype=torch.long,device=dev),torch.tensor(ea,device=dev),m)
            c0=torch.tensor(teachers[g]["C"],dtype=torch.float32,device=dev); l0=torch.tensor(teachers[g]["L"],dtype=torch.float32,device=dev)
            # Small physics weight: inverse physics is knowingly incomplete.
            loss=torch.mean((p-t)**2)+.003*base.physics_loss(p,c0,l0,y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); ls.append(float(loss.detach()))
        if ep%10==0 or ep==699:
            pv=predict(model,data,val,teachers,dev); vm=np.mean([float(abs(x[1]-x[2]).mean().cpu()) for x in pv.values()]); hist.append((ep,np.mean(ls),vm))
            if vm<best: best=vm; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); torch.save(model.state_dict(),OUT/"robust_edge_gnn.pt")
    pred=predict(model,data,test,teachers,dev); rows=[]
    for g,(b,pg,t,y) in pred.items():
        c0=torch.tensor(teachers[g]["C"],dtype=torch.float32,device=dev); l0=torch.tensor(teachers[g]["L"],dtype=torch.float32,device=dev); zero=torch.zeros_like(pg)
        p0,s0=refine(zero,c0,l0,y,None); ph,sh=refine(pg,c0,l0,y,pg)
        for name,th,sec in [("nominal",zero,0),("GNN direct",pg,0),("physics mismatch",p0,s0),("GNN regularized physics",ph,sh)]:
            a,r=simple_metrics(th,t,c0,l0,y); rows.append(dict(graph=g,method=name,theta_mae=a,response_error=r,seconds=sec))
    df=pd.DataFrame(rows); df.to_csv(OUT/"comparison_by_graph.csv",index=False); agg=df.groupby("method").mean(numeric_only=True).reset_index(); agg.to_csv(OUT/"comparison_summary.csv",index=False)
    pd.DataFrame(hist,columns=["epoch","train_loss","val_theta_mae"]).to_csv(OUT/"training_history.csv",index=False)
    summary={"scope":"simulation-only forward/inverse mismatch pilot","device":str(dev),"cuda":torch.cuda.is_available(),"torch":torch.__version__,
      "comsol_geometries":12,"states_per_geometry":80,"train_graphs":8,"validation_graphs":2,"test_graphs":2,"best_val_theta_mae":best,
      "training_seconds":time.time()-tic,"mismatch_terms":["temperature-dependent R","sqrt-frequency R","asymmetric ground C","probe gain","probe delay","current noise"],
      "results":agg.to_dict(orient="records"),"limitations":["synthetic nuisance distributions","12 geometries","no hardware","linear-core quasi-static COMSOL teachers"]}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(agg.to_string(index=False)); print(json.dumps({k:v for k,v in summary.items() if k!="results"},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
