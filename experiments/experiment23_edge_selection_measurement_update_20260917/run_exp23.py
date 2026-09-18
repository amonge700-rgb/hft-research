"""EXP-023: edge selection and measurement-conditioned coupling updates.

This simulation-only pilot separates two questions:
1) static topology: which turn-pair C/M edges should be retained;
2) dynamic inverse: which retained couplings changed, and by how much, from
   noisy multi-window two-port observations.

The physical matrix/Kron decoder remains authoritative.  The GNN predicts
edge scores and bounded log-scale corrections; it never predicts responses as
the final answer.
"""
from __future__ import annotations

import argparse, copy, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'matplotlib'))
EXP21=ROOT.parent/'experiment21_comsol_gnn_minimal_20260916'
sys.path.insert(0,str(EXP21))
from run_minimal_inverse import load_teachers, graph_edges, noisy_multiwindow_y, solve_port_np, FREQ

RESULTS=ROOT/'results'; FIGURES=ROOT/'figures'; N=8
PAIRS=[(i,j) for i in range(N) for j in range(i+1,N)]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rebuild_c(c0, log_delta):
    """Apply pairwise positive mutual-capacitance updates and preserve Maxwell form."""
    m=-c0.copy(); np.fill_diagonal(m,0); cref=c0.sum(1)
    for e,(i,j) in enumerate(PAIRS): m[i,j]=m[j,i]=m[i,j]*math.exp(float(log_delta[e]))
    c=-m; np.fill_diagonal(c,cref+m.sum(1)); return c

def repair_l_np(l, eps=1e-10):
    l=(l+l.T)/2; w=np.linalg.eigvalsh(l)
    if w[0]<eps: l=l+np.eye(N)*(eps-w[0])
    return l

def rebuild_l(l0, log_delta):
    l=l0.copy()
    for e,(i,j) in enumerate(PAIRS): l[i,j]=l[j,i]=l0[i,j]*math.exp(float(log_delta[e]))
    return repair_l_np(l)

def edge_features(t):
    _,directed=graph_edges(t); feats=[]
    # graph_edges is ordered i->j; reconstruct the seven physical features.
    lookup={}; k=0
    for i in range(N):
        for j in range(N):
            if i!=j: lookup[(i,j)]=directed[k]; k+=1
    for i,j in PAIRS: feats.append(lookup[(i,j)])
    return np.asarray(feats,np.float32)

def prune_c(c0, keep):
    m=-c0.copy(); np.fill_diagonal(m,0); cref=c0.sum(1)
    for e,(i,j) in enumerate(PAIRS):
        if not keep[e]: m[i,j]=m[j,i]=0
    c=-m; np.fill_diagonal(c,cref+m.sum(1)); return c

def prune_l(l0, keep):
    l=l0.copy()
    for e,(i,j) in enumerate(PAIRS):
        if not keep[e]: l[i,j]=l[j,i]=0
    return repair_l_np(l)

def topology_oracles(teachers):
    """Single-edge ablation scores: response damage caused by deleting each edge."""
    all_scores=[]
    for t in teachers:
        y=solve_port_np(t['C'],t['L']); sc=np.zeros((len(PAIRS),2))
        for e in range(len(PAIRS)):
            kc=np.ones(len(PAIRS),bool); kc[e]=False
            kl=np.ones(len(PAIRS),bool); kl[e]=False
            yc=solve_port_np(prune_c(t['C'],kc),t['L']); yl=solve_port_np(t['C'],prune_l(t['L'],kl))
            sc[e,0]=np.linalg.norm(yc-y)/np.linalg.norm(y)
            sc[e,1]=np.linalg.norm(yl-y)/np.linalg.norm(y)
        all_scores.append(sc)
    return all_scores

def make_data(teachers, states=60, seed=2301):
    rng=np.random.default_rng(seed); rows=[]
    for g,t in enumerate(teachers):
        feat=edge_features(t); c_strength=np.exp(feat[:,5]-feat[:,5].max()); l_strength=np.exp(feat[:,6]-feat[:,6].max())
        pc=.25+.75*c_strength/c_strength.sum()*len(PAIRS); pl=.25+.75*l_strength/l_strength.sum()*len(PAIRS)
        pc=pc/pc.sum(); pl=pl/pl.sum()
        for s in range(states):
            kc,kl=4,4; ic=rng.choice(len(PAIRS),kc,False,p=pc); il=rng.choice(len(PAIRS),kl,False,p=pl)
            dc=np.zeros(len(PAIRS)); dl=np.zeros(len(PAIRS)); dc[ic]=rng.uniform(-.22,.22,kc); dl[il]=rng.uniform(-.12,.12,kl)
            c=rebuild_c(t['C'],dc); l=rebuild_l(t['L'],dl); clean=solve_port_np(c,l); nominal=solve_port_np(t['C'],t['L'])
            snr=float(rng.uniform(34,52)); obs=noisy_multiwindow_y(clean,rng,snr)
            # Physics-anchored innovation: the known nominal graph explains the
            # large geometry-dependent response; the GNN sees only the complex
            # measurement residual that must be attributed to changed edges.
            innovation=(obs-nominal)/(np.abs(nominal)+1e-9)
            meas=np.concatenate([innovation.real.reshape(-1),innovation.imag.reshape(-1)]).astype(np.float32)
            rows.append(dict(graph=g,measurement=meas,yobs=obs.astype(np.complex64),dc=dc.astype(np.float32),dl=dl.astype(np.float32),
                             mc=(dc!=0).astype(np.float32),ml=(dl!=0).astype(np.float32),snr=snr))
    return rows

class EdgeInverseGNN(nn.Module):
    def __init__(self, meas_dim, hidden=72):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(7,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.meas=nn.Sequential(nn.Linear(meas_dim,144),nn.SiLU(),nn.Linear(144,hidden))
        self.static=nn.Linear(hidden,2)
        self.dynamic=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,4))
    def forward(self,edge,meas):
        e=self.edge(edge); q=self.meas(meas)
        static=self.static(e)
        z=torch.cat([e.unsqueeze(0).expand(meas.shape[0],-1,-1),q.unsqueeze(1).expand(-1,e.shape[0],-1)],-1)
        d=self.dynamic(z)
        return static,d[...,:2],torch.stack((.24*torch.tanh(d[...,2]),.14*torch.tanh(d[...,3])),-1)

def torch_rebuild(c0,l0,delta):
    b=delta.shape[0]; dev=c0.device; c0b=c0.expand(b,-1,-1); l0b=l0.expand(b,-1,-1)
    cref=c0.sum(1).expand(b,-1)
    c=torch.diag_embed(cref)
    l=torch.diag_embed(torch.diagonal(l0).expand(b,-1))
    for e,(i,j) in enumerate(PAIRS):
        cm=(-c0[i,j])*torch.exp(delta[:,e,0]); lm=l0[i,j]*torch.exp(delta[:,e,1])
        bc=torch.zeros((N,N),device=dev,dtype=c0.dtype); bc[i,i]=bc[j,j]=1; bc[i,j]=bc[j,i]=-1
        bl=torch.zeros((N,N),device=dev,dtype=l0.dtype); bl[i,j]=bl[j,i]=1
        c=c+cm[:,None,None]*bc; l=l+lm[:,None,None]*bl
    l=(l+l.transpose(1,2))/2; eig=torch.linalg.eigvalsh(l); shift=torch.relu(1e-10-eig[:,0]); l=l+shift[:,None,None]*torch.eye(N,device=l.device)
    return c,l

def incidence(device,dtype):
    a=torch.zeros((N,N),device=device,dtype=dtype)
    for o in (0,4):
        for k in range(4):
            a[o+k,o+k]=1
            if k<3:a[o+k+1,o+k]=-1
    return a

def torch_port(c,l):
    b=c.shape[0]; dev=c.device; a=incidence(dev,c.dtype); r=torch.eye(N,device=dev,dtype=c.dtype)*2.2e-3
    p=torch.tensor([0,4],device=dev); q=torch.tensor([1,2,3,5,6,7],device=dev); out=[]
    for f in FREQ:
        z=(r[None]+1j*2*math.pi*f*l).to(torch.complex64); y=a.to(torch.complex64)[None]@torch.linalg.solve(z,a.T.to(torch.complex64).expand(b,-1,-1))+1j*2*math.pi*f*c.to(torch.complex64)
        pp=y[:,p][:,:,p]; pi=y[:,p][:,:,q]; ip=y[:,q][:,:,p]; ii=y[:,q][:,:,q]; out.append(pp-pi@torch.linalg.solve(ii,ip))
    return torch.stack(out,1)

def batches(data,gids,size=32,shuffle=True):
    groups={}
    for x in data:
        if x['graph'] in gids: groups.setdefault(x['graph'],[]).append(x)
    keys=list(groups)
    if shuffle: random.shuffle(keys)
    for g in keys:
        z=groups[g]
        if shuffle: random.shuffle(z)
        for k in range(0,len(z),size): yield g,z[k:k+size]

def tensors(batch,device):
    def st(k): return torch.tensor(np.stack([x[k] for x in batch]),device=device)
    return st('measurement'),st('yobs'),st('mc'),st('ml'),st('dc'),st('dl')

def topk_mask(logits,k=4):
    ids=torch.topk(logits,k,dim=1).indices; z=torch.zeros_like(logits); return z.scatter(1,ids,1)

@torch.no_grad()
def evaluate(model,data,gids,teachers,oracles,device):
    model.eval(); rows=[]; keep_frac=.60; keep=max(1,round(keep_frac*len(PAIRS)))
    for g,batch in batches(data,gids,64,False):
        meas,yobs,mc,ml,dc,dl=tensors(batch,device); edge=torch.tensor(edge_features(teachers[g]),device=device)
        static,logits,delta=model(edge,meas); pc=topk_mask(logits[...,0]); pl=topk_mask(logits[...,1]); pred=delta*torch.stack((pc,pl),-1)
        c0=torch.tensor(teachers[g]['C'],dtype=torch.float32,device=device); l0=torch.tensor(teachers[g]['L'],dtype=torch.float32,device=device)
        yp=torch_port(*torch_rebuild(c0,l0,pred)); y0=torch_port(c0[None].expand(len(batch),-1,-1),l0[None].expand(len(batch),-1,-1))
        sc=static[:,0]; sl=static[:,1]; gc=torch.topk(sc,keep).indices.cpu().numpy(); gl=torch.topk(sl,keep).indices.cpu().numpy()
        oc=np.argsort(oracles[g][:,0])[-keep:]; ol=np.argsort(oracles[g][:,1])[-keep:]
        topo_c=len(set(gc)&set(oc))/keep; topo_l=len(set(gl)&set(ol))/keep
        for n,x in enumerate(batch):
            tp_c=float((pc[n]*mc[n]).sum()); tp_l=float((pl[n]*ml[n]).sum())
            f1c=2*tp_c/(float(pc[n].sum()+mc[n].sum())+1e-9); f1l=2*tp_l/(float(pl[n].sum()+ml[n].sum())+1e-9)
            rows.append(dict(graph=g,snr_db=x['snr'],topology_recall_C=topo_c,topology_recall_M=topo_l,
                update_f1_C=f1c,update_f1_M=f1l,delta_mae_C=float(torch.mean(abs(pred[n,:,0]-dc[n]))),delta_mae_M=float(torch.mean(abs(pred[n,:,1]-dl[n]))),
                response_error=float(torch.linalg.norm(yp[n]-yobs[n])/torch.linalg.norm(yobs[n])),nominal_error=float(torch.linalg.norm(y0[n]-yobs[n])/torch.linalg.norm(yobs[n]))))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=350); ap.add_argument('--states',type=int,default=70); ap.add_argument('--quick',action='store_true'); ap.add_argument('--seed',type=int,default=2301)
    args=ap.parse_args();
    if args.quick: args.epochs,args.states=12,12
    seed_all(args.seed); RESULTS.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); teachers=load_teachers(); oracles=topology_oracles(teachers); data=make_data(teachers,args.states,args.seed)
    train=set(range(8)); val={8,9}; test={10,11}; model=EdgeInverseGNN(len(data[0]['measurement'])).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-5)
    pos_weight=torch.tensor((len(PAIRS)-4)/4,device=device); bce=nn.BCEWithLogitsLoss(pos_weight=pos_weight); best=1e99; state=None; hist=[]; tic=time.time()
    for ep in range(args.epochs):
        model.train(); losses=[]
        for g,batch in batches(data,train,32,True):
            meas,yobs,mc,ml,dc,dl=tensors(batch,device); edge=torch.tensor(edge_features(teachers[g]),device=device); static,logits,delta=model(edge,meas)
            target_top=torch.tensor(oracles[g],dtype=torch.float32,device=device); target_top=target_top/(target_top.max(0).values+1e-12)
            ls=nn.functional.mse_loss(torch.sigmoid(static),target_top)
            lc=bce(logits[...,0],mc)+bce(logits[...,1],ml)
            mask=torch.stack((mc,ml),-1); ld=torch.sum(((delta-torch.stack((dc,dl),-1))*mask)**2)/(mask.sum()+1e-9)
            soft=torch.sigmoid(logits); pred=delta*soft; c0=torch.tensor(teachers[g]['C'],dtype=torch.float32,device=device); l0=torch.tensor(teachers[g]['L'],dtype=torch.float32,device=device)
            yp=torch_port(*torch_rebuild(c0,l0,pred)); scale=torch.mean(abs(yobs)).clamp_min(1e-7); lp=torch.mean(abs((yp-yobs)/scale)**2)
            loss=ls+lc+.8*ld+.015*lp; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); opt.step(); losses.append(float(loss.detach()))
        if ep%10==0 or ep==args.epochs-1:
            v=evaluate(model,data,val,teachers,oracles,device); score=float(v.response_error.mean()+.2*(2-v.update_f1_C.mean()-v.update_f1_M.mean())); hist.append(dict(epoch=ep,train_loss=np.mean(losses),val_score=score))
            if score<best: best=score; state=copy.deepcopy(model.state_dict())
        if ep%25==0: print(f'epoch={ep:03d} loss={np.mean(losses):.4g} best={best:.4g} device={device}')
    model.load_state_dict(state); torch.save(model.state_dict(),RESULTS/'edge_selector_updater.pt')
    df=evaluate(model,data,test,teachers,oracles,device); df.to_csv(RESULTS/'test_cases.csv',index=False); pd.DataFrame(hist).to_csv(RESULTS/'training_history.csv',index=False)
    metrics={k:float(df[k].mean()) for k in df.columns if k not in ('graph',)}
    metrics.update(dict(device=str(device),cuda_available=torch.cuda.is_available(),trainable_parameters=sum(p.numel() for p in model.parameters()),train_seconds=time.time()-tic,
      train_graphs=8,val_graphs=2,test_graphs=2,states_per_graph=args.states,scope='simulation-only minimum edge-selection and update pilot',
      interpretation='static edge ranking plus measurement-conditioned sparse C/M corrections; not hardware validation'))
    (RESULTS/'summary.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.2));
    axs[0].bar(['C','M'],[metrics['topology_recall_C'],metrics['topology_recall_M']],color=['#4477AA','#228C7B']); axs[0].set(title='Static topology recall@60%',ylim=(0,1))
    axs[1].bar(['C','M'],[metrics['update_f1_C'],metrics['update_f1_M']],color=['#4477AA','#228C7B']); axs[1].set(title='Changed-edge F1',ylim=(0,1))
    axs[2].bar(['Nominal','GNN update'],100*np.array([metrics['nominal_error'],metrics['response_error']]),color=['#888888','#C44E52']); axs[2].set(title='Port-response reconstruction',ylabel='Relative error (%)')
    for a,l in zip(axs,'abc'): a.text(-.15,1.05,l,transform=a.transAxes,fontweight='bold'); a.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/'exp23_summary.png',dpi=240); fig.savefig(FIGURES/'exp23_summary.pdf'); plt.close(fig)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
