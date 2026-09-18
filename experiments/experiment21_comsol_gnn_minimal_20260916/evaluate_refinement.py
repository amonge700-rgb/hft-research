"""Compare pure physics inversion with GNN-initialized physics refinement."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_minimal_inverse as exp

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"

def fit(raw0,c0,l0,yobs,steps=120):
    raw=torch.nn.Parameter(raw0.clone()); opt=torch.optim.Adam([raw],lr=.045)
    tic=time.time()
    for _ in range(steps):
        theta=.22*torch.tanh(raw)
        loss=exp.physics_loss(theta,c0,l0,yobs)+2e-5*torch.mean(theta**2)
        opt.zero_grad(); loss.backward(); opt.step()
    return (.22*torch.tanh(raw)).detach(),time.time()-tic

def metrics(theta,true,c0,l0,yobs):
    yp=exp.torch_port(*exp.torch_matrices(c0,l0,theta))
    er=(yp-yobs).reshape(len(theta),-1); yr=yobs.reshape(len(theta),-1)
    return float(torch.mean(abs(theta-true)).cpu()),float(torch.mean(torch.linalg.vector_norm(er,dim=1)/torch.linalg.vector_norm(yr,dim=1)).cpu())

def main():
    exp.seed_all(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers=exp.load_teachers(); data=exp.make_dataset(teachers,60); test={10,11}
    model=exp.EdgeGNN(len(data[0]["measurement"])).to(dev)
    model.load_state_dict(torch.load(OUT/"edge_gnn.pt",map_location=dev,weights_only=True)); model.eval()
    rows=[]
    for g,batch in exp.batches_by_graph(data,test,64,False):
        m,t,y=exp.tensors(batch,dev); ei,ea=exp.graph_edges(teachers[g]); b=len(batch)
        node=torch.tensor(teachers[g]["node"],dtype=torch.float32,device=dev).unsqueeze(0).expand(b,-1,-1)
        with torch.no_grad():
            pg=model(node,torch.tensor(ei,dtype=torch.long,device=dev),torch.tensor(ea,device=dev),m)
        c0=torch.tensor(teachers[g]["C"],dtype=torch.float32,device=dev); l0=torch.tensor(teachers[g]["L"],dtype=torch.float32,device=dev)
        z0=torch.zeros_like(pg); rg=torch.atanh(torch.clamp(pg/.22,-.999,.999))
        p0,sec0=fit(z0,c0,l0,y); pr,secr=fit(rg,c0,l0,y)
        a,b0=metrics(p0,t,c0,l0,y); c,d=metrics(pr,t,c0,l0,y); e,f=metrics(pg,t,c0,l0,y)
        rows += [dict(graph=g,method="GNN direct",theta_mae=e,response_error=f,seconds=0.0),
                 dict(graph=g,method="physics from zero",theta_mae=a,response_error=b0,seconds=sec0),
                 dict(graph=g,method="GNN + physics",theta_mae=c,response_error=d,seconds=secr)]
    df=pd.DataFrame(rows); df.to_csv(OUT/"refinement_comparison.csv",index=False)
    agg=df.groupby("method").agg({"theta_mae":"mean","response_error":"mean","seconds":"sum"}).reset_index()
    agg.to_csv(OUT/"refinement_summary.csv",index=False)
    print(agg.to_string(index=False))

if __name__=="__main__": main()
