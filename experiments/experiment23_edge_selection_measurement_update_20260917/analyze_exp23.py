"""Create EXP-023 diagnostic figures and one held-out edge example."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_exp23 import (ROOT,RESULTS,FIGURES,PAIRS,load_teachers,topology_oracles,
                       make_data,EdgeInverseGNN,edge_features,tensors,topk_mask)

FIGURES.mkdir(exist_ok=True); df=pd.read_csv(RESULTS/'test_cases.csv'); s=json.loads((RESULTS/'summary.json').read_text(encoding='utf-8'))
random_update=4/len(PAIRS); random_top=round(.60*len(PAIRS))/len(PAIRS)
s['random_update_f1']=random_update; s['random_topology_recall_at_60pct']=random_top
s['response_error_reduction_percent']=100*(1-s['response_error']/s['nominal_error'])
(RESULTS/'summary.json').write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')

fig,axs=plt.subplots(2,2,figsize=(9,6.5))
axs[0,0].bar(['C','M'],[s['topology_recall_C'],s['topology_recall_M']],color=['#4477AA','#228C7B']); axs[0,0].axhline(random_top,color='#777',ls='--',label='random'); axs[0,0].set(title='Static important-edge recall@60%',ylim=(0,1)); axs[0,0].legend()
axs[0,1].bar(['C','M'],[s['update_f1_C'],s['update_f1_M']],color=['#4477AA','#228C7B']); axs[0,1].axhline(random_update,color='#777',ls='--',label='random'); axs[0,1].set(title='Changed-edge localization F1',ylim=(0,.5)); axs[0,1].legend()
axs[1,0].scatter(100*df.nominal_error,100*df.response_error,c=df.snr_db,cmap='viridis',s=22,alpha=.75); lim=max(100*df.nominal_error.max(),100*df.response_error.max()); axs[1,0].plot([0,lim],[0,lim],'--',color='#777'); axs[1,0].set(xlabel='Nominal error (%)',ylabel='GNN-updated error (%)',title='Held-out geometry cases')
gain=100*(df.nominal_error-df.response_error)/df.nominal_error.clip(lower=1e-9); axs[1,1].hist(gain,bins=16,color='#C44E52',alpha=.85); axs[1,1].axvline(gain.mean(),color='black',ls='--',label=f'mean={gain.mean():.1f}%'); axs[1,1].set(xlabel='Response-error reduction (%)',ylabel='Cases',title='Benefit distribution'); axs[1,1].legend()
for a,l in zip(axs.flat,'abcd'): a.text(-.13,1.05,l,transform=a.transAxes,fontweight='bold'); a.grid(alpha=.18)
fig.tight_layout(); fig.savefig(FIGURES/'exp23_diagnostics.png',dpi=300); fig.savefig(FIGURES/'exp23_diagnostics.pdf'); plt.close(fig)

# Recreate deterministic held-out data and visualize one representative update.
teachers=load_teachers(); data=make_data(teachers,50,2301); sample=next(x for x in data if x['graph']==10); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=EdgeInverseGNN(len(sample['measurement'])).to(device); model.load_state_dict(torch.load(RESULTS/'edge_selector_updater.pt',map_location=device,weights_only=True)); model.eval()
edge=torch.tensor(edge_features(teachers[10]),device=device); batch=[sample]; meas,_,mc,ml,dc,dl=tensors(batch,device)
with torch.no_grad(): static,logits,delta=model(edge,meas); pc=topk_mask(logits[...,0])[0]; pl=topk_mask(logits[...,1])[0]; pred=delta[0]*torch.stack((pc,pl),-1)
rows=[]
for e,(i,j) in enumerate(PAIRS): rows.append(dict(edge=e,turn_i=i+1,turn_j=j+1,true_C=float(dc[0,e]),pred_C=float(pred[e,0]),true_M=float(dl[0,e]),pred_M=float(pred[e,1]),score_C=float(logits[0,e,0]),score_M=float(logits[0,e,1])))
pd.DataFrame(rows).to_csv(RESULTS/'heldout_edge_example.csv',index=False)

def mat(vals):
    z=np.zeros((8,8));
    for v,(i,j) in zip(vals,PAIRS): z[i,j]=z[j,i]=v
    return z
fig,axs=plt.subplots(2,2,figsize=(7.4,6.3)); panels=[mat(dc[0].cpu()),mat(pred[:,0].cpu()),mat(dl[0].cpu()),mat(pred[:,1].cpu())]; titles=['True C-edge changes','Predicted C-edge changes','True M-edge changes','Predicted M-edge changes']
for ax,z,t in zip(axs.flat,panels,titles):
    lim=max(abs(z).max(),1e-3); im=ax.imshow(z,cmap='coolwarm',vmin=-lim,vmax=lim); ax.set_title(t); ax.set_xticks(range(8),['P1','P2','P3','P4','S1','S2','S3','S4'],rotation=45); ax.set_yticks(range(8),['P1','P2','P3','P4','S1','S2','S3','S4']); fig.colorbar(im,ax=ax,fraction=.046,pad=.03)
for a,l in zip(axs.flat,'abcd'): a.text(-.15,1.05,l,transform=a.transAxes,fontweight='bold')
fig.tight_layout(); fig.savefig(FIGURES/'heldout_edge_update_example.png',dpi=300); fig.savefig(FIGURES/'heldout_edge_update_example.pdf'); plt.close(fig)
print(json.dumps(s,ensure_ascii=False,indent=2))
