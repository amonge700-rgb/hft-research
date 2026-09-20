"""EXP-027: DAB multi-window online identification in a physical 5-D graph space.

Three capacitance groups preserve the Maxwell-capacitance/Laplacian structure.
Two eigenmodes update the inductance matrix while preserving positive
definiteness.  A small spectral latent-attention network estimates these five
coordinates from reconstructed two-port spectra; a Gauss-Newton physics stage
then enforces response consistency.
"""
from __future__ import annotations

import argparse, copy, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'matplotlib'))
E24=ROOT.parent/'experiment24_fisher_identifiable_graph_modes_20260917'
E25=ROOT.parent/'experiment25_dab_pwm_time_domain_observability_20260917'
sys.path[:0]=[str(E24),str(E25)]
from run_exp24 import load_teachers, solve_port_np
from run_exp25 import WINDOWS

OUT=ROOT/'results'; FIG=ROOT/'figures'; P=3; N=8
# Preserve low-order power harmonics and add edge-band harmonics.  With a
# 20-kHz bridge this reaches 10.02 MHz, below both the synthesized waveform
# Nyquist limit (10.24 MHz) and the approximate 17.5-MHz bandwidth of a 20-ns
# SiC edge.  The number of stored samples remains 8192.
FS=20_000.; H=np.array([1,3,5,7,9,11,21,41,81,121,241,401,501]); FREQ=FS*H
NPP=1024; PERIODS=8; NS=NPP*PERIODS
PILOT_V=0.0
SPECTRAL_AVERAGES=8


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def c_group(i,j):
    same=(i<4)==(j<4)
    if not same: return 2
    return 0 if abs(i%4-j%4)==1 else 1


def rebuild_physical(t,z):
    """Three identifiable coordinates: aggregate C_cross and L_mode1/2."""
    z=np.asarray(z,float)
    c0=t['C']; mutual=-c0.copy(); np.fill_diagonal(mutual,0); cref=c0.sum(1)
    for i in range(N):
        for j in range(i+1,N):
            scale=math.exp(float(z[0])) if c_group(i,j)==2 else 1.0
            val=mutual[i,j]*scale
            mutual[i,j]=mutual[j,i]=val
    c=-mutual; np.fill_diagonal(c,cref+mutual.sum(1))

    l0=(t['L']+t['L'].T)/2
    w,q=np.linalg.eigh(l0); ids=np.argsort(w)[-2:]
    w=np.maximum(w,1e-12)
    for k,idx in enumerate(ids): w[idx]*=math.exp(float(z[1+k]))
    l=(q*w)@q.T
    return c,(l+l.T)/2


def bridge_phasors(ratio,phase_deg,vdc=400.,tr=20e-9,dead=100e-9):
    amp=4*vdc/(np.pi*H)*np.sinc(FREQ*tr)*np.cos(np.pi*FREQ*dead)
    v1=amp.astype(complex); v2=ratio*amp*np.exp(-1j*H*np.deg2rad(phase_deg))
    # Optional bounded diagnostic multisine in the edge band.  Different DAB
    # windows receive different phases, retaining two-port excitation rank.
    mask=H>=21
    code=np.deg2rad(phase_deg)*(1+(H[mask]%7))
    v1[mask]+=PILOT_V*np.exp(1j*code)
    v2[mask]+=PILOT_V*np.exp(-1j*(1.37*code+.2))
    return np.vstack((v1,v2)).T


def synthesize(x):
    n=np.arange(NS); ans=np.zeros((NS,x.shape[1]))
    for k,h in enumerate(H): ans+=np.real(x[k][None,:]*np.exp(1j*2*np.pi*h*n/NPP)[:,None])
    return ans


def extract(x):
    xf=np.fft.rfft(x,axis=0)
    return np.stack([2*xf[int(h*PERIODS)]/NS for h in H])


def observe_matrices(c,l,mode='multi_pwm',snr_db=None,rng=None):
    y=solve_port_np(c,l,FREQ); vs=[]; ins=[]
    for ratio,phi in WINDOWS[mode]:
        vp=bridge_phasors(ratio,phi); ip=np.einsum('kij,kj->ki',y,vp)
        repeats=SPECTRAL_AVERAGES if snr_db is not None else 1
        for _ in range(repeats):
            vt=synthesize(vp); it=synthesize(ip)
            if snr_db is not None:
                rng=np.random.default_rng() if rng is None else rng
                # Bridge voltage is synchronized to the known PWM command and
                # is treated as a high-SNR instrumental channel.  Current is
                # the limiting measurement channel.
                sv=np.sqrt(np.mean(vt*vt))*10**(-(snr_db+20)/20)
                si=np.sqrt(np.mean(it*it))*10**(-snr_db/20)
                vt+=rng.normal(0,sv,vt.shape); it+=rng.normal(0,si,it.shape)
            vs.append(extract(vt)); ins.append(extract(it))
    yh=[]; cond=[]
    for k in range(len(H)):
        v=np.stack([x[k] for x in vs],1); i=np.stack([x[k] for x in ins],1)
        cond.append(np.linalg.cond(v@v.conj().T)); yh.append(i@np.linalg.pinv(v))
    return np.stack(yh),np.asarray(cond),y


def y_scale(y0):
    """Noise-aware complex-response scale with a per-frequency floor."""
    floor=.02*np.linalg.norm(y0,axis=(1,2),keepdims=True)
    return np.maximum(np.abs(y0),floor+1e-12)


def response_feature(t,z):
    c,l=rebuild_physical(t,z); y=solve_port_np(c,l,FREQ); y0=solve_port_np(t['C'],t['L'],FREQ)
    q=(y-y0)/y_scale(y0)
    return np.r_[q.real.ravel(),q.imag.ravel()]


def measured_tokens(t,z,rng,snr):
    c,l=rebuild_physical(t,z); yn,cond,ytrue=observe_matrices(c,l,'multi_pwm',snr,rng)
    y0=solve_port_np(t['C'],t['L'],FREQ); q=(yn-y0)/y_scale(y0)
    toks=[]
    lf=(np.log10(FREQ)-np.log10(FREQ).mean())/np.log10(FREQ).std()
    for k in range(len(FREQ)):
        for i in range(2):
            for j in range(2):
                toks.append([lf[k],2*i-1,2*j-1,q[k,i,j].real,q[k,i,j].imag,np.log10(cond[k]+1)/8])
    return np.asarray(toks,np.float32),np.r_[q.real.ravel(),q.imag.ravel()].astype(np.float64),yn,ytrue,cond


def make_data(teachers,ids,states,seed):
    rng=np.random.default_rng(seed); rows=[]
    for gid in ids:
        for case in range(states):
            z=np.r_[rng.uniform(-.18,.18,1),rng.uniform(-.10,.10,2)]
            snr=float(rng.uniform(40,52)); tok,feat,yn,yt,cond=measured_tokens(teachers[gid],z,rng,snr)
            rows.append({'graph':gid,'case':case,'z':z.astype(np.float32),'tokens':tok,'feature':feat,
                         'snr':snr,'yobs':yn,'ytrue':yt,'cond':cond})
    return rows


class SpectralModePerceiver(nn.Module):
    def __init__(self,d=64,latents=8,heads=4):
        super().__init__(); self.embed=nn.Sequential(nn.Linear(6,d),nn.SiLU(),nn.Linear(d,d))
        self.latent=nn.Parameter(torch.randn(latents,d)*.03); self.query=nn.Parameter(torch.randn(P,d)*.03)
        self.read=nn.MultiheadAttention(d,heads,batch_first=True)
        enc=nn.TransformerEncoderLayer(d,heads,2*d,batch_first=True,activation='gelu',norm_first=True)
        self.body=nn.TransformerEncoder(enc,2); self.out=nn.MultiheadAttention(d,heads,batch_first=True)
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):
        b=x.shape[0]; e=self.embed(x); l=self.latent[None].expand(b,-1,-1)
        l=self.read(l,e,e,need_weights=False)[0]+l; l=self.body(l)
        q=self.query[None].expand(b,-1,-1); q=self.out(q,l,l,need_weights=False)[0]+q
        return self.head(q).squeeze(-1)


def jacobian(t,z,h=5e-4):
    z=np.asarray(z,float); j=np.empty((8*len(FREQ),P))
    for k in range(P):
        zp=z.copy(); zm=z.copy(); zp[k]+=h; zm[k]-=h
        j[:,k]=(response_feature(t,zp)-response_feature(t,zm))/(2*h)
    return j


def gn_refine(t,feature,z0,steps=4,lam=3e-4):
    z=np.asarray(z0,float).copy()
    for _ in range(steps):
        r=feature-response_feature(t,z); j=jacobian(t,z)
        dz=np.linalg.solve(j.T@j+lam*np.eye(P),j.T@r)
        z=np.clip(z+dz,[-.35,-.22,-.22],[.35,.22,.22])
    return z


def linear_physics(t,feature,rank=None):
    j=jacobian(t,np.zeros(P)); u,s,vh=np.linalg.svd(j,full_matrices=False)
    if rank is None: b=np.eye(P)
    else: b=vh.T[:,:rank]
    z=b@np.linalg.solve((j@b).T@(j@b)+3e-4*np.eye(b.shape[1]),(j@b).T@feature)
    return np.clip(z,[-.35,-.22,-.22],[.35,.22,.22])


def batches(data,size,shuffle=True):
    ids=np.arange(len(data));
    if shuffle: np.random.shuffle(ids)
    for i in range(0,len(ids),size): yield [data[j] for j in ids[i:i+size]]


def evaluate(model,data,teachers,device):
    model.eval(); rows=[]
    with torch.no_grad():
        for batch in batches(data,32,False):
            x=torch.tensor(np.stack([r['tokens'] for r in batch]),device=device); pred=model(x).cpu().numpy()
            for r,zn in zip(batch,pred):
                t=teachers[r['graph']]; zp=linear_physics(t,r['feature']); zf=linear_physics(t,r['feature'],3); zh=gn_refine(t,r['feature'],zn)
                for method,z in [('nominal',np.zeros(P)),('physics5',zp),('fisher3',zf),('spectral_gnn',zn),('gnn_physics',zh)]:
                    yp=solve_port_np(*rebuild_physical(t,z),FREQ)
                    rows.append({'graph':r['graph'],'case':r['case'],'method':method,'snr':r['snr'],
                      'parameter_rmse':float(np.sqrt(np.mean((z-r['z'])**2))),
                      'c_group_mae':float(np.mean(np.abs(z[:1]-r['z'][:1]))),
                      'm_mode_mae':float(np.mean(np.abs(z[1:]-r['z'][1:]))),
                      'response_error':float(np.linalg.norm(yp-r['ytrue'])/np.linalg.norm(r['ytrue'])),
                      'estimate_norm':float(np.linalg.norm(z))})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--states',type=int,default=80)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--quick',action='store_true'); ap.add_argument('--seed',type=int,default=2701)
    ap.add_argument('--pilot-v',type=float,default=0.0)
    a=ap.parse_args(); seed_all(a.seed)
    global PILOT_V,OUT,FIG
    PILOT_V=float(a.pilot_v)
    if PILOT_V>0:
        tag=f'pilot_{PILOT_V:g}V'; OUT=ROOT/f'results_{tag}'; FIG=ROOT/f'figures_{tag}'
    if a.quick: a.epochs=3; a.states=4
    device=torch.device(a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu')
    teachers=load_teachers(); train=make_data(teachers,range(8),a.states,a.seed); val=make_data(teachers,[8,9],max(8,a.states//4),a.seed+1); test=make_data(teachers,[10,11],max(20,a.states//2),a.seed+2)
    model=SpectralModePerceiver().to(device); opt=torch.optim.AdamW(model.parameters(),2e-3,weight_decay=2e-5)
    best=float('inf'); state=None; hist=[]; tic=time.time()
    for ep in range(a.epochs):
        model.train(); losses=[]
        for b in batches(train,32):
            x=torch.tensor(np.stack([r['tokens'] for r in b]),device=device); y=torch.tensor(np.stack([r['z'] for r in b]),device=device)
            p=model(x); scale=torch.tensor([.18,.10,.10],device=device); loss=torch.mean(((p-y)/scale)**2)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3); opt.step(); losses.append(float(loss.detach()))
        if ep%5==0 or ep==a.epochs-1:
            model.eval()
            with torch.no_grad():
                xv=torch.tensor(np.stack([r['tokens'] for r in val]),device=device); yv=torch.tensor(np.stack([r['z'] for r in val]),device=device)
                vl=float(torch.mean((model(xv)-yv)**2))
            hist.append({'epoch':ep,'train_loss':np.mean(losses),'val_mse':vl})
            if vl<best: best=vl; state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state); df=evaluate(model,test,teachers,device)
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True); df.to_csv(OUT/'test_cases.csv',index=False); pd.DataFrame(hist).to_csv(OUT/'training_history.csv',index=False)
    torch.save(model.state_dict(),OUT/'spectral_mode_perceiver.pt')
    metrics={m:{k:float(v) for k,v in d[['parameter_rmse','c_group_mae','m_mode_mae','response_error','estimate_norm']].mean().items()} for m,d in df.groupby('method')}
    summary={'experiment':'EXP-027','scope':'simulation-only DAB multi-window online identification in aggregate cross-winding C scale + 2 PSD L modes','pilot_v':PILOT_V,'device':str(device),'cuda_available':torch.cuda.is_available(),'epochs':a.epochs,'states_per_train_graph':a.states,'train_graphs':8,'val_graphs':2,'test_graphs':2,'test_states':len(test),'parameters':sum(p.numel() for p in model.parameters()),'runtime_seconds':time.time()-tic,'metrics':metrics}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    order=['nominal','physics5','fisher3','spectral_gnn','gnn_physics']; fig,axs=plt.subplots(1,3,figsize=(12,3.3))
    for ax,key,title in zip(axs,['parameter_rmse','response_error','estimate_norm'],['5-D parameter RMSE','Port-response error','Estimate norm']):
        ax.boxplot([df[df.method==m][key] for m in order],tick_labels=order,showfliers=False); ax.tick_params(axis='x',rotation=20); ax.set_title(title); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'exp27_summary.png',dpi=260); fig.savefig(FIG/'exp27_summary.pdf'); plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
