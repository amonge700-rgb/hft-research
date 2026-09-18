"""EXP-024: Fisher-identifiable modes for turn-level C/M graph updates."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parent
TEACHERS=ROOT.parent/'experiment21_comsol_gnn_minimal_20260916'/'data'/'comsol_teachers'
OUT=ROOT/'results'; FIG=ROOT/'figures'; N=8
FREQ=np.array([17e3,51e3,85e3,23e3,69e3,115e3],float)
PAIRS=[(i,j) for i in range(N) for j in range(i+1,N)]; P=2*len(PAIRS)

def load_teachers():
    manifest=pd.read_csv(TEACHERS/'manifest.csv'); rows=[]
    for _,r in manifest.iterrows():
        p=TEACHERS/r['sample_id']
        rows.append({'sample_id':r['sample_id'],
                     'C':pd.read_csv(p/'C.csv',index_col=0).to_numpy(float),
                     'L':pd.read_csv(p/'L.csv',index_col=0).to_numpy(float)})
    return rows

def rebuild_c(c0,log_delta):
    m=-c0.copy(); np.fill_diagonal(m,0); cref=c0.sum(1)
    for e,(i,j) in enumerate(PAIRS): m[i,j]=m[j,i]=m[i,j]*math.exp(float(log_delta[e]))
    c=-m; np.fill_diagonal(c,cref+m.sum(1)); return c

def rebuild_l(l0,log_delta):
    l=l0.copy()
    for e,(i,j) in enumerate(PAIRS): l[i,j]=l[j,i]=l0[i,j]*math.exp(float(log_delta[e]))
    l=(l+l.T)/2; w=np.linalg.eigvalsh(l)
    if w[0]<1e-10: l+=np.eye(N)*(1e-10-w[0])
    return l

def solve_port_np(c,l,freq=FREQ):
    a=np.zeros((N,N))
    for o in (0,4):
        for k in range(4):
            a[o+k,o+k]=1
            if k<3: a[o+k+1,o+k]=-1
    r=np.eye(N)*2.2e-3; ports=np.array([0,4]); internal=np.array([1,2,3,5,6,7]); ans=[]
    for f in freq:
        y=a@np.linalg.solve(r+1j*2*np.pi*f*l,a.T)+1j*2*np.pi*f*c
        pp=y[np.ix_(ports,ports)]; pi=y[np.ix_(ports,internal)]
        ip=y[np.ix_(internal,ports)]; ii=y[np.ix_(internal,internal)]
        ans.append(pp-pi@np.linalg.solve(ii,ip))
    return np.stack(ans)

def obs(t,z):
    y0=solve_port_np(t['C'],t['L']); y=solve_port_np(rebuild_c(t['C'],z[:28]),rebuild_l(t['L'],z[28:])); q=(y-y0)/(np.abs(y0)+1e-9)
    return np.r_[q.real.ravel(),q.imag.ravel()]

def jacobian(t,h=2e-4):
    z=np.zeros(P); j=np.zeros((48,P))
    for k in range(P):
        zp=z.copy(); zm=z.copy(); zp[k]=h; zm[k]=-h; j[:,k]=(obs(t,zp)-obs(t,zm))/(2*h)
    return j

def ridge(a,b,lam=1e-4): return np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@b)

def group_basis():
    b=np.zeros((P,5))
    for e,(i,j) in enumerate(PAIRS):
        same=(i<4)==(j<4); adj=same and abs(i%4-j%4)==1
        b[e,0 if adj else (1 if same else 2)]=1
        b[28+e,3 if same else 4]=1
    return b/np.maximum(np.linalg.norm(b,axis=0,keepdims=True),1e-12)

def estimate(j,r,mode,v,true):
    if mode=='full56': return ridge(j,r,2e-3)
    if mode in ('fisher3','fisher8'):
        rank=3 if mode=='fisher3' else 8; b=v[:,:rank]; return b@ridge(j@b,r,2e-4)
    if mode=='group5':
        b=group_basis(); return b@ridge(j@b,r,2e-4)
    ids=np.flatnonzero(abs(true)>0); a=j[:,ids]; z=np.zeros(P); z[ids]=ridge(a,r,2e-4); return z

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cases',type=int,default=60); ap.add_argument('--seeds',type=int,default=5); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True); teachers=load_teachers(); rows=[]; spectra=[]; bases={}; tic=time.time()
    for g in (10,11):
        j=jacobian(teachers[g]); u,s,vh=np.linalg.svd(j,full_matrices=False); v=vh.T; bases[g]=(j,v)
        spectra += [dict(graph=g,index=k+1,singular_value=x) for k,x in enumerate(s)]
        for seed in range(args.seeds):
            rng=np.random.default_rng(2400+100*g+seed)
            for case in range(args.cases):
                z=np.zeros(P); ic=rng.choice(28,4,False); im=rng.choice(28,4,False); z[ic]=rng.uniform(-.18,.18,4); z[28+im]=rng.uniform(-.10,.10,4)
                clean=obs(teachers[g],z)
                for noise in (0.001,0.003,0.01):
                    r=clean+rng.normal(0,noise,clean.shape)
                    for mode in ('full56','group5','fisher3','fisher8','oracle8'):
                        zh=estimate(j,r,mode,v,z); pred=obs(teachers[g],zh); proj=v[:,:8]@v[:,:8].T@z
                        active=np.flatnonzero(abs(z)>0)
                        rows.append(dict(graph=g,seed=seed,case=case,noise=noise,method=mode,
                            response_error=np.linalg.norm(pred-clean)/max(np.linalg.norm(clean),1e-12),
                            full_parameter_rmse=np.sqrt(np.mean((zh-z)**2)),active_parameter_mae=np.mean(abs(zh[active]-z[active])),
                            identifiable_coordinate_error=np.linalg.norm(v[:,:3].T@(zh-z))/max(np.linalg.norm(v[:,:3].T@z),1e-12),estimate_norm=np.linalg.norm(zh)))
    df=pd.DataFrame(rows); df.to_csv(OUT/'monte_carlo.csv',index=False); pd.DataFrame(spectra).to_csv(OUT/'singular_spectra.csv',index=False)
    agg=df.groupby(['noise','method']).agg(['mean','std']); agg.to_csv(OUT/'summary_table.csv')
    summary={'experiment':'EXP-024','test_geometries':[10,11],'cases_per_seed_geometry':args.cases,'seeds':args.seeds,'methods':['full56','group5','fisher3','fisher8','oracle8'],'runtime_seconds':time.time()-tic,
      'noise_0p003':{m:{k:float(v) for k,v in df[(df.noise==.003)&(df.method==m)][['response_error','full_parameter_rmse','active_parameter_mae','identifiable_coordinate_error','estimate_norm']].mean().items()} for m in df.method.unique()}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('matplotlib unavailable; numerical results were saved and figure generation was skipped')
        print(json.dumps(summary,ensure_ascii=False,indent=2)); return
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.2)); sp=pd.DataFrame(spectra)
    for g,d in sp.groupby('graph'): axs[0].semilogy(d['index'],d.singular_value,'o-',label=f'geometry {g}')
    axs[0].axvline(8.5,color='k',ls='--'); axs[0].set(title='Fisher/Jacobian spectrum',xlabel='Mode index',ylabel='Singular value'); axs[0].legend()
    d=df[df.noise==.003]; order=['full56','group5','fisher3','fisher8','oracle8']; axs[1].boxplot([d[d.method==m].response_error for m in order],tick_labels=order,showfliers=False); axs[1].set(title='Response reconstruction',ylabel='Relative error'); axs[1].tick_params(axis='x',rotation=20)
    axs[2].boxplot([d[d.method==m].estimate_norm for m in order],tick_labels=order,showfliers=False); axs[2].set(title='Noise amplification',ylabel=r'$||\hat z||_2$'); axs[2].tick_params(axis='x',rotation=20)
    for a,l in zip(axs,'abc'): a.text(-.14,1.05,l,transform=a.transAxes,fontweight='bold'); a.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIG/'exp24_summary.png',dpi=300); fig.savefig(FIG/'exp24_summary.pdf'); plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
