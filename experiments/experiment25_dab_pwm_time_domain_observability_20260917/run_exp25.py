"""EXP-025: DAB PWM time-domain observation versus ideal port observations.

This is a switching-waveform/LTI co-simulation, not a semiconductor device
model. Finite-rise-time DAB bridge harmonics are synthesized in time, noisy
v1/i1/v2/i2 windows are sampled, and the two-port Y matrix is reconstructed.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent
E24=ROOT.parent/'experiment24_fisher_identifiable_graph_modes_20260917'
sys.path.insert(0,str(E24))
from run_exp24 import load_teachers,rebuild_c,rebuild_l,solve_port_np,PAIRS

OUT=ROOT/'results'; FS=20_000.; H=np.array([1,3,5,7,9,11]); FREQ=FS*H
NPP=256; PERIODS=32; NS=NPP*PERIODS; P=56

WINDOWS={
 'multi_pwm':[(.86,8),(.96,17),(1.05,29),(1.14,41)],
 'degenerate_pwm':[(1.0,20)]*4,
 'single_pwm':[(1.0,20)],
}

def bridge_phasors(ratio,phase_deg,vdc=400.,tr=20e-9,dead=100e-9):
    amp=4*vdc/(np.pi*H)*np.sinc(FREQ*tr)*np.cos(np.pi*FREQ*dead)
    v1=amp.astype(complex); v2=ratio*amp*np.exp(-1j*H*np.deg2rad(phase_deg))
    return np.vstack((v1,v2)).T

def synthesize(x):
    n=np.arange(NS); ans=np.zeros((NS,x.shape[1]))
    for k,h in enumerate(H): ans+=np.real(x[k][None,:]*np.exp(1j*2*np.pi*h*n/NPP)[:,None])
    return ans

def extract(x):
    xf=np.fft.rfft(x,axis=0); return np.stack([2*xf[int(h*PERIODS)]/NS for h in H])

def observe(t,z,mode='multi_pwm',snr_db=None,rng=None):
    c=rebuild_c(t['C'],z[:28]); l=rebuild_l(t['L'],z[28:]); y=solve_port_np(c,l,FREQ)
    vs=[]; ins=[]
    for ratio,phi in WINDOWS[mode]:
        vp=bridge_phasors(ratio,phi); ip=np.einsum('kij,kj->ki',y,vp)
        vt=synthesize(vp); it=synthesize(ip)
        if snr_db is not None:
            rng=np.random.default_rng() if rng is None else rng
            for a in (vt,it):
                sigma=np.sqrt(np.mean(a*a))*10**(-snr_db/20); a+=rng.normal(0,sigma,a.shape)
        vs.append(extract(vt)); ins.append(extract(it))
    yh=[]; cond=[]
    for k in range(len(H)):
        v=np.stack([x[k] for x in vs],1); i=np.stack([x[k] for x in ins],1)
        cond.append(np.linalg.cond(v@v.conj().T))
        yh.append(i@np.linalg.pinv(v))
    return np.stack(yh),np.asarray(cond),y

def feature(t,z,mode):
    y,_,_=observe(t,z,mode); y0,_,_=observe(t,np.zeros(P),mode)
    q=(y-y0)/(np.abs(y0)+1e-12); return np.r_[q.real.ravel(),q.imag.ravel()]

def jacobian(t,mode,h=2e-4):
    j=np.zeros((48,P)); z=np.zeros(P)
    for k in range(P):
        zp=z.copy(); zm=z.copy(); zp[k]=h; zm[k]=-h
        j[:,k]=(feature(t,zp,mode)-feature(t,zm,mode))/(2*h)
    return j

def ridge(a,b,lam=2e-4): return np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@b)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cases',type=int,default=100); ap.add_argument('--seeds',type=int,default=5); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); teachers=load_teachers(); rows=[]; spectra=[]; diagnostics=[]; tic=time.time()
    for g in (10,11):
        t=teachers[g]; bases={}
        for mode in WINDOWS:
            j=jacobian(t,mode); _,s,vh=np.linalg.svd(j,full_matrices=False); bases[mode]=(j,vh.T[:,:3])
            spectra += [dict(graph=g,mode=mode,index=k+1,singular_value=x) for k,x in enumerate(s)]
            _,cond,ytrue=observe(t,np.zeros(P),mode)
            yhat,_,_=observe(t,np.zeros(P),mode)
            diagnostics.append(dict(graph=g,mode=mode,median_excitation_condition=float(np.median(cond)),nominal_y_error=float(np.linalg.norm(yhat-ytrue)/np.linalg.norm(ytrue))))
        for seed in range(args.seeds):
            rng=np.random.default_rng(2500+100*g+seed)
            for case in range(args.cases):
                z=np.zeros(P); z[rng.choice(28,4,False)]=rng.uniform(-.18,.18,4); z[28+rng.choice(28,4,False)]=rng.uniform(-.1,.1,4)
                for mode in WINDOWS:
                    j,b=bases[mode]; yn,cond,yt=observe(t,z,mode,45,rng); y0,_,_=observe(t,np.zeros(P),mode)
                    q=(yn-y0)/(np.abs(y0)+1e-12); r=np.r_[q.real.ravel(),q.imag.ravel()]
                    zh=b@ridge(j@b,r); yp=solve_port_np(rebuild_c(t['C'],zh[:28]),rebuild_l(t['L'],zh[28:]),FREQ)
                    rows.append(dict(graph=g,seed=seed,case=case,mode=mode,
                        y_measurement_error=np.linalg.norm(yn-yt)/np.linalg.norm(yt),
                        response_reconstruction_error=np.linalg.norm(yp-yt)/np.linalg.norm(yt),
                        parameter_rmse=np.sqrt(np.mean((zh-z)**2)),
                        active_parameter_mae=np.mean(np.abs(zh[np.flatnonzero(z)]-z[np.flatnonzero(z)])),
                        median_excitation_condition=np.median(cond)))
    df=pd.DataFrame(rows); df.to_csv(OUT/'monte_carlo.csv',index=False)
    pd.DataFrame(spectra).to_csv(OUT/'singular_spectra.csv',index=False); pd.DataFrame(diagnostics).to_csv(OUT/'observation_diagnostics.csv',index=False)
    summary={'experiment':'EXP-025','model_scope':'finite-rise-time PWM waveform plus linear turn-level network; no semiconductor device switching',
      'cases_per_seed_geometry':args.cases,'seeds':args.seeds,'snr_db':45,'runtime_seconds':time.time()-tic,
      'metrics':{m:{k:float(v) for k,v in df[df['mode']==m][['y_measurement_error','response_reconstruction_error','parameter_rmse','active_parameter_mae','median_excitation_condition']].mean().items()} for m in WINDOWS}}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
