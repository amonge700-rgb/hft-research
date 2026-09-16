"""Publication figures for the EXP-020--022 integrated report.

Figure contract
Core conclusion: the COMSOL-to-inverse chain is valid and physics inversion is
the current accuracy anchor; GNN adds fast approximate inference but has not
yet surpassed physics under forward/inverse mismatch.
Archetype: schematic-led composite plus quantitative validation grids.
Backend: Python/matplotlib exclusively. Outputs: PDF/SVG/PNG at 183 mm width.
Source data: EXP-021/022 CSV and COMSOL teacher matrices; no rows excluded.
Reviewer risk: only one seed, 12 geometries, synthetic nuisance, no hardware.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
E21=REPO/'experiments'/'experiment21_comsol_gnn_minimal_20260916'
E22=REPO/'experiments'/'experiment22_model_mismatch_robust_inverse_20260916'
E20=REPO/'experiments'/'experiment20_comsol_turn_matrix_teacher_20260915'
OUT=HERE/'figures'; OUT.mkdir(parents=True,exist_ok=True)

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
 'svg.fonttype':'none','pdf.fonttype':42,'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.8,'figure.dpi':150,'savefig.dpi':600})
BLUE='#4477AA'; TEAL='#228C7B'; ORANGE='#E6863B'; RED='#C44E52'; GREY='#858585'; LIGHT='#E9EEF2'

def save(fig,name):
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight',dpi=600)
    plt.close(fig)

def workflow():
    fig,ax=plt.subplots(figsize=(7.2,2.55)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    boxes=[(.03,.57,.18,.25,'参数化 COMSOL','12 geometries\n$C,L/M$ matrices',BLUE),(.28,.57,.18,.25,'复杂前向对象','DAB windows\nnoise + nuisance',ORANGE),(.53,.57,.18,.25,'端口观测',r'$v_1,i_1,v_2,i_2$'+'\n'+r'$\widehat{Y}(f)$',TEAL),(.78,.57,.18,.25,'在线反演','5-D physical update\n+ edge-centric GNN',RED)]
    for x,y,w,h,t,s,c in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=c,ec='none',alpha=.92)); ax.text(x+w/2,y+h*.66,t,ha='center',va='center',color='white',weight='bold',fontsize=9); ax.text(x+w/2,y+h*.28,s,ha='center',va='center',color='white',fontsize=7)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((boxes[i][0]+boxes[i][2],.695),(boxes[i+1][0],.695),arrowstyle='-|>',mutation_scale=12,color='#444'))
    ax.text(.5,.35,'可微物理闭环',ha='center',weight='bold',fontsize=10)
    ax.add_patch(FancyArrowPatch((.87,.56),(.14,.56),connectionstyle='arc3,rad=-.28',arrowstyle='-|>',mutation_scale=12,color='#555'))
    ax.text(.5,.13,'当前证据：链路可运行；物理反演主导精度；GNN 提供快速近似但尚未形成精度优势',ha='center',fontsize=8,color='#333')
    save(fig,'fig01_workflow')

def teacher_trends():
    d=pd.read_csv(E21/'results'/'teacher_audit.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.35))
    axs[0].scatter(d.g_ps_mm,d.WW_total_pF,c=d.z_offset_mm,cmap='viridis',s=34,edgecolor='white'); axs[0].set(xlabel='$g_{ps}$ (mm)',ylabel='WW total (pF)',title='Cross-winding capacitance')
    axs[1].scatter(d.g_turn_mm,d.IT_total_pF,c=d.g_ps_mm,cmap='plasma',s=34,edgecolor='white'); axs[1].set(xlabel='$g_t$ (mm)',ylabel='IT total (pF)',title='Adjacent-turn capacitance')
    axs[2].scatter(d.L_main_uH,d.L_min_uH,c=d.g_ps_mm,cmap='cividis',s=34,edgecolor='white'); axs[2].set(xlabel=r'Main eigenvalue ($\mu$H)',ylabel=r'Minimum eigenvalue ($\mu$H)',title='$L/M$ physical range')
    for a,l in zip(axs,'abc'): a.text(-.16,1.06,l,transform=a.transAxes,weight='bold',fontsize=10)
    fig.suptitle('COMSOL teacher-set physical sanity checks (n=12 geometries)',y=1.04,fontsize=10)
    fig.tight_layout(); save(fig,'fig02_teacher_trends')

def matrices():
    man=pd.read_csv(E21/'data'/'comsol_teachers'/'manifest.csv'); sid=man.iloc[5].sample_id; p=E21/'data'/'comsol_teachers'/sid
    c=pd.read_csv(p/'C.csv',index_col=0).to_numpy()*1e12; l=pd.read_csv(p/'L.csv',index_col=0).to_numpy()*1e6
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0)); labels=[f'P{i}' for i in range(1,5)]+[f'S{i}' for i in range(1,5)]
    for ax,z,title,cmap in [(axs[0],c,'Maxwell $C$ matrix (pF)','coolwarm'),(axs[1],l,r'$L/M$ matrix ($\mu$H)','magma')]:
        im=ax.imshow(z,cmap=cmap,aspect='equal'); ax.set_xticks(range(8),labels,rotation=45); ax.set_yticks(range(8),labels); ax.set_title(title); fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    axs[0].text(-.18,1.05,'a',transform=axs[0].transAxes,weight='bold',fontsize=10); axs[1].text(-.18,1.05,'b',transform=axs[1].transAxes,weight='bold',fontsize=10)
    fig.tight_layout(); save(fig,'fig03_matrices')

def training():
    h1=pd.read_csv(E21/'results'/'training_history.csv'); h2=pd.read_csv(E22/'results'/'training_history.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.55))
    for ax,h,title in [(axs[0],h1,'EXP-021 matched-model training'),(axs[1],h2,'EXP-022 mismatch training')]:
        ax.plot(h.epoch,h.val_theta_mae,color=RED,lw=1.7,label='validation parameter MAE'); ax.set(xlabel='Epoch',ylabel='Log-scale MAE',title=title); ax.grid(alpha=.18); ax.legend(fontsize=7)
    for a,l in zip(axs,'ab'): a.text(-.14,1.06,l,transform=a.transAxes,weight='bold',fontsize=10)
    fig.tight_layout(); save(fig,'fig04_training')

def methods():
    d1=pd.read_csv(E21/'results'/'refinement_summary.csv'); d2=pd.read_csv(E22/'results'/'comparison_summary.csv')
    map1={'GNN direct':'GNN','physics from zero':'Physics','GNN + physics':'GNN+physics'}
    map2={'nominal':'Nominal','GNN direct':'GNN','physics mismatch':'Physics','GNN regularized physics':'GNN+physics'}
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.65))
    for ax,d,mp,title in [(axs[0],d1,map1,'Matched forward/inverse model'),(axs[1],d2,map2,'Forward/inverse model mismatch')]:
        dd=d.copy(); dd['label']=dd.method.map(mp); x=np.arange(len(dd)); w=.36
        ax.bar(x-w/2,100*dd.theta_mae,w,color=BLUE,label='Parameter MAE'); ax.bar(x+w/2,100*dd.response_error,w,color=ORANGE,label='Response error')
        ax.set_xticks(x,dd.label,rotation=15); ax.set_ylabel('Error (%)'); ax.set_title(title); ax.grid(axis='y',alpha=.18); ax.legend(fontsize=7)
    for a,l in zip(axs,'ab'): a.text(-.14,1.06,l,transform=a.transAxes,weight='bold',fontsize=10)
    fig.tight_layout(); save(fig,'fig05_method_comparison')

def distributions():
    d=pd.read_csv(E21/'results'/'test_cases.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.55))
    axs[0].hist(100*d.theta_mae,bins=14,color=BLUE,alpha=.85); axs[0].axvline(100*d.theta_mae.mean(),color=RED,ls='--',label=f'mean={100*d.theta_mae.mean():.2f}%'); axs[0].set(xlabel='Five-parameter MAE (%)',ylabel='Test cases',title='EXP-021 unseen geometries'); axs[0].legend(fontsize=7)
    axs[1].scatter(100*d.nominal_response_error,100*d.response_error,c=d.snr_db,cmap='viridis',s=20,alpha=.8); lim=max(100*d.nominal_response_error.max(),100*d.response_error.max()); axs[1].plot([0,lim],[0,lim],ls='--',color=GREY,lw=1); axs[1].set(xlabel='Nominal response error (%)',ylabel='GNN response error (%)',title='Per-case improvement'); cb=fig.colorbar(axs[1].collections[0],ax=axs[1]); cb.set_label('SNR (dB)')
    for a,l in zip(axs,'ab'): a.text(-.14,1.06,l,transform=a.transAxes,weight='bold',fontsize=10)
    fig.tight_layout(); save(fig,'fig06_test_distribution')

def evidence_summary():
    fig,ax=plt.subplots(figsize=(7.2,2.3)); ax.axis('off')
    cols=[(.02,'Established',TEAL,['12 COMSOL geometries\npass matrix checks','Multi-window + Kron +\nCUDA loop runs','Physics inversion reaches\n~1% response error']),(.35,'Promising, not proven',BLUE,['GNN lowers nominal\nresponse error','Fast direct estimate /\nuseful initializer','Turn-level graph accepts\nC and L/M edges']),(.68,'Still missing',RED,['No hardware DAB\nvalidation','No multi-seed\nuncertainty','GNN does not beat\nphysics accuracy'])]
    for x,t,c,items in cols:
        ax.add_patch(FancyBboxPatch((x,.08),.29,.82,boxstyle='round,pad=.015',fc=c,alpha=.10,ec=c,lw=1.2)); ax.text(x+.145,.78,t,ha='center',weight='bold',color=c,fontsize=10)
        for k,it in enumerate(items): ax.text(x+.025,.61-k*.205,u'• '+it,fontsize=7,color='#222',va='top',linespacing=1.25)
    save(fig,'fig07_evidence_summary')

def exp20_validation():
    mesh=pd.read_csv(E20/'data'/'convergence'/'mesh_solver_metrics.csv')
    conv=pd.read_csv(E20/'data'/'convergence'/'mesh_convergence_summary.csv')
    fr=pd.read_csv(E20/'results'/'integration'/'frequency_comparison.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.55))
    x=np.arange(len(mesh)); axs[0].bar(x,mesh.elements,color=[GREY,BLUE,TEAL]); axs[0].set_xticks(x,mesh.mesh); axs[0].set_yscale('log'); axs[0].set(ylabel='Mesh elements (log scale)',title='Mesh refinement')
    labels=[]; diag=[]; mutual=[]
    for _,r in conv.iterrows(): labels.append(f"{r['matrix']}\n{r['from'][0].upper()}-{r['to'][0].upper()}"); diag.append(100*r.max_diagonal_relative_change); mutual.append(100*r.max_important_mutual_relative_change)
    xx=np.arange(len(labels)); w=.35; axs[1].bar(xx-w/2,diag,w,color=BLUE,label='diagonal'); axs[1].bar(xx+w/2,mutual,w,color=ORANGE,label='important mutual'); axs[1].axhline(1,color=RED,ls='--',lw=1,label='1% criterion'); axs[1].set_xticks(xx,labels); axs[1].set(ylabel='Maximum relative change (%)',title='Matrix convergence'); axs[1].legend(fontsize=6)
    axs[2].loglog(fr.frequency_hz,fr.abs_zin_full_ohm,color=BLUE,lw=1.8,label='full COMSOL C'); axs[2].loglog(fr.frequency_hz,fr.abs_zin_literature_ohm,color=ORANGE,ls='--',lw=1.2,label='TC/WW/IT sparse'); axs[2].set(xlabel='Frequency (Hz)',ylabel='$|Z_{in}|$ ($\Omega$)',title='Matrix-to-network integration'); axs[2].legend(fontsize=6)
    for a,l in zip(axs,'abc'): a.text(-.16,1.06,l,transform=a.transAxes,weight='bold',fontsize=10); a.grid(alpha=.18)
    fig.tight_layout(); save(fig,'fig08_exp20_validation')

def exp20_matrix_heatmaps():
    """Show the complete verified EXP-020 teacher matrices without subsampling."""
    labels=[f'P{i}' for i in range(1,5)]+[f'S{i}' for i in range(1,5)]
    c=pd.read_csv(E20/'data'/'raw'/'capacitance_maxwell.csv',index_col=0).to_numpy()*1e12
    l=pd.read_csv(E20/'data'/'raw'/'inductance_matrix.csv',index_col=0).to_numpy()*1e9
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.25))
    cmax=np.max(np.abs(c)); im0=axs[0].imshow(c,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-cmax,vcenter=0,vmax=cmax))
    im1=axs[1].imshow(l,cmap='viridis',vmin=np.min(l),vmax=np.max(l))
    for ax,title in zip(axs,['Maxwell capacitance $C$ (pF)','Inductance / mutual inductance $L/M$ (nH)']):
        ax.set_xticks(range(8),labels,rotation=45,ha='right'); ax.set_yticks(range(8),labels); ax.set_title(title,pad=7)
        ax.axvline(3.5,color='white',lw=1.2); ax.axhline(3.5,color='white',lw=1.2)
    for i in range(8):
        for j in range(8):
            axs[0].text(j,i,f'{c[i,j]:.1f}',ha='center',va='center',fontsize=5.2,color='white' if abs(c[i,j])>.42*cmax else '#222')
            frac=(l[i,j]-l.min())/(l.max()-l.min()); axs[1].text(j,i,f'{l[i,j]:.1f}',ha='center',va='center',fontsize=5.2,color='white' if frac<.23 or frac>.78 else '#222')
    cb0=fig.colorbar(im0,ax=axs[0],fraction=.046,pad=.04); cb0.set_label('pF')
    cb1=fig.colorbar(im1,ax=axs[1],fraction=.046,pad=.04); cb1.set_label('nH')
    for a,label in zip(axs,'ab'): a.text(-.18,1.07,label,transform=a.transAxes,weight='bold',fontsize=10)
    fig.suptitle('EXP-020 complete turn-level teacher matrices (P1--P4, S1--S4)',y=1.01,fontsize=10)
    fig.tight_layout(); save(fig,'fig09_exp20_matrix_heatmaps')

if __name__=='__main__':
    workflow(); teacher_trends(); matrices(); training(); methods(); distributions(); evidence_summary(); exp20_validation(); exp20_matrix_heatmaps()
