"""Batched differentiable complex nodal solver; never forms an explicit inverse."""
from dataclasses import dataclass
import torch

@dataclass
class FrequencySolution:
    frequencies_hz: torch.Tensor
    branch_impedance: torch.Tensor
    nodal_admittance: torch.Tensor
    port_admittance: torch.Tensor
    internal_recovery: torch.Tensor
    port_indices: torch.Tensor
    internal_indices: torch.Tensor

def solve_sweep(frequencies_hz, bundle, *, regularization=0.0):
    f=torch.as_tensor(frequencies_hz,dtype=bundle.A.dtype,device=bundle.A.device).reshape(-1)
    w=2*torch.pi*f
    cdtype=torch.complex128 if bundle.A.dtype==torch.float64 else torch.complex64
    A=bundle.A.to(cdtype); R=bundle.R.to(cdtype); L=bundle.L.to(cdtype)
    C=bundle.C.to(cdtype); G=bundle.G.to(cdtype)
    Z=R[None]+1j*w[:,None,None]*L[None]
    if regularization:
        Z=Z+regularization*torch.eye(Z.shape[-1],dtype=cdtype,device=Z.device)
    X=torch.linalg.solve(Z,A.T.expand(len(f),-1,-1))
    Yn=A[None]@X+G[None]+1j*w[:,None,None]*C[None]
    p=bundle.external_indices
    mask=torch.ones(Yn.shape[-1],dtype=torch.bool,device=Yn.device); mask[p]=False
    q=torch.where(mask)[0]
    Ypp=Yn[:,p][:,:,p]
    if len(q)==0:
        rec=torch.zeros((len(f),0,len(p)),dtype=cdtype,device=Yn.device); Yport=Ypp
    else:
        Ypi=Yn[:,p][:,:,q]; Yip=Yn[:,q][:,:,p]; Yii=Yn[:,q][:,:,q]
        rec=-torch.linalg.solve(Yii,Yip)
        Yport=Ypp+Ypi@rec
    return FrequencySolution(f,Z,Yn,Yport,rec,p,q)

def solve_frequency(frequency_hz,bundle,**kwargs):
    return solve_sweep(torch.as_tensor([frequency_hz],device=bundle.A.device),bundle,**kwargs)
