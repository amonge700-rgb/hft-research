"""Transparent passive 4+4 and 8+8 reference HFT graph factories."""
from __future__ import annotations
import torch
from .schema import *

def make_reference_graph(n: int = 4, *, dtype=torch.float64, device="cpu") -> HFTGraph:
    if n not in (4, 8):
        raise ValueError("reference regression models support n=4 or n=8")
    ratio = 4.0
    nodes = [PotentialNode("gnd", node_type="reference", is_reference=True,
                           is_internal=False, boundary="reference")]
    segments = []
    for winding, prefix, rtot, lktot, cser, ground_pf in (
        ("primary", "p", .18, 3.6e-6, 72e-12, (7., 4.)),
        ("secondary", "s", .18/ratio**2, 3.6e-6/ratio**2, 42e-12, (5., 3.5))):
        for k in range(n):
            nodes.append(PotentialNode(f"{prefix}{k}", winding=winding,
                is_external=(k == 0), is_internal=(k != 0), xyz_m=(0., float(k), 0.)))
            end = f"{prefix}{k+1}" if k + 1 < n else "gnd"
            segments.append(ConductorSegment(f"{prefix}_seg{k}", winding, k,
                f"{prefix}{k}", end, resistance_ohm=torch.tensor(rtot/n,dtype=dtype,device=device),
                self_inductance_h=torch.tensor(lktot/n,dtype=dtype,device=device)))
    flux = torch.cat((torch.ones(n,dtype=dtype,device=device),
                      -torch.ones(n,dtype=dtype,device=device)/ratio))
    common = (220e-6/n**2) * torch.outer(flux, flux)
    for i,s in enumerate(segments):
        s.self_inductance_h = s.self_inductance_h + common[i,i]
    mags=[]
    for i in range(2*n):
        for j in range(i+1,2*n):
            if common[i,j] != 0:
                mags.append(MagneticEdge(segments[i].id,segments[j].id,common[i,j]))
    caps=[]
    for w,prefix,cser,ends in ((0,"p",72e-12,(7.,4.)),(1,"s",42e-12,(5.,3.5))):
        for k in range(n):
            b=f"{prefix}{k+1}" if k+1<n else "gnd"
            caps.append(CapacitanceEdge(f"{prefix}{k}",b,torch.tensor(cser,dtype=dtype,device=device),"longitudinal"))
            gpf=ends[0]+(ends[1]-ends[0])*k/max(n-1,1)
            caps.append(CapacitanceEdge(f"{prefix}{k}","gnd",torch.tensor(gpf*1e-12,dtype=dtype,device=device),"ground"))
    for k in range(n):
        caps.append(CapacitanceEdge(f"p{k}",f"s{k}",torch.tensor(28.36e-12/n,dtype=dtype,device=device),"interwinding"))
    return HFTGraph(nodes,segments,mags,caps,metadata={"source":"EXP-010 transparent pilot","n":n,"turns_ratio":ratio})
