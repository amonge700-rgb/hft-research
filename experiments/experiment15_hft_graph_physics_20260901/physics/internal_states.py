import torch

def recover_internal_states(solution, port_voltage):
    vp=torch.as_tensor(port_voltage,dtype=solution.port_admittance.dtype,
                       device=solution.port_admittance.device)
    if vp.ndim==1: vp=vp.expand(len(solution.frequencies_hz),-1)
    vi=torch.einsum("fip,fp->fi",solution.internal_recovery,vp)
    v=torch.zeros((len(vp),solution.nodal_admittance.shape[-1]),dtype=vp.dtype,device=vp.device)
    v[:,solution.port_indices]=vp; v[:,solution.internal_indices]=vi
    i=torch.einsum("fij,fj->fi",solution.nodal_admittance,v)
    return {"node_voltage":v,"node_current":i,"port_current":i[:,solution.port_indices],
            "internal_kcl_residual":i[:,solution.internal_indices]}
