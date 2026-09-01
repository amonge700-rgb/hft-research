import torch

def physical_diagnostics(bundle, port_admittance=None):
    L=(bundle.L+bundle.L.T)/2; C=(bundle.C+bundle.C.T)/2; G=(bundle.G+bundle.G.T)/2
    out={"L_symmetry":float(torch.linalg.norm(bundle.L-bundle.L.T).detach().cpu()),
         "C_symmetry":float(torch.linalg.norm(bundle.C-bundle.C.T).detach().cpu()),
         "L_min_eig":float(torch.linalg.eigvalsh(L).min().detach().cpu()),
         "C_min_eig":float(torch.linalg.eigvalsh(C).min().detach().cpu()),
         "G_min_eig":float(torch.linalg.eigvalsh(G).min().detach().cpu()),
         "R_min_diag":float(torch.diag(bundle.R).min().detach().cpu())}
    if port_admittance is not None:
        herm=(port_admittance+port_admittance.mH)/2
        out["port_real_min_eig"]=float(torch.linalg.eigvalsh(herm).min().detach().cpu())
        out["port_reciprocity_error"]=float(torch.linalg.norm(port_admittance-port_admittance.transpose(-1,-2)).detach().cpu())
    return out
