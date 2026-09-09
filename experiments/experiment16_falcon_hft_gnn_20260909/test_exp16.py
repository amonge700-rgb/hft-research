import importlib.util, sys
from pathlib import Path
import torch

P=Path(__file__).with_name("run_exp16.py")
S=importlib.util.spec_from_file_location("exp16",P); m=importlib.util.module_from_spec(S); sys.modules["exp16"]=m; S.loader.exec_module(m)

def test_positive_structured_matrices():
    t=m.make_template(4,"cpu"); b=m.scaled_bundle(t,torch.zeros(8,dtype=torch.float64))
    assert torch.linalg.eigvalsh(b.L).min()>0
    assert torch.linalg.eigvalsh(b.C).min()>-1e-18

def test_response_and_gradients():
    t=m.make_template(4,"cpu"); z=torch.zeros(8,dtype=torch.float64,requires_grad=True)
    y=m.response(t,z,torch.logspace(4,7,6,dtype=torch.float64)); y.square().mean().backward()
    assert y.shape==(6,6) and torch.isfinite(z.grad).all()

def test_models_shape():
    t=m.make_template(4,"cpu"); node,adj=m.graph_tensors(t,torch.zeros(8,dtype=torch.float64))
    for hetero in (False,True):
        net=m.RelationMP(d=16,layers=2,hetero=hetero)
        y=net(node[None],adj[None],torch.linspace(4,7,6)[None])
        assert y.shape==(1,6,6)

if __name__ == "__main__":
    test_positive_structured_matrices(); test_response_and_gradients(); test_models_shape()
    print("3 tests passed")
