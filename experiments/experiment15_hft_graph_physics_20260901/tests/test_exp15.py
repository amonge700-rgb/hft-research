import sys, unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from hft_graph import *
from hft_graph.reference_models import make_reference_graph
from hft_graph.validation import validate_graph
from physics import *

class TestExp15(unittest.TestCase):
 def setUp(self): self.g=make_reference_graph(4); self.b=graph_to_matrices(self.g)
 def test_schema_validation(self): self.assertEqual(validate_graph(self.g)["segments"],8)
 def test_shapes_4x4(self): self.assertEqual(tuple(self.b.A.shape),(8,8))
 def test_shapes_8x8(self):
  b=graph_to_matrices(make_reference_graph(8)); self.assertEqual(tuple(b.C.shape),(16,16))
 def test_incidence_columns(self): self.assertTrue(torch.all(self.b.A.abs().sum(0)>=1))
 def test_roundtrip_canonical(self):
  b2=graph_to_matrices(matrix_to_graph(self.b)); self.assertTrue(torch.allclose(self.b.C,b2.C))
 def test_L_symmetric_pd(self):
  self.assertTrue(torch.allclose(self.b.L,self.b.L.T)); self.assertGreater(float(torch.linalg.eigvalsh(self.b.L).min()),0)
 def test_C_symmetric_psd(self):
  self.assertTrue(torch.allclose(self.b.C,self.b.C.T)); self.assertGreaterEqual(float(torch.linalg.eigvalsh(self.b.C).min()),-1e-20)
 def test_capacitance_sign(self): self.assertLessEqual(float((self.b.C-torch.diag(torch.diag(self.b.C))).max()),1e-30)
 def test_kron_shapes(self): self.assertEqual(tuple(solve_sweep([1e3,1e6],self.b).port_admittance.shape),(2,2,2))
 def test_reciprocity(self):
  y=solve_sweep([1e5],self.b).port_admittance; self.assertLess(float(torch.linalg.norm(y-y.transpose(-1,-2))),1e-12)
 def test_internal_kcl(self):
  s=solve_sweep([1e4,1e6],self.b); st=recover_internal_states(s,torch.tensor([1+0j,0+0j])); self.assertLess(float(st["internal_kcl_residual"].abs().max()),1e-9)
 def test_passivity(self):
  y=solve_sweep([1e4,1e6],self.b).port_admittance; self.assertGreaterEqual(physical_diagnostics(self.b,y)["port_real_min_eig"],-1e-10)
 def test_cps_gradient(self):
  x=self.g.capacitance_edges[-1].capacitance_f.detach().clone().requires_grad_(True); self.g.capacitance_edges[-1].capacitance_f=x
  loss=solve_sweep([1e6],graph_to_matrices(self.g)).port_admittance.abs().square().sum(); loss.backward(); self.assertTrue(torch.isfinite(x.grad)); self.assertNotEqual(float(x.grad),0)
 def test_resistance_gradient(self):
  x=self.g.conductor_segments[0].resistance_ohm.detach().clone().requires_grad_(True); self.g.conductor_segments[0].resistance_ohm=x
  solve_sweep([1e5],graph_to_matrices(self.g)).port_admittance.abs().sum().backward(); self.assertTrue(torch.isfinite(x.grad))
 def test_self_inductance_gradient(self):
  x=self.g.conductor_segments[0].self_inductance_h.detach().clone().requires_grad_(True); self.g.conductor_segments[0].self_inductance_h=x
  solve_sweep([1e5],graph_to_matrices(self.g)).port_admittance.abs().sum().backward(); self.assertTrue(torch.isfinite(x.grad)); self.assertNotEqual(float(x.grad),0)
 def test_mutual_inductance_gradient(self):
  x=self.g.magnetic_edges[0].mutual_inductance_h.detach().clone().requires_grad_(True); self.g.magnetic_edges[0].mutual_inductance_h=x
  solve_sweep([1e5],graph_to_matrices(self.g)).port_admittance.abs().sum().backward(); self.assertTrue(torch.isfinite(x.grad)); self.assertNotEqual(float(x.grad),0)
 def test_ground_capacitance_gradient(self):
  edge=next(e for e in self.g.capacitance_edges if e.subtype=="ground")
  x=edge.capacitance_f.detach().clone().requires_grad_(True); edge.capacitance_f=x
  solve_sweep([1e6],graph_to_matrices(self.g)).port_admittance.abs().sum().backward(); self.assertTrue(torch.isfinite(x.grad)); self.assertNotEqual(float(x.grad),0)
 def test_gradient_finite_difference(self):
  e=self.g.capacitance_edges[-1]; x=e.capacitance_f.detach().clone().requires_grad_(True); e.capacitance_f=x
  fun=lambda: solve_sweep([1e6],graph_to_matrices(self.g)).port_admittance[0,0,1].imag
  y=fun(); y.backward(); ad=float(x.grad); eps=1e-16
  with torch.no_grad(): x.add_(eps); yp=float(fun()); x.sub_(2*eps); ym=float(fun()); x.add_(eps)
  self.assertLess(abs(ad-(yp-ym)/(2*eps))/max(abs(ad),1),1e-5)
 def test_invalid_duplicate_node(self):
  self.g.potential_nodes.append(self.g.potential_nodes[0]); self.assertRaises(ValueError,validate_graph,self.g)
 def test_singular_model_failure(self):
  g=make_reference_graph(4)
  for s in g.conductor_segments: s.resistance_ohm=torch.tensor(0.); s.self_inductance_h=torch.tensor(0.)
  g.magnetic_edges=[]
  with self.assertRaises(RuntimeError): solve_sweep([1e3],graph_to_matrices(g))
if __name__=='__main__': unittest.main()
