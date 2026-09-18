import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp20", ROOT / "run_exp20_integration.py")
exp20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp20)


class Exp20IntegrationTests(unittest.TestCase):
    def setUp(self):
        _, self.c = exp20.read_square("capacitance_maxwell.csv")
        _, self.l = exp20.read_square("inductance_matrix.csv")

    def test_comsol_matrices_are_positive_definite(self):
        self.assertGreater(np.linalg.eigvalsh(self.c).min(), 0)
        self.assertGreater(np.linalg.eigvalsh(self.l).min(), 0)

    def test_incidence_has_one_branch_per_turn(self):
        a = exp20.branch_incidence()
        self.assertEqual(a.shape, (8, 8))
        self.assertTrue(np.all((np.abs(a).sum(axis=0) >= 1) & (np.abs(a).sum(axis=0) <= 2)))

    def test_literature_base_preserves_reference_and_ww(self):
        base, rows = exp20.literature_base_capacitance(self.c)
        np.testing.assert_allclose(base.sum(1), self.c.sum(1), rtol=1e-12, atol=1e-24)
        np.testing.assert_allclose(base[:4, 4:], self.c[:4, 4:], rtol=0, atol=0)
        self.assertTrue(all(x["retained"] for x in rows if x["edge_class"] in ("WW", "IT")))
        self.assertTrue(all(not x["retained"] for x in rows if x["edge_class"] == "same_layer_nonadjacent"))


if __name__ == "__main__": unittest.main()
