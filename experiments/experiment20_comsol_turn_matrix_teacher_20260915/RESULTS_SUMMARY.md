# EXP-020 result summary

## Completed artifacts

- 4+4 turn 2D axisymmetric COMSOL teacher model.
- Reduced 8 x 8 Maxwell capacitance matrix with node order P1,P2,P3,P4,S1,S2,S3,S4.
- Positive mutual-capacitance edge table and capacitance-to-reference values.
- Full 8 x 8 inductance/mutual-inductance matrix.
- Coupling-coefficient matrix.
- All 36 electrostatic and 36 magnetic field-energy cases.

## Validation

- C symmetry relative Frobenius error: 0 by reciprocal energy assembly.
- C diagonal entries are positive and all off-diagonal entries are negative.
- Minimum C eigenvalue: 2.124785857e-12 F.
- L/M symmetry relative Frobenius error: 0 by reciprocal energy assembly.
- Minimum L/M eigenvalue: 1.302931106e-8 H.
- Maximum off-diagonal coupling coefficient: 0.7809620223.

## Model scope

The magnetic result is a quasi-static, linear-core teacher matrix with relative permeability 2000. It is suitable as the first EXP-020 L/M teacher target. It is not yet the frequency-dependent Z(f) or AC-loss matrix.

## EXP-015 matrix-solver integration

The COMSOL matrices were mapped into the EXP-015 incidence/Kron formulation:

\[
Y_n(\omega)=A(R+j\omega L)^{-1}A^T+j\omega C,
\]

with `P1...P4,S1...S4` mapped to the eight turn branches and turn-potential
nodes.  The winding chains terminate at the mathematical reference; `p0` and
`s0` are the two retained external ports.  Because EXP-020 has not extracted
frequency-dependent loss, diagonal resistance is a documented DC geometric
estimate from copper conductivity and turn dimensions.

The 1 kHz--10 MHz sweep used 241 logarithmic frequency points.  The maximum
internal KCL residual was `2.12e-13 A`, confirming that the COMSOL matrices can
be consumed by the existing physical network equations.

## Literature TC/WW/IL/IT sparsity baseline

For this single-layer 4+4 geometry, the literature baseline retains:

- all turn-to-reference capacitances (TC/reference class);
- all 16 cross-winding capacitances (WW);
- six adjacent same-layer interturn capacitances (IT);
- no non-adjacent same-layer turn capacitances.

It therefore retains 22 of 28 mutual-capacitance edges (78.57%).  The six
removed edges contain `1.2163 pF`, or 2.054% of total mutual capacitance.

Against the full COMSOL matrix:

| Metric | TC/WW/IT baseline error |
|---|---:|
| Port-admittance relative Frobenius error | `1.162e-7` |
| Maximum port-magnitude difference | `0.00457 dB` |
| Internal-voltage relative L2 error | `1.089e-6` |
| Cross-winding displacement-current relative L2 error | `2.190e-6` |

Thus, the literature shielding rule is an excellent baseline for this first
small, uniform, single-layer model.  This does not establish universality for
interleaved, multilayer, planar, shielded-edge, or geometrically asymmetric
HFTs.  Those structures are where learned edge selection must demonstrate an
advantage.

The maximum open-secondary input impedance occurs at the upper 10 MHz sweep
boundary for both models.  Therefore this sweep did not identify an interior
resonance and no resonance-preservation claim is made yet.

## Generated integration artifacts

- `run_exp20_integration.py`: COMSOL-to-network mapping and comparison;
- `test_exp20_integration.py`: matrix, incidence and sparsity tests;
- `results/integration/metrics.json`: machine-readable metrics;
- `results/integration/capacitance_edge_classes.csv`: TC/WW/IT edge labels;
- `results/integration/capacitance_literature_base.csv`: restamped sparse matrix;
- `results/integration/frequency_comparison.csv`: full/sparse frequency traces;
- `figures/exp20_matrix_and_sparse_comparison.svg`: matrix and impedance figure.

## Remaining COMSOL validation before training

1. Solve coarse, medium and fine meshes and require each important `Cij` and
   `Mij` to change by less than 1% between the final two levels.
2. Extract conductor charge/terminal quantities independently of the energy
   assembly and compare the resulting capacitance matrix.
3. Record element count, degrees of freedom and solver convergence for every
   mesh.
4. Export electric-field, magnetic-flux-density and mesh figures.
5. Repair the frequency-list creation in the connector, then extract
   frequency-dependent `Z(f)` and AC loss as a later EXP-020C stage.
