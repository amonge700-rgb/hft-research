# EXP-021: COMSOL-to-GNN minimal identification loop

## Objective

Build the smallest defensible multi-geometry dataset for a turn-level graph
prior and a low-dimensional online identification experiment.  This experiment
does **not** claim hardware validation or direct DAB control.

## Geometry design

The first factorial set contains 12 geometries:

- primary-secondary radial gap: 1.0, 1.5, 2.0 mm;
- same-winding turn gap: 0.15, 0.25 mm;
- secondary axial offset: 0, 0.625 mm.

The source EXP-020 model used hard-coded rectangle positions.  Therefore this
experiment explicitly rewrites all P/S rectangle coordinates before every
geometry build; merely changing global parameter strings is not accepted as a
geometry sweep.

## Teacher extraction

- Medium mesh (`hauto=3`) for all 12 exploratory samples.
- Maxwell capacitance matrix: 8 independent 1 V terminal-charge solves.
- L/M matrix: 36 reciprocal magnetic-energy cases.
- Each sample stores matrices, turn metadata, solver timing, eigenvalue and
  symmetry checks, and the exact geometry parameters.

The validated EXP-020 fine mesh remains the high-fidelity anchor.  EXP-021 is
an exploratory multi-geometry set; corner samples must later be checked on the
fine mesh before publication claims.

## Intended minimum inverse problem

The online estimator must not infer every matrix entry independently.  It
estimates a small set of grouped log-scale corrections for IT, WW/reference,
main-flux and leakage subspaces.  The decoded matrices are passed through the
same incidence/Kron solver used in EXP-015.

## Reproduction

```powershell
D:\高频变压器\.venv-gnn\Scripts\python.exe run_minimal_inverse.py --epochs 800 --states 60
D:\高频变压器\.venv-gnn\Scripts\python.exe evaluate_refinement.py
```

See `实验二十一阶段报告.md` for equations, split policy, results, and limitations.
