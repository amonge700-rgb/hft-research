# EXP-020 remaining COMSOL task

Use the existing model
`model/exp20_teacher_4p4_axisym_C_LM.mph`; do not overwrite the accepted raw
matrices.  Save each new model revision and all new outputs under this EXP-020
directory.

1. Run electrostatic and stationary magnetic extraction on three mesh levels:
   coarse, medium and fine.  Export `C` and `L/M` for each level, together with
   element count, degrees of freedom, solve time and convergence status.
2. Compute elementwise relative changes from coarse to medium and medium to
   fine.  Require important mutual terms and all diagonal terms to change by
   less than 1% for acceptance.
3. Independently extract terminal/conductor charge for every 1 V excitation.
   Assemble `C_charge` from `Q=C V` and compare it with the existing reciprocal
   energy matrix.  Report matrix Frobenius error and worst elementwise error.
4. Export publication-readable images of geometry, final mesh, representative
   P1 electric field, P1 magnetic flux density, and one P1+S1 coupled case.
5. Do not add DAB switching, thermal physics, nonlinear core material or GNN
   training in this task.
