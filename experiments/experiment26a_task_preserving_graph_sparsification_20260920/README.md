# EXP-026A Task-preserving C/M graph sparsification

## Question

How many turn-level capacitive and magnetic coupling edges can be removed while preserving port admittance, internal turn voltage, and branch-current stress?

## Scope boundary

- This is a forward-model sparsification experiment, not online inverse identification.
- Existing EXP-020/021 COMSOL 4+4 teachers remain the high-fidelity anchor.
- The 4+4 to 32+32 cases in this first scaling run are matrix-physics stress tests, not new COMSOL models.
- Both cross-winding capacitance edges and all off-diagonal mutual-inductance edges are gated.

## Equal-budget baselines

1. distance;
2. coupling magnitude;
3. physics burden proxy from the full solution;
4. CUDA-trained shared edge gate.

All methods retain exactly the same fraction of C and M candidates.

## Outputs

- `results/per_case_budget.csv`
- `results/aggregate.csv`
- `results/summary.json`
- `results/dual_gate.pt`
- `figures/*error*.png`
- `figures/training_curve.png`

## Formal run (2026-09-20)

- GPU: NVIDIA GeForce RTX 5070 (`device=cuda`)
- training sizes: 4+4, 8+8, and 16+16 turns, three cases per size
- test sizes: 4+4, 8+8, 16+16, and unseen 32+32 turns, two cases per size
- gate training: 20 epochs
- retained-edge budgets: 40%, 60%, 80%, 90%, 95%, and 100%

At 80% C-edge retention, the physics proxy gives interwinding
displacement-current errors of 3.65%, 7.14%, 3.35%, and 5.59% for the four
test sizes. The learned gate gives 3.65%, 7.92%, 3.71%, and 6.52%.

Direct deletion of mutual-inductance entries is not a valid reduction rule.
It changes the global flux mode and can require an SPD-repair diagonal shift.
The next magnetic reduction must therefore use a PSD-preserving low-rank/modal
parameterization rather than independent binary edge deletion.

The present dense matrix solver does not become faster when an edge is gated
off. This run establishes reducibility and task error, not sparse-solver
speedup.

## Run

```powershell
python test_exp26a.py
python run_exp26a.py --device cuda
```
