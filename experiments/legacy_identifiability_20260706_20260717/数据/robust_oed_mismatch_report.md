# Robust OED mismatch experiment

Generated: 2026-07-11 21:34:41

## Purpose

Compare uniform, random, nominal E-opt, prior-averaged E-opt, and worst-case robust E-opt frequency selection under model mismatch.

## Truth model

- asymmetric two-port
- Rac(f) on both ports
- Lsigma(f) on both ports
- Cg1/Cg2 asymmetry and weak loss terms

Ls1=1.700e-06 H, Ls2=2.400e-06 H, C12=8.500e-11 F, Cg1=9.500e-11 F, Cg2=1.500e-10 F.

## Reduced identification model

theta=[Lsigma,Cps,Cg,Rac], fixed Gg=1e-6 S.

Nominal theta: 2.000e-06, 8.000e-11, 1.200e-10, 2.000e-01.

Full-band best equivalent theta: 2.092e-06, 9.948e-11, 9.948e-11, 9.920e-02, cost=3.759e+03.

## Monte Carlo response-fit error

| strategy | K | meanFitRelErrPct | stdFitRelErrPct |
|---|---:|---:|---:|
| random_40 | 40 | 3.63 | 0.01736 |
| band_robust_eopt_40 | 40 | 4.976 | 0.02328 |
| uniform_40 | 40 | 5.481 | 0.01818 |
| full_260 | 260 | 6.782 | 0.0066 |
| prior_avg_eopt_40 | 40 | 7.264 | 0.01278 |
| nominal_eopt_40 | 40 | 29.04 | 0.02437 |
| robust_worst_eopt_40 | 40 | 29.18 | 0.02958 |

## Design diagnostics

| strategy | K | nominalLambdaMin | priorWorstLambdaMin | priorMeanLambdaMin |
|---|---:|---:|---:|---:|
| uniform_40 | 40 | 2.0277e+03 | 3.3656e+02 | 6.4647e+03 |
| random_40 | 40 | 2.0873e+02 | 6.9054e+01 | 2.9713e+02 |
| nominal_eopt_40 | 40 | 6.7470e+03 | 1.3713e+03 | 1.1384e+04 |
| prior_avg_eopt_40 | 40 | 2.3570e+03 | 4.9743e+02 | 5.0670e+03 |
| robust_worst_eopt_40 | 40 | 6.7405e+03 | 1.3696e+03 | 1.5901e+04 |
| band_robust_eopt_40 | 40 | 4.7682e+03 | 8.3864e+02 | 7.1869e+03 |
| full_260 | 260 | 6.9105e+03 | 1.4213e+03 | 1.0084e+05 |

## Initial interpretation

This is the first closed-loop robust-OED test. If robust_worst_eopt improves the worst-prior FIM but not the rich-truth response error, the next step is mismatch-aware or Bayesian OED that evaluates expected fit error rather than only reduced-model FIM.
