# Literature-feature to online-pulse bridge demo

Generated: 2026-07-12 00:59:09

## Purpose

Build a first-layer innovation baseline: move literature-style offline broadband observables (Zoc, Zsc, voltage transfer) toward online steep-pulse reconstruction and feature-based parameter extraction.

## Model

Estimated theta=[Lsigma,Cps,Cg1,Cg2,Rac]. Fixed Lm=4.200e-04 H, Rm=6.000e+03 Ohm.

Truth theta: 2.0000e-06, 8.5000e-11, 9.5000e-11, 1.5000e-10, 1.2000e-01.

Initial nominal theta: 2.2000e-06, 7.5000e-11, 1.1000e-10, 1.3000e-10, 1.8000e-01.

Pulse: fs=1.200e+08 Hz, N=65536, tr=5.000e-08 s, width=7.000e-06 s.

## Results

| method | K | meanResponseErrPct | err Lsigma | err Cps | err Cg1 | err Cg2 | err Rac |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline_full_sweep | 300 | 0.3997 | -0.2095 | 1.219 | 0.06256 | -0.3639 | -0.006334 |
| offline_feature_points | 54 | 0.03481 | -0.03552 | 0.04459 | 0.05835 | 0.04251 | -0.01866 |
| pulse_fft_full_valid | 193 | 25.41 | 0.1078 | -90.91 | -7.113 | 4.549 | -0.03613 |
| pulse_fft_feature_points | 54 | 429.6 | -0.08298 | 102.4 | 7.96 | -4.961 | -0.0635 |
| pulse_fft_reliable_features | 54 | 26.14 | 0.4674 | -83.93 | 0.2129 | 0.03796 | -0.03568 |

## Interpretation

The offline full sweep is the upper baseline. Feature-point extraction tests whether literature-style characteristic frequencies carry enough information. Pulse-FFT methods test whether the same observables can be reconstructed from a finite steep-pulse waveform. This is the bridge from offline literature methods to online natural excitation.
