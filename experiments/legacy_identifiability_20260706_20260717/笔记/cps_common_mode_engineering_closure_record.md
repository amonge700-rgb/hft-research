# Cps online identification and common-mode-current engineering closure

## Scope

This model-level experiment uses the Liu 2017 prototype-1 three-capacitance model.
Known capacitance is added across the winding ports, a fixed noisy binary excitation is used to recover the transfer zero, and the estimated Cps predicts displacement current.
It is not a loaded DAB experiment and does not establish a unique relationship between Cps and insulation aging.

## Settings

- Baseline Cps: 28.36 pF
- Added capacitance: [0.0, 2.5, 5.0, 10.0, 20.0] pF
- Sampling rate: 50.0 MHz
- FFT length: 262144
- Spectral averages: 16
- Repeated windows per state: 8
- Reference dv/dt: 16.0 kV/us

## Results

| Added Cps (pF) | True Cps (pF) | Estimated Cps (pF) | Std (pF) | Cps error (%) | True peak current (A) | Predicted peak current (A) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 28.360 | 28.210 | 0.148 | -0.530 | 0.4538 | 0.4514 |
| 2.50 | 30.860 | 30.694 | 0.049 | -0.537 | 0.4938 | 0.4911 |
| 5.00 | 33.360 | 33.401 | 0.013 | 0.121 | 0.5338 | 0.5344 |
| 10.00 | 38.360 | 38.742 | 0.055 | 0.997 | 0.6138 | 0.6199 |
| 20.00 | 48.360 | 49.108 | 0.410 | 1.546 | 0.7738 | 0.7857 |

- Mean absolute Cps error: 0.746%
- Mean absolute common-mode-current prediction error: 0.746%
- Smallest adjacent Cps increment with non-overlapping 95% single-window intervals: 2.50 pF

## Interpretation

The result closes the chain from an online frequency-response feature to a directly measurable engineering quantity.
The current result is intentionally narrow: it demonstrates sensitivity to known Cps perturbations under an open-secondary literature model.
The next experiment must replace the open-secondary transfer function with loaded multiport observations and test load, temperature, grounding, and measurement-chain nuisance variables.
