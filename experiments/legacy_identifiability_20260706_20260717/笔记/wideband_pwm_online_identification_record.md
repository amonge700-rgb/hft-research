# Wideband mechanism model online PWM identification

This experiment uses Liu 2016 Tables II-IV as the mechanism-model anchor.
It reconstructs LV input impedance from time-domain voltage/current; no sweep data enter the online estimator.

## Simulation settings

- fs = 1000000 Hz
- N = 131072
- averages = 10
- voltage noise = 0.0005 relative RMS
- current noise = 0.0030 relative RMS
- normalized excitation-energy threshold = 1.0e-06

## Method ranking

- dual_window_targeted: MAE f1 0.775%, f2 0.026%; missing 0.0%; penalized score 0.401
- random_prbs: MAE f1 1.637%, f2 0.000%; missing 0.0%; penalized score 0.819
- targeted_binary_pwm: MAE f1 5.928%, f2 0.000%; missing 0.0%; penalized score 2.964
- conventional_20k_pwm: MAE f1 NaN%, f2 5.562%; missing 50.0%; penalized score 55.562
- single_steep_pulse: MAE f1 6.873%, f2 NaN%; missing 50.0%; penalized score 56.873

## Interpretation

The sweep solution is the model truth; Table IV is the literature target. The comparison therefore separates online reconstruction error from the small table-model reproduction error.
The dual-window method uses a long low-band record for f1 and a shorter high-band record for f2.
This remains a simulation study and must later be validated with measured PWM voltage/current.
