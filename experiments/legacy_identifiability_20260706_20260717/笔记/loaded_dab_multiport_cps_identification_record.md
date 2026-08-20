# Loaded DAB operating-window multiport Cps identification

## Scope and model boundary

The bridge terminal voltages are prescribed DAB-like square waves. Seven operating windows vary phase shift, voltage ratio, and duty ratio. The Liu 2017 prototype-1 two-port model maps the two terminal voltages to both terminal currents.
This is not yet a complete switching-level DAB with controller, dead time, semiconductor parasitics, or a measured load.

## Identification chain

1. Reconstruct the full two-port admittance matrix from v1, i1, v2, and i2 over multiple operating windows.
2. Jointly estimate leakage inductance and Cps from the reconstructed mutual admittance Y12.
3. Compare against fixed-Ls estimation and the incorrect practice of treating loaded I1/V1 as Y11.
4. Use estimated Cps to predict the interwinding displacement-current waveform.

## Aggregate results

- Multiport joint Ls+Cps mean absolute Cps error: 0.047%
- Multiport fixed-Ls mean absolute Cps error: 61.400%
- Loaded single-port mean absolute Cps error: 85.561%
- Repeated trials per state: 6
- Valid reconstructed harmonics in the representative case: 2713

## Interpretation

The full port equations separate transformer mutual admittance from the operating-point-dependent V2/V1 contribution. The single-port apparent admittance cannot make this separation and therefore attributes load/phase-shift effects to Cps.
Jointly estimating leakage inductance prevents leakage drift from being misidentified as capacitance drift. The estimated Cps also closes the engineering chain to a switching-edge displacement-current prediction.

## Next validation

Replace prescribed bridge voltages with a switching-level DAB model and then a low-voltage bench. Add known capacitors across the isolation barrier, independently measure the offline reference Cps, and test load, grounding, temperature, dead time, and probe-chain disturbances.
