# EXP-027 DAB multi-window online modal identification

This experiment is the main online-identification continuation of EXP-023--025.

It replaces the ill-conditioned 56 independent turn-pair updates with three
physical coordinates:

1. aggregate cross-winding capacitance scale;
2. dominant inductance eigenmode scale;
3. second dominant inductance eigenmode scale.

The capacitance construction preserves the Maxwell capacitance structure. The
inductance construction preserves symmetry and positive definiteness. A compact
spectral latent-attention model reads multi-window DAB two-port spectra, and a
Gauss-Newton physics stage enforces response consistency.

```powershell
python test_exp27.py
python run_exp27.py --device cuda
```
