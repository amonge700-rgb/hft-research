"""Loaded DAB operating-window identification of interwinding capacitance.

The DAB bridge terminal voltages are prescribed square-wave operating windows
with different phase shifts, voltage ratios, and duty ratios. The transformer
is the Liu 2017 prototype-1 two-port model. For every frequency, the full
two-port admittance matrix is reconstructed from v1, i1, v2, and i2 across the
operating windows. Cps and leakage inductance are then jointly fitted from Y12.

This is a model-level bridge-voltage experiment. It is not yet a complete DAB
switching model with control loops, dead time, semiconductor parasitics, or a
measured load.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


SEED = 20260714
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
NOTE_DIR = ROOT / "笔记"


def two_port_y(freq_hz: np.ndarray, params: dict) -> np.ndarray:
    """Return Y with shape (frequency, 2, 2)."""
    freq = np.maximum(np.asarray(freq_hz, dtype=float), 1e-3)
    s = 1j * 2 * np.pi * freq
    yl = 1.0 / (params["rs"] + s * params["ls"])
    ym = 1.0 / params["rm"] + 1.0 / (s * params["lm"])
    y = np.empty((freq.size, 2, 2), dtype=complex)
    y[:, 0, 0] = params["n"] ** 2 * yl + ym + s * (params["cp"] + params["cps"])
    y[:, 0, 1] = -params["n"] * yl - s * params["cps"]
    y[:, 1, 0] = y[:, 0, 1]
    y[:, 1, 1] = yl + s * (params["cs"] + params["cps"])
    return y


def bridge_square(t: np.ndarray, fsw: float, phase_deg: float, duty: float) -> np.ndarray:
    phase_cycles = phase_deg / 360.0
    cycle = np.mod(fsw * t - phase_cycles, 1.0)
    return np.where(cycle < duty, 1.0, -1.0)


def edge_filter(signal: np.ndarray, fs: float, edge_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(signal - np.mean(signal))
    freq = np.fft.rfftfreq(signal.size, 1.0 / fs)
    response = 1.0 / (1.0 + 1j * freq / edge_hz)
    return np.fft.irfft(spectrum * response, n=signal.size)


def operating_windows(t: np.ndarray, sim: dict) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    settings = [
        (5.0, 0.92, 0.50, 0.50),
        (12.0, 1.00, 0.49, 0.50),
        (20.0, 1.08, 0.50, 0.51),
        (30.0, 0.96, 0.51, 0.49),
        (40.0, 1.04, 0.48, 0.52),
        (50.0, 1.10, 0.52, 0.50),
        (-20.0, 0.90, 0.50, 0.48),
    ]
    windows = []
    for phase, ratio, duty1, duty2 in settings:
        v1 = sim["v1_dc"] * bridge_square(t, sim["fsw"], 0.0, duty1)
        v2 = sim["v1_dc"] * sim["n"] * ratio * bridge_square(t, sim["fsw"], phase, duty2)
        v1 = edge_filter(v1, sim["fs"], sim["edge_hz"])
        v2 = edge_filter(v2, sim["fs"], sim["edge_hz"])
        windows.append((v1, v2, {"phase_deg": phase, "voltage_ratio": ratio}))
    return windows


def reconstruct_y(params_true: dict, params_nominal: dict, sim: dict, rng: np.random.Generator) -> dict:
    n_samples = sim["n_samples"]
    fs = sim["fs"]
    freq = np.fft.rfftfreq(n_samples, 1.0 / fs)
    truth = two_port_y(freq, params_true)
    truth[0] = truth[1]
    t = np.arange(n_samples) / fs
    windows = operating_windows(t, sim)

    v_stack = np.empty((freq.size, 2, len(windows)), dtype=complex)
    i_stack = np.empty_like(v_stack)
    representative = None
    for index, (v1, v2, setting) in enumerate(windows):
        vf = np.stack((np.fft.rfft(v1), np.fft.rfft(v2)), axis=1)
        current_f = np.einsum("fij,fj->fi", truth, vf)
        i1 = np.fft.irfft(current_f[:, 0], n=n_samples)
        i2 = np.fft.irfft(current_f[:, 1], n=n_samples)

        v1m = v1 + sim["voltage_noise"] * np.std(v1) * rng.standard_normal(n_samples)
        v2m = v2 + sim["voltage_noise"] * np.std(v2) * rng.standard_normal(n_samples)
        i_scale = max(np.std(i1), np.std(i2), 1e-9)
        i1m = i1 + sim["current_noise"] * i_scale * rng.standard_normal(n_samples)
        i2m = i2 + sim["current_noise"] * i_scale * rng.standard_normal(n_samples)
        v_stack[:, 0, index] = np.fft.rfft(v1m)
        v_stack[:, 1, index] = np.fft.rfft(v2m)
        i_stack[:, 0, index] = np.fft.rfft(i1m)
        i_stack[:, 1, index] = np.fft.rfft(i2m)

        if setting["phase_deg"] == 20.0:
            representative = {"t": t, "v1": v1, "v2": v2, "i1": i1, "i2": i2}

    scales = np.array([sim["v1_dc"], sim["v1_dc"] * sim["n"]])
    y_est = np.zeros_like(truth)
    excitation_condition = np.full(freq.size, np.inf)
    excitation_energy = np.zeros(freq.size)
    for k in range(1, freq.size):
        z = v_stack[k] / scales[:, None]
        gram = z @ z.conj().T
        excitation_condition[k] = np.linalg.cond(gram)
        excitation_energy[k] = np.trace(gram).real
        regularizer = sim["ridge"] * max(np.trace(gram).real / 2.0, 1e-30)
        gain = i_stack[k] @ z.conj().T @ np.linalg.inv(gram + regularizer * np.eye(2))
        y_est[k] = gain / scales[None, :]
    y_est[0] = y_est[1]
    excitation_energy /= max(np.max(excitation_energy), 1e-30)

    valid = (freq >= sim["fit_band_hz"][0]) & (freq <= sim["fit_band_hz"][1])
    valid &= excitation_energy >= sim["energy_threshold"]
    valid &= excitation_condition <= sim["condition_limit"]

    # Apparent input admittance from one loaded operating window. Treating it
    # as Y11 is the intentionally incorrect single-port comparison.
    naive_window = 2
    y_app = i_stack[:, 0, naive_window] / (v_stack[:, 0, naive_window] + 1e-30)
    return {
        "freq": freq,
        "y_true": truth,
        "y_est": y_est,
        "y_app": y_app,
        "valid": valid,
        "energy": excitation_energy,
        "condition": excitation_condition,
        "representative": representative,
        "params_nominal": params_nominal,
    }


def fit_joint_ls_cps(record: dict, params_nominal: dict) -> tuple[float, float]:
    freq = record["freq"][record["valid"]]
    measured = record["y_est"][record["valid"], 0, 1]
    energy = record["energy"][record["valid"]]
    condition = record["condition"][record["valid"]]
    select = np.arange(freq.size)[::2]
    freq = freq[select]
    measured = measured[select]
    weights = np.sqrt(energy[select] / (1.0 + condition[select] / 50.0))

    def residual(log_values: np.ndarray) -> np.ndarray:
        ls, cps = np.exp(log_values)
        s = 1j * 2 * np.pi * freq
        model = -params_nominal["n"] / (params_nominal["rs"] + s * ls) - s * cps
        scale = np.maximum(np.abs(model), 2e-4)
        value = weights * (model - measured) / scale
        return np.concatenate((value.real, value.imag))

    initial = np.log([params_nominal["ls"], params_nominal["cps"]])
    lower = np.log([0.65 * params_nominal["ls"], 5e-12])
    upper = np.log([1.35 * params_nominal["ls"], 90e-12])
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=250)
    ls, cps = np.exp(result.x)
    return float(ls), float(cps)


def fit_fixed_ls_cps(record: dict, params_nominal: dict) -> float:
    freq = record["freq"][record["valid"]][::2]
    measured = record["y_est"][record["valid"], 0, 1][::2]
    energy = record["energy"][record["valid"]][::2]
    s = 1j * 2 * np.pi * freq
    magnetic = -params_nominal["n"] / (params_nominal["rs"] + s * params_nominal["ls"])
    target = measured - magnetic
    weight = energy / np.max(energy)
    basis = -s
    cps = np.real(np.sum(weight * np.conj(basis) * target) / np.sum(weight * np.abs(basis) ** 2))
    return float(np.clip(cps, 5e-12, 90e-12))


def fit_naive_cps(record: dict, params_nominal: dict) -> float:
    freq = record["freq"][record["valid"]][::2]
    measured = record["y_app"][record["valid"]][::2]
    weight = record["energy"][record["valid"]][::2]
    weight /= np.max(weight)

    def objective(cps_pf: float) -> float:
        params = {**params_nominal, "cps": cps_pf * 1e-12}
        model = two_port_y(freq, params)[:, 0, 0]
        scale = np.maximum(np.abs(model), 2e-4)
        return float(np.mean(weight * np.abs((model - measured) / scale) ** 2))

    result = minimize_scalar(objective, bounds=(5.0, 90.0), method="bounded")
    return float(result.x * 1e-12)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict]) -> None:
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        x = np.array([row["cps_true_pf"] for row in subset])
        axes[0].plot(x, [row["cps_joint_pf"] for row in subset], "o-", label=f"multiport: {scenario}")
    x_all = np.array([min(row["cps_true_pf"] for row in rows), max(row["cps_true_pf"] for row in rows)])
    axes[0].plot(x_all, x_all, "k--", linewidth=1.2, label="ideal")
    axes[0].set_xlabel("True Cps (pF)")
    axes[0].set_ylabel("Estimated Cps (pF)")
    axes[0].set_title("Loaded multiport Cps tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    x_pos = np.arange(len(scenarios))
    joint_mae = [np.mean([abs(row["joint_error_pct"]) for row in rows if row["scenario"] == name]) for name in scenarios]
    fixed_mae = [np.mean([abs(row["fixed_error_pct"]) for row in rows if row["scenario"] == name]) for name in scenarios]
    axes[1].bar(x_pos - 0.18, joint_mae, width=0.36, label="joint Ls+Cps")
    axes[1].bar(x_pos + 0.18, fixed_mae, width=0.36, label="fixed Ls")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x_pos, [name.replace("_", "\n") for name in scenarios], fontsize=8)
    axes[1].set_ylabel("Mean absolute Cps error (%)")
    axes[1].set_title("Leakage-inductance drift test")
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    nominal = [row for row in rows if row["scenario"] == "nominal"]
    x = np.array([row["cps_true_pf"] for row in nominal])
    axes[2].plot(x, [abs(row["joint_error_pct"]) for row in nominal], "o-", label="multiport joint Ls+Cps")
    axes[2].plot(x, [abs(row["fixed_error_pct"]) for row in nominal], "s-", label="multiport fixed Ls")
    axes[2].plot(x, [abs(row["naive_error_pct"]) for row in nominal], "^-", label="loaded single-port (incorrect)")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("True Cps (pF)")
    axes[2].set_ylabel("Absolute error (%)")
    axes[2].set_title("Why full port observation is required")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.savefig(DATA_DIR / "loaded_dab_multiport_cps_summary.png", dpi=190)
    plt.close(fig)


def plot_diagnostics(record: dict, params_true: dict, cps_est: float) -> None:
    freq = record["freq"]
    valid = record["valid"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    axes[0].loglog(freq[valid], np.abs(record["y_true"][valid, 0, 1]), "k-", label="true |Y12|")
    axes[0].loglog(freq[valid], np.abs(record["y_est"][valid, 0, 1]), ".", markersize=3, label="reconstructed |Y12|")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Mutual admittance (S)")
    axes[0].set_title("Full two-port reconstruction")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    rep = record["representative"]
    t = rep["t"]
    dvdt = np.gradient(rep["v1"] - rep["v2"], t)
    i_true = params_true["cps"] * dvdt
    i_est = cps_est * dvdt
    edge = int(np.argmax(np.abs(dvdt)))
    half = 80
    sl = slice(max(edge - half, 0), min(edge + half, t.size))
    axes[1].plot((t[sl] - t[edge]) * 1e6, i_true[sl], "k-", label="true")
    axes[1].plot((t[sl] - t[edge]) * 1e6, i_est[sl], "--", label="predicted from estimated Cps")
    axes[1].set_xlabel("Time around switching edge (us)")
    axes[1].set_ylabel("Interwinding displacement current (A)")
    axes[1].set_title("Engineering output: edge current")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(DATA_DIR / "loaded_dab_multiport_y12_and_current.png", dpi=190)
    plt.close(fig)


def write_note(rows: list[dict], config: dict) -> None:
    joint_mae = np.mean([abs(row["joint_error_pct"]) for row in rows])
    fixed_mae = np.mean([abs(row["fixed_error_pct"]) for row in rows])
    naive_mae = np.mean([abs(row["naive_error_pct"]) for row in rows])
    lines = [
        "# Loaded DAB operating-window multiport Cps identification",
        "",
        "## Scope and model boundary",
        "",
        "The bridge terminal voltages are prescribed DAB-like square waves. Seven operating windows vary phase shift, voltage ratio, and duty ratio. The Liu 2017 prototype-1 two-port model maps the two terminal voltages to both terminal currents.",
        "This is not yet a complete switching-level DAB with controller, dead time, semiconductor parasitics, or a measured load.",
        "",
        "## Identification chain",
        "",
        "1. Reconstruct the full two-port admittance matrix from v1, i1, v2, and i2 over multiple operating windows.",
        "2. Jointly estimate leakage inductance and Cps from the reconstructed mutual admittance Y12.",
        "3. Compare against fixed-Ls estimation and the incorrect practice of treating loaded I1/V1 as Y11.",
        "4. Use estimated Cps to predict the interwinding displacement-current waveform.",
        "",
        "## Aggregate results",
        "",
        f"- Multiport joint Ls+Cps mean absolute Cps error: {joint_mae:.3f}%",
        f"- Multiport fixed-Ls mean absolute Cps error: {fixed_mae:.3f}%",
        f"- Loaded single-port mean absolute Cps error: {naive_mae:.3f}%",
        f"- Repeated trials per state: {config['repeats']}",
        f"- Valid reconstructed harmonics in the representative case: {config['valid_frequency_count']}",
        "",
        "## Interpretation",
        "",
        "The full port equations separate transformer mutual admittance from the operating-point-dependent V2/V1 contribution. The single-port apparent admittance cannot make this separation and therefore attributes load/phase-shift effects to Cps.",
        "Jointly estimating leakage inductance prevents leakage drift from being misidentified as capacitance drift. The estimated Cps also closes the engineering chain to a switching-edge displacement-current prediction.",
        "",
        "## Next validation",
        "",
        "Replace prescribed bridge voltages with a switching-level DAB model and then a low-voltage bench. Add known capacitors across the isolation barrier, independently measure the offline reference Cps, and test load, grounding, temperature, dead time, and probe-chain disturbances.",
    ]
    (NOTE_DIR / "loaded_dab_multiport_cps_identification_record.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    nominal = {
        "n": 4.0,
        "lm": 94.5e-3,
        "ls": 96.0e-6,
        "rs": 0.08,
        "rm": 1e9,
        "cp": 5.31e-12,
        "cs": 215.59e-12,
        "cps": 28.36e-12,
    }
    sim = {
        "fs": 50e6,
        "n_samples": 2**16,
        "fsw": 100e3,
        "v1_dc": 100.0,
        "n": nominal["n"],
        "edge_hz": 8e6,
        "voltage_noise": 4e-4,
        "current_noise": 8e-4,
        "ridge": 2e-7,
        "energy_threshold": 2e-7,
        "condition_limit": 2e4,
        "fit_band_hz": (0.08e6, 10.0e6),
    }
    scenarios = {
        "nominal": {"ls_factor": 1.0, "rs_factor": 1.0},
        "hot_Rs_x1.4": {"ls_factor": 1.0, "rs_factor": 1.4},
        "Ls_minus_5pct": {"ls_factor": 0.95, "rs_factor": 1.0},
        "Ls_plus_5pct": {"ls_factor": 1.05, "rs_factor": 1.0},
    }
    added_pf = [0.0, 2.5, 5.0, 10.0, 20.0]
    repeats = 6
    trial_rows = []
    summary_rows = []
    representative = None
    representative_params = None
    representative_est = None

    for scenario, factors in scenarios.items():
        for added in added_pf:
            true = {
                **nominal,
                "ls": nominal["ls"] * factors["ls_factor"],
                "rs": nominal["rs"] * factors["rs_factor"],
                "cps": nominal["cps"] + added * 1e-12,
            }
            estimates = []
            for repeat in range(repeats):
                record = reconstruct_y(true, nominal, sim, rng)
                ls_joint, cps_joint = fit_joint_ls_cps(record, nominal)
                cps_fixed = fit_fixed_ls_cps(record, nominal)
                cps_naive = fit_naive_cps(record, nominal)
                estimates.append((ls_joint, cps_joint, cps_fixed, cps_naive))
                trial_rows.append({
                    "scenario": scenario,
                    "added_cps_pf": added,
                    "repeat": repeat + 1,
                    "cps_true_pf": true["cps"] * 1e12,
                    "ls_true_uh": true["ls"] * 1e6,
                    "cps_joint_pf": cps_joint * 1e12,
                    "ls_joint_uh": ls_joint * 1e6,
                    "cps_fixed_pf": cps_fixed * 1e12,
                    "cps_naive_pf": cps_naive * 1e12,
                    "valid_frequency_count": int(np.sum(record["valid"])),
                })
                if scenario == "nominal" and added == 10.0 and repeat == 0:
                    representative = record
                    representative_params = true
                    representative_est = cps_joint

            values = np.asarray(estimates)
            means = np.mean(values, axis=0)
            stds = np.std(values, axis=0, ddof=1)
            cps_true_pf = true["cps"] * 1e12
            row = {
                "scenario": scenario,
                "added_cps_pf": added,
                "cps_true_pf": cps_true_pf,
                "ls_true_uh": true["ls"] * 1e6,
                "cps_joint_pf": means[1] * 1e12,
                "cps_joint_std_pf": stds[1] * 1e12,
                "ls_joint_uh": means[0] * 1e6,
                "ls_joint_std_uh": stds[0] * 1e6,
                "cps_fixed_pf": means[2] * 1e12,
                "cps_naive_pf": means[3] * 1e12,
                "joint_error_pct": 100 * (means[1] * 1e12 - cps_true_pf) / cps_true_pf,
                "fixed_error_pct": 100 * (means[2] * 1e12 - cps_true_pf) / cps_true_pf,
                "naive_error_pct": 100 * (means[3] * 1e12 - cps_true_pf) / cps_true_pf,
                "current_prediction_error_pct": 100 * (means[1] * 1e12 - cps_true_pf) / cps_true_pf,
            }
            summary_rows.append(row)

    write_csv(DATA_DIR / "loaded_dab_multiport_cps_trials.csv", trial_rows)
    write_csv(DATA_DIR / "loaded_dab_multiport_cps_summary.csv", summary_rows)
    plot_summary(summary_rows)
    plot_diagnostics(representative, representative_params, representative_est)
    config = {
        **sim,
        "scenarios": scenarios,
        "added_cps_pf": added_pf,
        "repeats": repeats,
        "operating_window_count": 7,
        "valid_frequency_count": int(np.sum(representative["valid"])),
    }
    (DATA_DIR / "loaded_dab_multiport_cps_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_note(summary_rows, config)

    print("Loaded DAB multiport Cps experiment complete")
    for scenario in scenarios:
        subset = [row for row in summary_rows if row["scenario"] == scenario]
        joint = np.mean([abs(row["joint_error_pct"]) for row in subset])
        fixed = np.mean([abs(row["fixed_error_pct"]) for row in subset])
        naive = np.mean([abs(row["naive_error_pct"]) for row in subset])
        print(f"{scenario:>16s}: joint={joint:7.3f}%, fixed={fixed:7.3f}%, naive={naive:7.3f}%")
    print(f"valid reconstructed frequencies: {config['valid_frequency_count']}")


if __name__ == "__main__":
    main()
