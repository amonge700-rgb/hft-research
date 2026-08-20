"""Model-level engineering closure for online Cps identification.

Literature anchor:
    Liu et al., IEEE TPEL 2017, prototype 1 three-capacitance model.

The experiment applies known interwinding-capacitance perturbations, recovers
the voltage-transfer zero from noisy time-domain binary excitation, estimates
Cps, and uses the estimate to predict common-mode displacement current.

This is a simulation baseline. It is not a loaded DAB model and does not prove
insulation aging diagnosis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


SEED = 20260713
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
NOTE_DIR = ROOT / "笔记"


def paper_observables(freq_hz: np.ndarray, cps_f: float, paper: dict) -> tuple[np.ndarray, np.ndarray]:
    freq = np.maximum(np.asarray(freq_hz, dtype=float), 1e-3)
    s = 1j * 2 * np.pi * freq
    yl = 1.0 / (paper["rs"] + s * paper["ls"])
    ym = 1.0 / paper["rm"] + 1.0 / (s * paper["lm"])
    y11 = paper["n"] ** 2 * yl + ym + s * (paper["cp"] + cps_f)
    y12 = -paper["n"] * yl - s * cps_f
    y22 = yl + s * (paper["cs"] + cps_f)
    yin_oc = y11 - y12**2 / y22
    hv = -y12 / y22
    return yin_oc, hv


def make_binary_excitation(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Randomized binary switching gives continuous coverage over the possible
    # zero-frequency drift band. It is a diagnostic-friendly baseline rather
    # than a claim that ordinary converter PWM has this exact spectrum.
    chip_samples = 4
    levels = rng.choice((-1.0, 1.0), size=int(np.ceil(t.size / chip_samples)))
    voltage = np.repeat(levels, chip_samples)[: t.size]
    voltage -= np.mean(voltage)
    return voltage / np.max(np.abs(voltage))


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def fit_cps_from_hv(
    freq: np.ndarray, hv_est: np.ndarray, energy: np.ndarray, paper: dict, sim: dict
) -> float:
    fit_band = (freq >= sim["fit_band"][0]) & (freq <= sim["fit_band"][1])
    fit_band &= energy >= sim["energy_threshold"]
    # Downsample for optimization speed while retaining the whole drift band.
    selected = np.flatnonzero(fit_band)[::8]
    freq_fit = freq[selected]
    measured = hv_est[selected]
    weight = np.sqrt(energy[selected] / np.max(energy[selected]))
    def objective(cps_pf: float) -> float:
        _, model = paper_observables(freq_fit, cps_pf * 1e-12, paper)
        # Complex residual uses phase as well as magnitude. The denominator
        # limits the influence of the high-frequency output-noise floor while
        # preserving the pole shoulder and zero-motion information.
        scale = np.maximum(np.abs(model), 0.08)
        residual = (model - measured) / scale
        squared = np.minimum(np.abs(residual) ** 2, 16.0)
        return float(np.mean(weight**2 * squared))

    result = minimize_scalar(objective, bounds=(10.0, 80.0), method="bounded")
    return float(result.x * 1e-12)


def reconstruct_hv(cps_f: float, paper: dict, sim: dict, rng: np.random.Generator) -> dict:
    n = sim["n"]
    fs = sim["fs"]
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    _, hv_true = paper_observables(freq, cps_f, paper)
    hv_true[0] = hv_true[1]

    svv = np.zeros(freq.size)
    sv2v = np.zeros(freq.size, dtype=complex)
    t = np.arange(n) / fs

    for _ in range(sim["averages"]):
        voltage = make_binary_excitation(t, rng)
        voltage_fft = np.fft.rfft(voltage)
        output = np.fft.irfft(hv_true * voltage_fft, n=n)

        voltage_measured = voltage + sim["input_noise"] * np.std(voltage) * rng.standard_normal(n)
        output_measured = output + sim["output_noise"] * np.std(output) * rng.standard_normal(n)
        v_fft = np.fft.rfft(voltage_measured)
        v2_fft = np.fft.rfft(output_measured)
        svv += np.abs(v_fft) ** 2
        sv2v += v2_fft * np.conj(v_fft)

    regularizer = 1e-12 * np.max(svv)
    hv_est = sv2v / (svv + regularizer)
    energy = svv / np.max(svv)
    valid = (freq >= sim["feature_band"][0]) & (freq <= sim["feature_band"][1])
    valid &= energy >= sim["energy_threshold"]

    feature_freq = freq[valid]
    feature_mag = moving_average(np.log10(np.abs(hv_est[valid]) + 1e-15), 11)
    edge = min(10, feature_mag.size // 10)
    if edge:
        feature_freq = feature_freq[edge:-edge]
        feature_mag = feature_mag[edge:-edge]
    fzero_raw = feature_freq[np.argmin(feature_mag)]
    fzero_true = 1.0 / (2 * np.pi * np.sqrt(paper["ls"] * cps_f / paper["n"]))
    cps_fit = fit_cps_from_hv(freq, hv_est, energy, paper, sim)
    fzero_est = 1.0 / (2 * np.pi * np.sqrt(paper["ls"] * cps_fit / paper["n"]))

    return {
        "freq": freq,
        "hv_true": hv_true,
        "hv_est": hv_est,
        "energy": energy,
        "fzero_true": fzero_true,
        "fzero_est": fzero_est,
        "fzero_raw": fzero_raw,
        "cps_fit": cps_fit,
    }


def estimate_cps(fzero_hz: float, paper: dict) -> float:
    return paper["n"] / (paper["ls"] * (2 * np.pi * fzero_hz) ** 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_transfer_curves(records: list[dict], baseline_cps_pf: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(records)))
    for color, record in zip(colors, records):
        freq = record["freq"]
        band = (freq >= 0.4e6) & (freq <= 10e6)
        label = f"Cps={record['cps_true_pf']:.2f} pF"
        ax.semilogx(freq[band], 20 * np.log10(np.abs(record["hv_est"][band]) + 1e-15), color=color, label=label)
        ax.axvline(record["fzero_est"], color=color, linestyle=":", alpha=0.55)
    ax.set_title(f"Online transfer-zero tracking around baseline Cps={baseline_cps_pf:.2f} pF")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Estimated |Hv| (dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(DATA_DIR / "cps_common_mode_transfer_zero_tracking.png", dpi=180)
    plt.close(fig)


def plot_engineering_results(rows: list[dict]) -> None:
    added = np.array([row["added_cps_pf"] for row in rows])
    cps_true = np.array([row["cps_true_pf"] for row in rows])
    cps_est = np.array([row["cps_est_pf"] for row in rows])
    cps_std = np.array([row["cps_est_std_pf"] for row in rows])
    current_true = np.array([row["icm_peak_true_a_at_16kv_us"] for row in rows])
    current_est = np.array([row["icm_peak_est_a_at_16kv_us"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    axes[0].plot(added, cps_true, "o-k", label="true")
    axes[0].errorbar(added, cps_est, yerr=1.96 * cps_std, fmt="s--", capsize=4, label="online estimate (95% window)")
    axes[0].set_xlabel("Added interwinding capacitance (pF)")
    axes[0].set_ylabel("Effective Cps (pF)")
    axes[0].set_title("Known Cps perturbation tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(added, current_true, "o-k", label="true")
    current_std = cps_std * 1e-12 * 16e9
    axes[1].errorbar(added, current_est, yerr=1.96 * current_std, fmt="s--", capsize=4, label="predicted (95% window)")
    axes[1].set_xlabel("Added interwinding capacitance (pF)")
    axes[1].set_ylabel("Peak displacement current at 16 kV/us (A)")
    axes[1].set_title("Common-mode current prediction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(DATA_DIR / "cps_common_mode_engineering_closure.png", dpi=180)
    plt.close(fig)


def write_note(rows: list[dict], config: dict) -> None:
    cps_mae = float(np.mean([abs(row["cps_error_pct"]) for row in rows]))
    current_mae = float(np.mean([abs(row["icm_error_pct"]) for row in rows]))
    separated_steps = []
    for left, right in zip(rows[:-1], rows[1:]):
        left_upper = left["cps_est_pf"] + 1.96 * left["cps_est_std_pf"]
        right_lower = right["cps_est_pf"] - 1.96 * right["cps_est_std_pf"]
        if left_upper < right_lower:
            separated_steps.append(right["cps_true_pf"] - left["cps_true_pf"])
    min_step = min(separated_steps) if separated_steps else float("nan")
    path = NOTE_DIR / "cps_common_mode_engineering_closure_record.md"
    lines = [
        "# Cps online identification and common-mode-current engineering closure",
        "",
        "## Scope",
        "",
        "This model-level experiment uses the Liu 2017 prototype-1 three-capacitance model.",
        "Known capacitance is added across the winding ports, a fixed noisy binary excitation is used to recover the transfer zero, and the estimated Cps predicts displacement current.",
        "It is not a loaded DAB experiment and does not establish a unique relationship between Cps and insulation aging.",
        "",
        "## Settings",
        "",
        f"- Baseline Cps: {config['baseline_cps_pf']:.2f} pF",
        f"- Added capacitance: {config['added_cps_pf']} pF",
        f"- Sampling rate: {config['fs']/1e6:.1f} MHz",
        f"- FFT length: {config['n']}",
        f"- Spectral averages: {config['averages']}",
        f"- Repeated windows per state: {config['repeats']}",
        f"- Reference dv/dt: {config['dvdt_kv_us']:.1f} kV/us",
        "",
        "## Results",
        "",
        "| Added Cps (pF) | True Cps (pF) | Estimated Cps (pF) | Std (pF) | Cps error (%) | True peak current (A) | Predicted peak current (A) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['added_cps_pf']:.2f} | {row['cps_true_pf']:.3f} | {row['cps_est_pf']:.3f} | "
            f"{row['cps_est_std_pf']:.3f} | {row['cps_error_pct']:.3f} | {row['icm_peak_true_a_at_16kv_us']:.4f} | "
            f"{row['icm_peak_est_a_at_16kv_us']:.4f} |"
        )
    lines += [
        "",
        f"- Mean absolute Cps error: {cps_mae:.3f}%",
        f"- Mean absolute common-mode-current prediction error: {current_mae:.3f}%",
        f"- Smallest adjacent Cps increment with non-overlapping 95% single-window intervals: {min_step:.2f} pF",
        "",
        "## Interpretation",
        "",
        "The result closes the chain from an online frequency-response feature to a directly measurable engineering quantity.",
        "The current result is intentionally narrow: it demonstrates sensitivity to known Cps perturbations under an open-secondary literature model.",
        "The next experiment must replace the open-secondary transfer function with loaded multiport observations and test load, temperature, grounding, and measurement-chain nuisance variables.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    paper = {
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
        "n": 2**18,
        "averages": 16,
        "input_noise": 1e-3,
        "output_noise": 2e-3,
        "energy_threshold": 1e-7,
        "feature_band": (3.0e6, 10.0e6),
        "fit_band": (0.55e6, 9.5e6),
    }
    added_cps_pf = [0.0, 2.5, 5.0, 10.0, 20.0]
    dvdt_values_kv_us = [8.0, 16.0, 24.0]
    repeats = 8

    rows: list[dict] = []
    trial_rows: list[dict] = []
    records: list[dict] = []
    for added_pf in added_cps_pf:
        cps_true = paper["cps"] + added_pf * 1e-12
        state_trials = []
        representative = None
        for repeat in range(repeats):
            record = reconstruct_hv(cps_true, paper, sim, rng)
            cps_est_trial = record["cps_fit"]
            state_trials.append(cps_est_trial)
            trial_rows.append(
                {
                    "added_cps_pf": added_pf,
                    "repeat": repeat + 1,
                    "cps_true_pf": cps_true * 1e12,
                    "cps_est_pf": cps_est_trial * 1e12,
                    "cps_error_pct": 100 * (cps_est_trial - cps_true) / cps_true,
                    "fzero_true_hz": record["fzero_true"],
                    "fzero_est_hz": record["fzero_est"],
                    "fzero_raw_hz": record["fzero_raw"],
                }
            )
            if representative is None:
                representative = record

        cps_est = float(np.mean(state_trials))
        cps_est_std = float(np.std(state_trials, ddof=1))
        fzero_true = 1.0 / (2 * np.pi * np.sqrt(paper["ls"] * cps_true / paper["n"]))
        fzero_est = 1.0 / (2 * np.pi * np.sqrt(paper["ls"] * cps_est / paper["n"]))
        row = {
            "added_cps_pf": added_pf,
            "cps_true_pf": cps_true * 1e12,
            "cps_est_pf": cps_est * 1e12,
            "cps_est_std_pf": cps_est_std * 1e12,
            "cps_error_pct": 100 * (cps_est - cps_true) / cps_true,
            "fzero_true_hz": fzero_true,
            "fzero_est_hz": fzero_est,
            "fzero_error_pct": 100 * (fzero_est - fzero_true) / fzero_true,
        }
        for dvdt in dvdt_values_kv_us:
            scale = dvdt * 1e9  # 1 kV/us = 1e9 V/s
            key = f"at_{int(dvdt)}kv_us"
            row[f"icm_peak_true_a_{key}"] = cps_true * scale
            row[f"icm_peak_est_a_{key}"] = cps_est * scale
        row["icm_error_pct"] = row["cps_error_pct"]
        rows.append(row)
        records.append({**representative, "cps_true_pf": cps_true * 1e12})

    write_csv(DATA_DIR / "cps_common_mode_engineering_summary.csv", rows)
    write_csv(DATA_DIR / "cps_common_mode_engineering_trials.csv", trial_rows)
    plot_transfer_curves(records, paper["cps"] * 1e12)
    plot_engineering_results(rows)
    config = {
        "baseline_cps_pf": paper["cps"] * 1e12,
        "added_cps_pf": added_cps_pf,
        "fs": sim["fs"],
        "n": sim["n"],
        "averages": sim["averages"],
        "repeats": repeats,
        "dvdt_kv_us": 16.0,
    }
    write_note(rows, config)
    (DATA_DIR / "cps_common_mode_engineering_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Cps engineering closure complete")
    for row in rows:
        print(
            f"added={row['added_cps_pf']:>4.1f} pF, "
            f"Cps={row['cps_est_pf']:.3f} pF, "
            f"error={row['cps_error_pct']:+.3f}%, "
            f"Icm@16kV/us={row['icm_peak_est_a_at_16kv_us']:.4f} A"
        )


if __name__ == "__main__":
    main()
