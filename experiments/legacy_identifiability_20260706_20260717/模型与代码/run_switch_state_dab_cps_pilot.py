"""Experiment 8: switch-state DAB pilot for loaded multiport Cps estimation.

The forward model uses stiff DC buses and explicit bipolar bridge switch states
with dead-time zero intervals and finite Coss-equivalent edge dynamics.  The
transformer contains the Liu-style main RL coupling path, magnetizing branch,
terminal/interwinding/ground capacitances, and an optional weak cross-port RLC
mode.  Load power determines the nominal DAB phase-shift operating point.

The inverse remains the compact Y12 model used in Experiments 6-7.  This pilot
is intentionally between a frequency-domain model and a device-level SPICE
model: it is waveform-level and switching-state based, but it does not model
individual MOSFET nonlinear capacitances or a closed-loop output capacitor.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from run_loaded_dab_forward_model_mismatch import (
    apply_measurement_chain,
    fit_simplified_y12,
)


SEED = 20260715
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT.parents[0] / "experiment8_switch_state_dab_20260715"
DATA_DIR = OUTPUT_ROOT / "data"
NOTE_DIR = OUTPUT_ROOT / "notes"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def phase_for_power(power_w: float, v1: float, v2: float, n: float, fsw: float, ls: float) -> float:
    """Single-phase-shift DAB approximation, returns the small root in rad."""
    gain = n * v1 * v2 / (2.0 * np.pi * fsw * ls)
    x = power_w / max(gain, 1e-12)
    discriminant = max(np.pi**2 - 4.0 * np.pi * x, 0.0)
    return 0.5 * (np.pi - np.sqrt(discriminant))


def deadtime_state(
    t: np.ndarray, fsw: float, phase_rad: float, duty: float, deadtime_s: float
) -> np.ndarray:
    cycle = np.mod(fsw * t - phase_rad / (2.0 * np.pi), 1.0)
    raw = np.where(cycle < duty, 1.0, -1.0)
    state = raw.copy()
    dead_steps = max(1, int(round(deadtime_s / (t[1] - t[0]))))
    transitions = np.flatnonzero(raw[1:] != raw[:-1]) + 1
    for index in transitions:
        state[index : min(index + dead_steps, state.size)] = 0.0
    return state


def finite_edge_voltage(target: np.ndarray, fs: float, tau_s: float, ringing_gain: float) -> np.ndarray:
    freq = np.fft.rfftfreq(target.size, 1.0 / fs)
    s = 1j * 2.0 * np.pi * freq
    edge = 1.0 / (1.0 + s * tau_s)
    w0 = 2.0 * np.pi * 4.2e6
    q = 3.0
    bandpass = (s * w0 / q) / (s**2 + s * w0 / q + w0**2)
    return np.fft.irfft(np.fft.rfft(target) * edge * (1.0 + ringing_gain * bandpass), n=target.size)


def rl_current(u: np.ndarray, fs: float, r: float, l: float) -> np.ndarray:
    freq = np.fft.rfftfreq(u.size, 1.0 / fs)
    transfer = 1.0 / (r + 1j * 2.0 * np.pi * freq * l)
    return np.fft.irfft(np.fft.rfft(u) * transfer, n=u.size)


def rlc_series_current(u: np.ndarray, fs: float, r: float, l: float, c: float) -> np.ndarray:
    freq = np.fft.rfftfreq(u.size, 1.0 / fs)
    s = 1j * 2.0 * np.pi * freq
    transfer = np.zeros_like(s, dtype=complex)
    transfer[1:] = 1.0 / (r + s[1:] * l + 1.0 / (s[1:] * c))
    return np.fft.irfft(np.fft.rfft(u) * transfer, n=u.size)


def periodic_derivative(x: np.ndarray, fs: float) -> np.ndarray:
    freq = np.fft.rfftfreq(x.size, 1.0 / fs)
    return np.fft.irfft(np.fft.rfft(x) * (1j * 2.0 * np.pi * freq), n=x.size)


def magnetizing_current(v: np.ndarray, fs: float, lm: float) -> np.ndarray:
    freq = np.fft.rfftfreq(v.size, 1.0 / fs)
    s = 1j * 2.0 * np.pi * freq
    transfer = np.zeros_like(s, dtype=complex)
    transfer[1:] = 1.0 / (s[1:] * lm)
    return np.fft.irfft(np.fft.rfft(v) * transfer, n=v.size)


def make_windows(base_phase: float, voltage_ratio: float) -> list[dict]:
    perturb = [
        (-0.08, -0.04, 0.00, 0.00),
        (-0.04, 0.03, -0.01, 0.00),
        (0.00, 0.00, 0.00, 0.01),
        (0.04, -0.02, 0.01, -0.01),
        (0.08, 0.04, -0.02, 0.02),
        (0.12, -0.03, 0.02, 0.00),
        (-0.12, 0.02, 0.00, -0.02),
    ]
    return [
        {
            "phase": base_phase + dphi,
            "ratio": voltage_ratio * (1.0 + dratio),
            "d1": 0.5 + dd1,
            "d2": 0.5 + dd2,
        }
        for dphi, dratio, dd1, dd2 in perturb
    ]


def generate_window(params: dict, sim: dict, setting: dict) -> dict:
    total = sim["record_samples"]
    t = np.arange(total) / sim["fs_hz"]
    s1 = deadtime_state(t, sim["switching_hz"], 0.0, setting["d1"], sim["deadtime_s"])
    s2 = deadtime_state(t, sim["switching_hz"], setting["phase"], setting["d2"], sim["deadtime_s"])
    target1 = sim["vdc1"] * s1
    target2 = params["n"] * sim["vdc1"] * setting["ratio"] * s2
    v1 = finite_edge_voltage(target1, sim["fs_hz"], sim["edge_tau_primary_s"], sim["ringing_gain"])
    v2 = finite_edge_voltage(target2, sim["fs_hz"], sim["edge_tau_secondary_s"], sim["ringing_gain"] * 1.15)

    coupling_voltage = params["n"] * v1 - v2
    q_main = rl_current(coupling_voltage, sim["fs_hz"], params["rs"], params["ls"])
    if params["mode_enabled"]:
        q_mode = rlc_series_current(
            coupling_voltage,
            sim["fs_hz"],
            params["mode_r"],
            params["mode_l"],
            params["mode_c"],
        )
    else:
        q_mode = np.zeros_like(q_main)
    q = q_main + q_mode

    dv1 = periodic_derivative(v1, sim["fs_hz"])
    dv2 = periodic_derivative(v2, sim["fs_hz"])
    dvps = periodic_derivative(v1 - v2, sim["fs_hz"])
    im_l = magnetizing_current(v1, sim["fs_hz"], params["lm"])
    i1 = params["n"] * q + im_l + v1 / params["rm"]
    i1 += (params["cp"] + params["cpg"]) * dv1 + params["cps"] * dvps
    i2 = -q + (params["cs"] + params["csg"]) * dv2 - params["cps"] * dvps

    sl = slice(0, total)
    return {
        "v1": v1[sl],
        "v2": v2[sl],
        "i1": i1[sl],
        "i2": i2[sl],
        "ips": (params["cps"] * dvps)[sl],
        "iground_source": (params["cpg"] * dv1 + params["csg"] * dv2)[sl],
    }


def reconstruct_from_windows(
    windows: list[dict], params: dict, sim: dict, rng: np.random.Generator,
    measurement_chain: bool,
) -> dict:
    n = sim["record_samples"]
    freq = np.fft.rfftfreq(n, 1.0 / sim["fs_hz"])
    v_stack = np.empty((freq.size, 2, len(windows)), dtype=complex)
    i_stack = np.empty_like(v_stack)
    for index, data in enumerate(windows):
        vscale = max(np.std(data["v1"]), 1e-9)
        iscale = max(np.std(data["i1"]), np.std(data["i2"]), 1e-9)
        spectra = {}
        for name in ("v1", "v2"):
            noisy = data[name] + rng.normal(0.0, sim["voltage_noise_rel"] * vscale, n)
            spectra[name] = apply_measurement_chain(np.fft.rfft(noisy), freq, name, measurement_chain)
        for name in ("i1", "i2"):
            noisy = data[name] + rng.normal(0.0, sim["current_noise_rel"] * iscale, n)
            spectra[name] = apply_measurement_chain(np.fft.rfft(noisy), freq, name, measurement_chain)
        v_stack[:, :, index] = np.stack([spectra["v1"], spectra["v2"]], axis=1)
        i_stack[:, :, index] = np.stack([spectra["i1"], spectra["i2"]], axis=1)

    scales = np.asarray([sim["vdc1"], sim["vdc1"] * params["n"]])
    yhat = np.zeros((freq.size, 2, 2), dtype=complex)
    energy = np.zeros(freq.size)
    condition = np.full(freq.size, np.inf)
    for k in range(1, freq.size):
        vk = v_stack[k] / scales[:, None]
        gram = vk @ vk.conj().T
        energy[k] = np.trace(gram).real
        condition[k] = np.linalg.cond(gram)
        ridge = sim["ridge_rel"] * max(energy[k] / 2.0, 1e-30)
        gain = i_stack[k] @ vk.conj().T @ np.linalg.inv(gram + ridge * np.eye(2))
        yhat[k] = gain / scales[None, :]
    energy /= max(np.max(energy), 1e-30)
    valid = (freq >= sim["fit_low_hz"]) & (freq <= sim["fit_high_hz"])
    valid &= energy >= sim["energy_rel_gate"]
    valid &= condition <= sim["max_condition"]
    return {"freq": freq, "yhat": yhat, "energy": energy, "condition": condition, "valid": valid}


def summarize(trials: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in trials:
        key = (
            row["forward_scenario"], row["load_w"], row["temperature_c"],
            row["grounding"], row["added_cps_pf"]
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for key, rows in grouped.items():
        true = float(rows[0]["true_cps_pf"])
        estimates = np.asarray([float(r["estimated_cps_pf"]) for r in rows])
        errors = 100.0 * (estimates - true) / true
        total_errors = np.asarray([float(r["cps_only_total_source_error_pct"]) for r in rows])
        out.append(
            {
                "forward_scenario": key[0],
                "load_w": key[1],
                "temperature_c": key[2],
                "grounding": key[3],
                "added_cps_pf": key[4],
                "true_cps_pf": true,
                "mean_estimated_cps_pf": float(np.mean(estimates)),
                "std_estimated_cps_pf": float(np.std(estimates, ddof=1)),
                "mean_cps_error_pct": float(np.mean(errors)),
                "cps_mae_pct": float(np.mean(np.abs(errors))),
                "mean_cps_only_total_source_error_pct": float(np.mean(total_errors)),
                "mean_valid_frequency_count": float(np.mean([r["valid_frequency_count"] for r in rows])),
            }
        )
    return sorted(out, key=lambda r: (
        r["forward_scenario"], r["load_w"], r["temperature_c"], r["grounding"], r["added_cps_pf"]
    ))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    nominal = {
        "n": 4.0,
        "lm": 47.0e-3,
        "rm": 25.0e3,
        "ls": 96.0e-6,
        "rs": 0.095,
        "cp": 5.31e-12,
        "cs": 215.59e-12,
        "cps": 28.36e-12,
        "cpg": 10.0e-12,
        "csg": 25.0e-12,
        "mode_enabled": False,
        "mode_r": 30000.0,
        "mode_l": 5.0e-6,
        "mode_c": 120.0e-12,
    }
    sim = {
        "fs_hz": 50.0e6,
        "record_samples": 65500,
        "switching_hz": 100.0e3,
        "vdc1": 100.0,
        "deadtime_s": 100.0e-9,
        "edge_tau_primary_s": 32.0e-9,
        "edge_tau_secondary_s": 45.0e-9,
        "ringing_gain": 0.018,
        "voltage_noise_rel": 4.0e-4,
        "current_noise_rel": 8.0e-4,
        "fit_low_hz": 80.0e3,
        "fit_high_hz": 10.0e6,
        "ridge_rel": 2.0e-7,
        "energy_rel_gate": 2.0e-7,
        "max_condition": 2.0e4,
        "repeats": 3,
    }
    load_levels = [250.0, 750.0]
    temperatures = [25.0, 80.0]
    grounding = {
        "baseline": (10.0e-12, 25.0e-12),
        "asymmetric": (35.0e-12, 90.0e-12),
    }
    added_cps_pf = [0.0, 5.0, 10.0]
    forward_scenarios = {
        "matched_switching": {"mode_enabled": False, "measurement_chain": False},
        "weak_cross_mode": {"mode_enabled": True, "measurement_chain": False},
        "full_chain": {"mode_enabled": True, "measurement_chain": True},
    }
    trials: list[dict] = []
    representative = None

    for forward_name, forward_setting in forward_scenarios.items():
        for load_w in load_levels:
            for temperature in temperatures:
                for ground_name, (cpg, csg) in grounding.items():
                    for added_pf in added_cps_pf:
                        params = dict(nominal)
                        params["rs"] *= 1.0 + 0.00393 * (temperature - 25.0)
                        params["cpg"], params["csg"] = cpg, csg
                        params["cps"] += added_pf * 1e-12
                        params["mode_enabled"] = forward_setting["mode_enabled"]
                        v2_nominal = params["n"] * sim["vdc1"]
                        phase = phase_for_power(
                            load_w, sim["vdc1"], v2_nominal, params["n"], sim["switching_hz"], params["ls"]
                        )
                        settings = make_windows(phase, 1.0)
                        waveforms = [generate_window(params, sim, setting) for setting in settings]
                        for repeat in range(sim["repeats"]):
                            record = reconstruct_from_windows(
                                waveforms, params, sim, rng, forward_setting["measurement_chain"]
                            )
                            fit = fit_simplified_y12(
                                record["freq"],
                                record["yhat"][:, 0, 1],
                                record["valid"],
                                nominal,
                                fit_rs=True,
                                energy=record["energy"],
                                condition=record["condition"],
                            )
                            ref = waveforms[2]
                            true_ips_rms = float(np.sqrt(np.mean(ref["ips"] ** 2)))
                            predicted_ips_rms = true_ips_rms * fit["cps"] / params["cps"]
                            total_source = ref["ips"] + ref["iground_source"]
                            total_rms = float(np.sqrt(np.mean(total_source**2)))
                            cps_only_total_error = 100.0 * abs(predicted_ips_rms - total_rms) / max(total_rms, 1e-12)
                            trials.append(
                                {
                                    "forward_scenario": forward_name,
                                    "load_w": load_w,
                                    "phase_deg": float(np.rad2deg(phase)),
                                    "temperature_c": temperature,
                                    "grounding": ground_name,
                                    "added_cps_pf": added_pf,
                                    "repeat": repeat,
                                    "true_rs_ohm": params["rs"],
                                    "true_cps_pf": params["cps"] * 1e12,
                                    "estimated_rs_ohm": fit["rs"],
                                    "estimated_ls_uh": fit["ls"] * 1e6,
                                    "estimated_cps_pf": fit["cps"] * 1e12,
                                    "cps_error_pct": 100.0 * (fit["cps"] - params["cps"]) / params["cps"],
                                    "true_ips_rms_a": true_ips_rms,
                                    "predicted_ips_rms_a": predicted_ips_rms,
                                    "total_capacitive_source_rms_a": total_rms,
                                    "cps_only_total_source_error_pct": cps_only_total_error,
                                    "valid_frequency_count": int(np.sum(record["valid"])),
                                    "median_condition": float(np.median(record["condition"][record["valid"]])),
                                    "physics_fit_cost": fit["cost_rms"],
                                }
                            )
                            if forward_name == "full_chain" and load_w == 750.0 and temperature == 80.0 and ground_name == "asymmetric" and added_pf == 10.0 and repeat == 0:
                                representative = {"record": record, "fit": fit, "wave": ref, "params": params}

    summary = summarize(trials)
    write_csv(DATA_DIR / "switch_state_dab_cps_trials.csv", trials)
    write_csv(DATA_DIR / "switch_state_dab_cps_summary.csv", summary)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    for ground_name in grounding:
        selected = [
            r for r in summary
            if r["forward_scenario"] == "full_chain" and r["temperature_c"] == 80.0 and r["grounding"] == ground_name
        ]
        for load_w in load_levels:
            rows = [r for r in selected if r["load_w"] == load_w]
            axes[0].plot(
                [r["true_cps_pf"] for r in rows],
                [r["mean_estimated_cps_pf"] for r in rows],
                marker="o",
                label=f"{ground_name}, {load_w:.0f} W",
            )
    limits = [27.0, 40.5]
    axes[0].plot(limits, limits, "k--", label="ideal")
    axes[0].set(xlabel="True Cps (pF)", ylabel="Estimated Cps (pF)", title="80 C cross-condition tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    labels = list(forward_scenarios)
    values = []
    for forward_name in forward_scenarios:
        rows = [r for r in summary if r["forward_scenario"] == forward_name]
        values.append(float(np.mean([r["cps_mae_pct"] for r in rows])))
    axes[1].bar(labels, values)
    axes[1].set_ylabel("Cps MAE (%)")
    axes[1].set_title("Load/temperature/grounding robustness")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "switch_state_dab_cps_summary.png", dpi=220)
    plt.close(fig)

    if representative is not None:
        record = representative["record"]
        valid = record["valid"]
        fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0))
        axes[0].loglog(record["freq"][valid], np.abs(record["yhat"][valid, 0, 1]))
        axes[0].set(ylabel="Reconstructed |Y12| (S)", title="Representative switch-state reconstruction")
        axes[0].grid(True, which="both", alpha=0.3)
        time_us = np.arange(1800) / sim["fs_hz"] * 1e6
        wave = representative["wave"]
        axes[1].plot(time_us, wave["v1"][:1800] / sim["vdc1"], label="v1 / Vdc1")
        axes[1].plot(time_us, wave["v2"][:1800] / (sim["vdc1"] * nominal["n"]), label="v2 / (n Vdc1)")
        axes[1].set(xlabel="Time (us)", ylabel="Normalized bridge voltage")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(DATA_DIR / "switch_state_dab_representative_waveform.png", dpi=220)
        plt.close(fig)

    metrics = {
        "overall_cps_mae_pct": float(np.mean([r["cps_mae_pct"] for r in summary])),
        "matched_switching_cps_mae_pct": float(np.mean([r["cps_mae_pct"] for r in summary if r["forward_scenario"] == "matched_switching"])),
        "weak_cross_mode_cps_mae_pct": float(np.mean([r["cps_mae_pct"] for r in summary if r["forward_scenario"] == "weak_cross_mode"])),
        "full_chain_cps_mae_pct": float(np.mean([r["cps_mae_pct"] for r in summary if r["forward_scenario"] == "full_chain"])),
        "mean_cps_only_total_source_error_pct": float(np.mean([r["mean_cps_only_total_source_error_pct"] for r in summary])),
    }
    config = {
        "seed": SEED,
        "nominal": nominal,
        "simulation": sim,
        "load_levels_w": load_levels,
        "temperatures_c": temperatures,
        "grounding_capacitances_f": grounding,
        "added_cps_pf": added_cps_pf,
        "forward_scenarios": forward_scenarios,
        "metrics": metrics,
        "boundaries": [
            "Stiff-DC-bus switch-state DAB pilot; load is represented by its phase-shift operating point.",
            "Dead time and Coss are represented by zero states and finite edge dynamics, not nonlinear device models.",
            "Interwinding displacement current is not identical to total system common-mode current.",
            "Results are simulation evidence, not hardware accuracy.",
        ],
    }
    with (DATA_DIR / "switch_state_dab_cps_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    note = f"""# 实验八：开关状态 DAB 多工况 Cps 辨识先导实验

## 结果

- 全部多工况 Cps MAE：{metrics['overall_cps_mae_pct']:.3f}%
- 匹配开关状态基线 Cps MAE：{metrics['matched_switching_cps_mae_pct']:.3f}%
- 弱附加跨端口模态 Cps MAE：{metrics['weak_cross_mode_cps_mae_pct']:.3f}%
- 完整测量链路场景 Cps MAE：{metrics['full_chain_cps_mae_pct']:.3f}%
- 仅用 Cps 预测总电容源电流的平均误差：{metrics['mean_cps_only_total_source_error_pct']:.3f}%

## 边界

本实验是刚性直流母线下的开关状态先导模型。负载通过相移运行点进入；死区和 Coss 用零状态与有限边沿表示。它不是逐器件 SPICE，也没有闭环输出电容动态。输出中的 `ips` 是跨绕组位移电流；`iground_source` 是对地电容源电流，两者之和仍不等同于完整系统共模回流电流。

## 保存文件

- `模型与代码/run_switch_state_dab_cps_pilot.py`
- `数据/switch_state_dab_cps_trials.csv`
- `数据/switch_state_dab_cps_summary.csv`
- `数据/switch_state_dab_cps_config.json`
- `数据/switch_state_dab_cps_summary.png`
- `数据/switch_state_dab_representative_waveform.png`
"""
    (NOTE_DIR / "switch_state_dab_cps_pilot_record.md").write_text(note, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
