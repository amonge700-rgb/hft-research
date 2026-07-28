"""Experiment 10B: quantify unit-level versus grouped spatial resolution.

This deliberately uses a local linearized inverse problem around a known
healthy baseline.  The purpose is not to advertise a final estimator, but to
test whether the four local Cps directions are distinguishable before a more
expensive nonlinear/sparse inverse solver is built.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hft_segmented_ladder import (
    make_four_section_baseline,
    perturb_local_cps,
    solve_sweep,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def independent_channels(y: np.ndarray) -> np.ndarray:
    return np.column_stack([y[:, 0, 0], y[:, 0, 1], y[:, 1, 1]])


def pack(values: np.ndarray, channel_scale: np.ndarray) -> np.ndarray:
    normalized = values / channel_scale
    return np.r_[normalized.real.ravel(), normalized.imag.ravel()]


def build_local_model(freq: np.ndarray, params, step: float = 1e-4):
    baseline_y = independent_channels(solve_sweep(freq, params)["port_admittance"])
    channel_scale = np.maximum(
        np.sqrt(np.mean(np.abs(baseline_y) ** 2, axis=0)), 1e-15
    )
    h0 = pack(baseline_y, channel_scale)
    columns = []
    for section in range(params.n):
        yp = independent_channels(
            solve_sweep(freq, perturb_local_cps(params, section, step))[
                "port_admittance"
            ]
        )
        ym = independent_channels(
            solve_sweep(freq, perturb_local_cps(params, section, -step))[
                "port_admittance"
            ]
        )
        columns.append((pack(yp, channel_scale) - pack(ym, channel_scale)) / (2 * step))
    return baseline_y, channel_scale, h0, np.column_stack(columns)


def add_complex_noise(
    values: np.ndarray, relative_sigma: float, rng: np.random.Generator
) -> np.ndarray:
    noise = (
        rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape)
    ) / np.sqrt(2.0)
    floor = np.maximum(np.abs(values), 0.02 * np.max(np.abs(values), axis=0))
    return values + relative_sigma * floor * noise


def ridge_solve(jacobian: np.ndarray, residual: np.ndarray, relative_lambda: float) -> np.ndarray:
    gram = jacobian.T @ jacobian
    regularizer = relative_lambda * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(
        gram + regularizer * np.eye(gram.shape[0]), jacobian.T @ residual
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    params = make_four_section_baseline()
    freq = np.geomspace(1.0e4, 2.0e7, 260)
    baseline_y, scale, h0, jacobian = build_local_model(freq, params)

    # Two identifiable regions suggested by Experiment 10A:
    # region A = terminal section 1; region B = interior sections 2--4.
    grouping = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    grouped_jacobian = jacobian @ grouping

    noise_levels = [0.0005, 0.002, 0.005]
    fault_changes = [0.05, 0.10]
    repeats = 200
    rng = np.random.default_rng(20260726)
    detailed: list[dict] = []

    for noise in noise_levels:
        for fault_change in fault_changes:
            true_log_change = np.log1p(fault_change)
            for fault_section in range(params.n):
                truth_params = perturb_local_cps(params, fault_section, fault_change)
                truth_y = independent_channels(
                    solve_sweep(freq, truth_params)["port_admittance"]
                )
                true_group = 0 if fault_section == 0 else 1
                # The grouped parameter represents the fractional change of
                # total capacitance in that region.  A change in one of three
                # equal interior sections changes the group total by one third.
                group_size = 1 if true_group == 0 else 3
                true_group_log_change = np.log1p(fault_change / group_size)
                for repeat in range(repeats):
                    measured = add_complex_noise(truth_y, noise, rng)
                    residual = pack(measured, scale) - h0

                    delta_full = ridge_solve(jacobian, residual, relative_lambda=1e-5)
                    delta_group = ridge_solve(
                        grouped_jacobian, residual, relative_lambda=1e-5
                    )
                    predicted_section = int(np.argmax(delta_full))
                    predicted_group = int(np.argmax(delta_group))

                    detailed.append(
                        {
                            "noise_pct": 100.0 * noise,
                            "fault_change_pct": 100.0 * fault_change,
                            "fault_section": fault_section + 1,
                            "repeat": repeat,
                            "predicted_section": predicted_section + 1,
                            "full_location_correct": int(predicted_section == fault_section),
                            "predicted_group": predicted_group + 1,
                            "true_group": true_group + 1,
                            "group_location_correct": int(predicted_group == true_group),
                            "full_true_section_amplitude_error_pct": 100.0
                            * abs(delta_full[fault_section] - true_log_change)
                            / true_log_change,
                            "group_true_amplitude_error_pct": 100.0
                            * abs(delta_group[true_group] - true_group_log_change)
                            / true_group_log_change,
                            "delta_cps1_pct": 100.0 * np.expm1(delta_full[0]),
                            "delta_cps2_pct": 100.0 * np.expm1(delta_full[1]),
                            "delta_cps3_pct": 100.0 * np.expm1(delta_full[2]),
                            "delta_cps4_pct": 100.0 * np.expm1(delta_full[3]),
                        }
                    )

    summary: list[dict] = []
    for noise in noise_levels:
        for change in fault_changes:
            for section in range(params.n):
                selected = [
                    row
                    for row in detailed
                    if row["noise_pct"] == 100.0 * noise
                    and row["fault_change_pct"] == 100.0 * change
                    and row["fault_section"] == section + 1
                ]
                summary.append(
                    {
                        "noise_pct": 100.0 * noise,
                        "fault_change_pct": 100.0 * change,
                        "fault_section": section + 1,
                        "unit_location_accuracy_pct": 100.0
                        * np.mean([row["full_location_correct"] for row in selected]),
                        "group_location_accuracy_pct": 100.0
                        * np.mean([row["group_location_correct"] for row in selected]),
                        "unit_amplitude_mae_pct": float(
                            np.mean(
                                [
                                    row["full_true_section_amplitude_error_pct"]
                                    for row in selected
                                ]
                            )
                        ),
                        "group_amplitude_mae_pct": float(
                            np.mean(
                                [
                                    row["group_true_amplitude_error_pct"]
                                    for row in selected
                                ]
                            )
                        ),
                    }
                )

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    aggregate = {
        "sample_count": len(detailed),
        "frequency_count": int(freq.size),
        "jacobian_singular_values": singular_values.tolist(),
        "jacobian_condition_number": float(singular_values[0] / singular_values[-1]),
        "group_definition": {
            "group_1": ["Cps1"],
            "group_2": ["Cps2", "Cps3", "Cps4"],
        },
        "overall_unit_location_accuracy_pct": float(
            100.0 * np.mean([row["full_location_correct"] for row in detailed])
        ),
        "overall_group_location_accuracy_pct": float(
            100.0 * np.mean([row["group_location_correct"] for row in detailed])
        ),
        "overall_unit_amplitude_mae_pct": float(
            np.mean([row["full_true_section_amplitude_error_pct"] for row in detailed])
        ),
        "overall_group_amplitude_mae_pct": float(
            np.mean([row["group_true_amplitude_error_pct"] for row in detailed])
        ),
    }
    write_csv(RESULTS / "local_inversion_trials.csv", detailed)
    write_csv(RESULTS / "local_inversion_summary.csv", summary)
    (RESULTS / "local_inversion_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
