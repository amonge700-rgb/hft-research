"""Experiment 10A: validate the segmented matrix model and test Cps resolution."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Numerical validation must not depend on plotting.
    plt = None

from hft_segmented_ladder import (
    make_four_section_baseline,
    perturb_local_cps,
    recover_all_node_voltages,
    solve_frequency,
    solve_sweep,
    validate_parameters,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def complex_observation(y: np.ndarray) -> np.ndarray:
    """Use reciprocal independent entries Y11, Y12 and Y22."""

    values = np.column_stack([y[:, 0, 0], y[:, 0, 1], y[:, 1, 1]])
    # Channel normalization prevents Y11 from dominating solely by units/scale.
    scale = np.maximum(np.sqrt(np.mean(np.abs(values) ** 2, axis=0)), 1e-15)
    normalized = values / scale
    return np.r_[normalized.real.ravel(), normalized.imag.ravel()]


def log_sensitivity(freq: np.ndarray, baseline, relative_step: float = 1e-4) -> np.ndarray:
    columns = []
    for section in range(baseline.n):
        plus = perturb_local_cps(baseline, section, relative_step)
        minus = perturb_local_cps(baseline, section, -relative_step)
        y_plus = solve_sweep(freq, plus)["port_admittance"]
        y_minus = solve_sweep(freq, minus)["port_admittance"]
        # d h / d log(C_i), using a symmetric log-like perturbation.
        columns.append(
            (complex_observation(y_plus) - complex_observation(y_minus))
            / (2.0 * relative_step)
        )
    return np.column_stack(columns)


def correlation_matrix(jacobian: np.ndarray) -> np.ndarray:
    unit = jacobian / np.maximum(np.linalg.norm(jacobian, axis=0, keepdims=True), 1e-30)
    return unit.T @ unit


def run_checks(freq: np.ndarray, params) -> dict[str, float]:
    checks = validate_parameters(params)
    reciprocity = []
    passivity_floor = []
    for f in freq:
        solution = solve_frequency(float(f), params)
        y = solution.port_admittance
        reciprocity.append(abs(y[0, 1] - y[1, 0]) / max(np.linalg.norm(y), 1e-30))
        hermitian = 0.5 * (y + y.conj().T)
        passivity_floor.append(np.linalg.eigvalsh(hermitian).min())

    test_frequency = 1.0e6
    port_voltage = np.array([1.0 + 0.0j, 0.37 - 0.11j])
    voltage, port_current = recover_all_node_voltages(test_frequency, params, port_voltage)
    solution = solve_frequency(test_frequency, params)
    full_current = solution.nodal_admittance @ voltage
    internal = [k for k in range(2 * params.n) if k not in (0, params.n)]
    kcl_internal = np.linalg.norm(full_current[internal])
    direct_port_error = np.linalg.norm(
        port_current - solution.port_admittance @ port_voltage
    )

    checks.update(
        {
            "max_relative_reciprocity_error": float(max(reciprocity)),
            "minimum_port_passivity_eigenvalue_s": float(min(passivity_floor)),
            "internal_kcl_residual_a": float(kcl_internal),
            "kron_direct_port_current_error_a": float(direct_port_error),
        }
    )
    return checks


def save_csv(path: Path, headers: list[str], rows: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    params = make_four_section_baseline()
    freq = np.geomspace(1.0e4, 2.0e7, 260)
    sweep = solve_sweep(freq, params)
    y = sweep["port_admittance"]

    checks = run_checks(freq, params)
    jacobian = log_sensitivity(freq, params)
    fisher = jacobian.T @ jacobian
    fisher_eigenvalues = np.linalg.eigvalsh(fisher)
    fisher_condition = float(fisher_eigenvalues.max() / fisher_eigenvalues.min())
    correlation = correlation_matrix(jacobian)

    sensitivity_norm = np.linalg.norm(jacobian, axis=0)
    summary = {
        **checks,
        "frequency_min_hz": float(freq.min()),
        "frequency_max_hz": float(freq.max()),
        "frequency_count": int(freq.size),
        "fisher_eigenvalues": fisher_eigenvalues.tolist(),
        "fisher_condition_number": fisher_condition,
        "jacobian_rank": int(np.linalg.matrix_rank(jacobian, tol=1e-9 * np.linalg.svd(jacobian, compute_uv=False)[0])),
        "maximum_offdiagonal_sensitivity_correlation": float(
            np.max(np.abs(correlation - np.eye(params.n)))
        ),
        "section_sensitivity_norms": sensitivity_norm.tolist(),
        "baseline_total_cps_pf": float(params.interwinding_capacitance.sum() * 1e12),
    }
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_csv(
        RESULTS / "port_admittance.csv",
        [
            "frequency_hz",
            "Y11_real_s",
            "Y11_imag_s",
            "Y12_real_s",
            "Y12_imag_s",
            "Y22_real_s",
            "Y22_imag_s",
        ],
        [
            [
                float(freq[k]),
                float(y[k, 0, 0].real),
                float(y[k, 0, 0].imag),
                float(y[k, 0, 1].real),
                float(y[k, 0, 1].imag),
                float(y[k, 1, 1].real),
                float(y[k, 1, 1].imag),
            ]
            for k in range(freq.size)
        ],
    )
    save_csv(
        RESULTS / "cps_sensitivity_correlation.csv",
        ["section", "Cps1", "Cps2", "Cps3", "Cps4"],
        [[i + 1, *correlation[i].tolist()] for i in range(params.n)],
    )
    save_csv(
        RESULTS / "fisher_matrix.csv",
        ["section", "Cps1", "Cps2", "Cps3", "Cps4"],
        [[i + 1, *fisher[i].tolist()] for i in range(params.n)],
    )

    if plt is None:
        print("matplotlib is unavailable; numerical JSON/CSV results were saved without figures.")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    axes[0, 0].loglog(freq, np.abs(y[:, 0, 0]), label=r"$|Y_{11}|$")
    axes[0, 0].loglog(freq, np.abs(y[:, 0, 1]), label=r"$|Y_{12}|$")
    axes[0, 0].loglog(freq, np.abs(y[:, 1, 1]), label=r"$|Y_{22}|$")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Admittance magnitude (S)")
    axes[0, 0].set_title("Kron-reduced two-port response")
    axes[0, 0].grid(True, which="both", alpha=0.25)
    axes[0, 0].legend()

    for section in range(params.n):
        changed = solve_sweep(
            freq, perturb_local_cps(params, section, 0.10)
        )["port_admittance"]
        delta = np.linalg.norm(changed - y, axis=(1, 2)) / np.maximum(
            np.linalg.norm(y, axis=(1, 2)), 1e-30
        )
        axes[0, 1].loglog(freq, delta, label=f"Cps{section + 1} +10%")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Relative port-response change")
    axes[0, 1].set_title("Local perturbation signatures")
    axes[0, 1].grid(True, which="both", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    image = axes[1, 0].imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1, 0].set_xticks(range(params.n), [f"Cps{i+1}" for i in range(params.n)])
    axes[1, 0].set_yticks(range(params.n), [f"Cps{i+1}" for i in range(params.n)])
    axes[1, 0].set_title("Normalized sensitivity correlation")
    for i in range(params.n):
        for j in range(params.n):
            axes[1, 0].text(j, i, f"{correlation[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].semilogy(
        np.arange(1, params.n + 1), np.maximum(fisher_eigenvalues[::-1], 1e-30), "o-"
    )
    axes[1, 1].set_xlabel("Eigenvalue index")
    axes[1, 1].set_ylabel("Fisher eigenvalue (normalized)")
    axes[1, 1].set_title(f"Fisher spectrum, condition={fisher_condition:.2e}")
    axes[1, 1].grid(True, which="both", alpha=0.25)

    fig.savefig(RESULTS / "forward_and_identifiability.png", dpi=220)
    fig.savefig(RESULTS / "forward_and_identifiability.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
