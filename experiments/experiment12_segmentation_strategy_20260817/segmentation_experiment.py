"""EXP-012: compare contiguous winding segmentation strategies.

The reference is a passive synthetic turn-level two-winding network.  It is
not a calibrated model of a specific transformer.  Every coarse model is
obtained from the same congruence projection, so the comparison isolates the
effect of segmentation boundaries instead of parameter-refitting capacity.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


@dataclass(frozen=True)
class FineModel:
    n: int
    turns_ratio: float
    resistance: np.ndarray
    inductance: np.ndarray
    capacitance: np.ndarray
    incidence: np.ndarray
    cps_components: tuple[np.ndarray, ...]


def stamp_pair(matrix: np.ndarray, a: int, b: int | None, value: float) -> None:
    matrix[a, a] += value
    if b is not None:
        matrix[b, b] += value
        matrix[a, b] -= value
        matrix[b, a] -= value


def branch_incidence(n: int) -> np.ndarray:
    incidence = np.zeros((2 * n, 2 * n))
    for winding in range(2):
        for k in range(n):
            node = winding * n + k
            branch = winding * n + k
            incidence[node, branch] += 1.0
            if k + 1 < n:
                incidence[node + 1, branch] -= 1.0
    return incidence


def make_passive_reference(config: dict) -> FineModel:
    """Create a reproducible full-coupling synthetic reference model."""

    n = int(config["fine_sections_per_winding"])
    ratio = float(config["turns_ratio"])
    rng = np.random.default_rng(int(config["seed"]))
    position = (np.arange(n) + 0.5) / n

    # Positive series resistance with smooth spatial nonuniformity.
    primary_shape = 1.0 + 0.18 * np.sin(2 * np.pi * position) + 0.05 * rng.normal(size=n)
    primary_shape = np.maximum(primary_shape, 0.55)
    rp = 0.18 * primary_shape / primary_shape.sum()
    secondary_shape = 1.0 + 0.12 * np.cos(2 * np.pi * position + 0.3)
    rs = (0.18 / ratio**2) * secondary_shape / secondary_shape.sum()
    resistance = np.r_[rp, rs]

    # SPD inductance: positive leakage diagonal + PSD common/local flux terms.
    leakage_p = 3.6e-6 / n * (1.0 + 0.22 * np.cos(2 * np.pi * position))
    leakage_s = leakage_p / ratio**2 * (1.0 + 0.08 * np.sin(4 * np.pi * position))
    leakage = np.r_[leakage_p, leakage_s]
    flux = np.r_[np.ones(n), -np.ones(n) / ratio]
    common_flux = (220e-6 / n**2) * np.outer(flux, flux)
    distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    kernel = np.exp(-distance / 3.2)
    local_flux = np.zeros((2 * n, 2 * n))
    local_flux[:n, :n] = 0.22e-6 / n * kernel
    local_flux[n:, n:] = 0.22e-6 / (ratio**2 * n) * kernel
    inductance = np.diag(leakage) + common_flux + local_flux

    capacitance = np.zeros((2 * n, 2 * n))
    # Longitudinal capacitance varies strongly near layer transitions.
    layer_edges = np.cumsum(config["physical_layer_sizes"])[:-1]
    for winding in range(2):
        offset = winding * n
        base = 10.0e-12 if winding == 0 else 6.0e-12
        for k in range(n):
            edge_boost = 1.0 + 1.8 * sum(np.exp(-0.5 * ((k + 1 - e) / 0.8) ** 2) for e in layer_edges)
            b = offset + k + 1 if k + 1 < n else None
            stamp_pair(capacitance, offset + k, b, base * edge_boost)

    # Ground capacitance has end effects and asymmetric shield proximity.
    for k in range(n):
        end = np.exp(-k / 3.5) + 0.7 * np.exp(-(n - 1 - k) / 4.0)
        stamp_pair(capacitance, k, None, (0.45 + 0.75 * end) * 1e-12)
        stamp_pair(capacitance, n + k, None, (0.35 + 0.55 * end) * 1e-12)

    # Full interwinding capacitance kernel. Components group each primary row
    # and provide local Cps perturbations for port-sensitivity features.
    coupling = np.exp(-0.5 * (np.subtract.outer(np.arange(n), np.arange(n)) / 1.55) ** 2)
    coupling *= 1.0 + 0.18 * np.sin(2 * np.pi * position)[:, None]
    coupling = 28.36e-12 * coupling / coupling.sum()
    components = []
    for i in range(n):
        component = np.zeros_like(capacitance)
        for j in range(n):
            stamp_pair(component, i, n + j, coupling[i, j])
        capacitance += component
        components.append(component)

    model = FineModel(
        n=n,
        turns_ratio=ratio,
        resistance=resistance,
        inductance=inductance,
        capacitance=capacitance,
        incidence=branch_incidence(n),
        cps_components=tuple(components),
    )
    validate_reference(model)
    return model


def validate_reference(model: FineModel) -> dict[str, float]:
    eig_l = np.linalg.eigvalsh(0.5 * (model.inductance + model.inductance.T))
    eig_c = np.linalg.eigvalsh(0.5 * (model.capacitance + model.capacitance.T))
    if eig_l.min() <= 0 or eig_c.min() < -1e-18 or np.any(model.resistance <= 0):
        raise ValueError("Reference network violates passivity constraints")
    return {
        "minimum_inductance_eigenvalue_h": float(eig_l.min()),
        "minimum_capacitance_eigenvalue_f": float(eig_c.min()),
        "inductance_symmetry_residual": float(np.linalg.norm(model.inductance - model.inductance.T)),
        "capacitance_symmetry_residual": float(np.linalg.norm(model.capacitance - model.capacitance.T)),
    }


def fine_nodal_admittance(model: FineModel, frequency_hz: float, capacitance: np.ndarray | None = None) -> np.ndarray:
    omega = 2 * np.pi * float(frequency_hz)
    z = np.diag(model.resistance) + 1j * omega * model.inductance
    c = model.capacitance if capacitance is None else capacitance
    return model.incidence @ np.linalg.solve(z, model.incidence.T) + 1j * omega * c


def kron_port_and_voltage(y: np.ndarray, n_each: int, port_voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ports = np.array([0, n_each], dtype=int)
    internal = np.setdiff1d(np.arange(2 * n_each), ports)
    ypp = y[np.ix_(ports, ports)]
    ypi = y[np.ix_(ports, internal)]
    yip = y[np.ix_(internal, ports)]
    yii = y[np.ix_(internal, internal)]
    recovery = -np.linalg.solve(yii, yip)
    yport = ypp + ypi @ recovery
    voltage = np.zeros(2 * n_each, dtype=complex)
    voltage[ports] = port_voltage
    voltage[internal] = recovery @ port_voltage
    return yport, voltage


def uniform_boundaries(n: int, m: int) -> list[int]:
    return np.rint(np.linspace(0, n, m + 1)).astype(int).tolist()


def physical_boundaries(n: int, m: int, layer_sizes: list[int]) -> list[int]:
    """Allocate segments within nonuniform layers while preserving layer edges."""

    layer_edges = np.r_[0, np.cumsum(layer_sizes)]
    layer_count = len(layer_sizes)
    if m < layer_count:
        target = np.arange(1, m) * n / m
        internal_edges = list(layer_edges[1:-1])
        selected = []
        for value in target:
            available = [edge for edge in internal_edges if edge not in selected]
            selected.append(min(available, key=lambda edge: abs(edge - value)))
        return [0] + sorted(selected) + [n]

    allocation = np.ones(layer_count, dtype=int)
    remaining = m - layer_count
    if remaining:
        quota = remaining * np.asarray(layer_sizes, dtype=float) / n
        allocation += np.floor(quota).astype(int)
        leftover = m - int(allocation.sum())
        order = np.argsort(-(quota - np.floor(quota)))
        allocation[order[:leftover]] += 1

    boundaries = [0]
    for layer, segment_count in enumerate(allocation):
        left, right = int(layer_edges[layer]), int(layer_edges[layer + 1])
        local = np.rint(np.linspace(left, right, segment_count + 1)).astype(int)
        boundaries.extend(local[1:].tolist())
    if len(set(boundaries)) != m + 1:
        raise ValueError("Physical segmentation produced duplicate boundaries")
    return boundaries


def prolongation(n: int, boundaries: list[int]) -> np.ndarray:
    """Piecewise-linear voltage prolongation, with the final boundary grounded."""

    m = len(boundaries) - 1
    p = np.zeros((n, m))
    for segment in range(m):
        left, right = boundaries[segment], boundaries[segment + 1]
        width = right - left
        for x in range(left, right):
            alpha = (x - left) / width
            p[x, segment] += 1.0 - alpha
            if segment + 1 < m:
                p[x, segment + 1] += alpha
    return p


def block_prolongation(n: int, boundaries: list[int]) -> np.ndarray:
    pw = prolongation(n, boundaries)
    zeros = np.zeros_like(pw)
    return np.block([[pw, zeros], [zeros, pw]])


def branch_current_embedding(n: int, boundaries: list[int]) -> np.ndarray:
    """Map each coarse series current to all fine branches in its group."""

    m = len(boundaries) - 1
    tw = np.zeros((n, m))
    for group, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        tw[left:right, group] = 1.0
    zeros = np.zeros_like(tw)
    return np.block([[tw, zeros], [zeros, tw]])


def build_coarse_matrices(model: FineModel, boundaries: list[int]) -> tuple[np.ndarray, ...]:
    """Energy-equivalent, structure-preserving coarse ladder matrices.

    The branch embedding enforces equal current through fine series branches
    merged into one coarse section.  The nodal prolongation approximates the
    internal voltage distribution by piecewise-linear interpolation.
    """

    p = block_prolongation(model.n, boundaries)
    t = branch_current_embedding(model.n, boundaries)
    resistance = t.T @ np.diag(model.resistance) @ t
    inductance = t.T @ model.inductance @ t
    capacitance = p.T @ model.capacitance @ p
    incidence = branch_incidence(len(boundaries) - 1)
    return resistance, inductance, capacitance, incidence, p


def port_observation(yport: np.ndarray) -> np.ndarray:
    values = np.array([yport[0, 0], yport[0, 1], yport[1, 1]])
    return np.r_[values.real, values.imag]


def build_information_features(model: FineModel, frequencies: np.ndarray) -> np.ndarray:
    """Combine internal-voltage change and whitened local-Cps port signatures."""

    n = model.n
    base_obs = []
    voltage_features = np.zeros((n, 4 * len(frequencies)))
    excitations = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    for k, f in enumerate(frequencies):
        y = fine_nodal_admittance(model, f)
        yport, _ = kron_port_and_voltage(y, n, excitations[0])
        base_obs.append(port_observation(yport))
        for e, excitation in enumerate(excitations):
            _, voltage = kron_port_and_voltage(y, n, excitation)
            pair = 0.5 * (voltage[:n] + voltage[n:])
            voltage_features[:, (2 * e) * len(frequencies) + k] = pair.real
            voltage_features[:, (2 * e + 1) * len(frequencies) + k] = pair.imag

    base_obs = np.asarray(base_obs)
    channel_scale = np.maximum(np.sqrt(np.mean(base_obs**2, axis=0)), 1e-14)
    sensitivity = np.zeros((n, len(frequencies) * 6))
    relative_step = 1e-3
    for i, component in enumerate(model.cps_components):
        rows = []
        for f in frequencies:
            plus = model.capacitance + relative_step * component
            minus = model.capacitance - relative_step * component
            yp, _ = kron_port_and_voltage(fine_nodal_admittance(model, f, plus), n, excitations[0])
            ym, _ = kron_port_and_voltage(fine_nodal_admittance(model, f, minus), n, excitations[0])
            rows.append((port_observation(yp) - port_observation(ym)) / (2 * relative_step) / channel_scale)
        sensitivity[i] = np.asarray(rows).ravel()

    def normalize_rows(x: np.ndarray) -> np.ndarray:
        scale = np.maximum(np.std(x, axis=0, keepdims=True), 1e-12)
        return (x - np.mean(x, axis=0, keepdims=True)) / scale

    return np.c_[normalize_rows(voltage_features), normalize_rows(sensitivity)]


def information_boundaries(features: np.ndarray, m: int) -> list[int]:
    """Place boundaries at the largest adjacent electrical-behavior changes."""

    n = features.shape[0]
    left = features[:-1]
    right = features[1:]
    jump = np.linalg.norm(right - left, axis=1) / np.sqrt(features.shape[1])
    # Mild spacing penalty prevents adjacent singleton boundaries.
    selected: list[int] = []
    for _ in range(m - 1):
        candidate_score = jump.copy()
        for boundary in selected:
            distance = np.abs(np.arange(1, n) - boundary)
            candidate_score *= 1.0 - 0.72 * np.exp(-distance / 1.2)
        candidate_score[[b - 1 for b in selected]] = -np.inf
        selected.append(int(np.argmax(candidate_score) + 1))
    return [0] + sorted(selected) + [n]


def local_maxima(values: np.ndarray, maximum_count: int = 8) -> np.ndarray:
    logv = np.log(np.maximum(values, 1e-30))
    index = np.where((logv[1:-1] > logv[:-2]) & (logv[1:-1] >= logv[2:]))[0] + 1
    if index.size == 0:
        return index
    prominence = logv[index] - 0.5 * (logv[index - 1] + logv[index + 1])
    keep = index[np.argsort(prominence)[::-1][:maximum_count]]
    return np.sort(keep)


def match_resonances(reference_frequency: np.ndarray, candidate_frequency: np.ndarray) -> tuple[float, int]:
    if reference_frequency.size == 0:
        return 0.0, 0
    if candidate_frequency.size == 0:
        return 1.0, int(reference_frequency.size)
    unused = list(candidate_frequency)
    errors = []
    for value in reference_frequency:
        nearest = min(unused, key=lambda x: abs(np.log(x / value)))
        errors.append(abs(nearest - value) / value)
        unused.remove(nearest)
        if not unused:
            break
    missing = reference_frequency.size - len(errors)
    errors.extend([1.0] * missing)
    return float(np.mean(errors)), int(missing)


def refine_peak_frequencies(frequency: np.ndarray, trace: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Quadratic interpolation in log-frequency/log-magnitude coordinates."""

    x = np.log(frequency)
    y = np.log(np.maximum(trace, 1e-30))
    refined = []
    for i in index:
        if i <= 0 or i >= len(frequency) - 1:
            refined.append(frequency[i])
            continue
        coefficient = np.polyfit(x[i - 1 : i + 2], y[i - 1 : i + 2], 2)
        vertex = -coefficient[1] / (2 * coefficient[0]) if coefficient[0] < 0 else x[i]
        vertex = np.clip(vertex, x[i - 1], x[i + 1])
        refined.append(float(np.exp(vertex)))
    return np.asarray(refined)


def key_modal_frequencies(
    inductance: np.ndarray,
    capacitance: np.ndarray,
    incidence: np.ndarray,
    port_nodes: list[int],
    frequency_min: float,
    frequency_max: float,
    maximum_modes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the lossless generalized eigenproblem and rank port-visible modes."""

    stiffness = incidence @ np.linalg.solve(inductance, incidence.T)
    chol = np.linalg.cholesky(capacitance)
    left = np.linalg.solve(chol, stiffness)
    transformed = np.linalg.solve(chol, left.T).T
    eigenvalue, eigenvector_t = np.linalg.eigh(0.5 * (transformed + transformed.T))
    valid = eigenvalue > 0
    eigenvalue = eigenvalue[valid]
    eigenvector = np.linalg.solve(chol.T, eigenvector_t[:, valid])
    frequency = np.sqrt(eigenvalue) / (2 * np.pi)
    participation = np.linalg.norm(eigenvector[port_nodes], axis=0) / np.maximum(
        np.linalg.norm(eigenvector, axis=0), 1e-30
    )
    in_band = (frequency >= frequency_min) & (frequency <= frequency_max)
    frequency, participation = frequency[in_band], participation[in_band]
    selected = np.argsort(participation)[::-1][:maximum_modes]
    order = np.argsort(frequency[selected])
    return frequency[selected][order], participation[selected][order]


def evaluate_strategy(
    model: FineModel,
    frequencies: np.ndarray,
    reference_y: np.ndarray,
    reference_voltage: np.ndarray,
    boundaries: list[int],
) -> dict:
    n = model.n
    m = len(boundaries) - 1
    reduction_start = time.perf_counter()
    resistance, inductance, capacitance, incidence, p = build_coarse_matrices(model, boundaries)
    reduction_elapsed = time.perf_counter() - reduction_start
    candidate_y = np.empty_like(reference_y)
    candidate_voltage = np.empty_like(reference_voltage)
    start = time.perf_counter()
    for k, f in enumerate(frequencies):
        omega = 2 * np.pi * float(f)
        impedance = resistance + 1j * omega * inductance
        yc = incidence @ np.linalg.solve(impedance, incidence.T) + 1j * omega * capacitance
        yport, vc = kron_port_and_voltage(yc, m, np.array([1.0, 0.0]))
        candidate_y[k] = yport
        candidate_voltage[k] = p @ vc
    elapsed = time.perf_counter() - start

    port_error = np.linalg.norm(candidate_y - reference_y) / np.linalg.norm(reference_y)
    internal_mask = np.ones(2 * n, dtype=bool)
    internal_mask[[0, n]] = False
    ref_internal = np.abs(reference_voltage[:, internal_mask])
    candidate_internal = np.abs(candidate_voltage[:, internal_mask])

    # Driving-point impedance exposes the dominant parallel resonance even
    # when the corresponding admittance magnitude remains monotonic.
    ref_trace = np.abs(np.linalg.inv(reference_y)[:, 0, 0])
    candidate_trace = np.abs(np.linalg.inv(candidate_y)[:, 0, 0])
    ref_peak_index = local_maxima(ref_trace)
    candidate_peak_index = local_maxima(candidate_trace)
    reference_resonances = refine_peak_frequencies(frequencies, ref_trace, ref_peak_index)
    candidate_resonances = refine_peak_frequencies(frequencies, candidate_trace, candidate_peak_index)
    observed_resonance_error, observed_missing = match_resonances(
        reference_resonances, candidate_resonances
    )

    reference_modes, reference_participation = key_modal_frequencies(
        model.inductance,
        model.capacitance,
        model.incidence,
        [0, n],
        float(frequencies.min()),
        float(frequencies.max()),
    )
    candidate_modes, candidate_participation = key_modal_frequencies(
        inductance,
        capacitance,
        incidence,
        [0, m],
        float(frequencies.min()),
        float(frequencies.max()),
    )
    modal_error, modal_missing = match_resonances(reference_modes, candidate_modes)

    ref_global_index = np.unravel_index(np.argmax(ref_internal), ref_internal.shape)
    candidate_global_index = np.unravel_index(np.argmax(candidate_internal), candidate_internal.shape)
    ref_global_peak = ref_internal[ref_global_index]
    candidate_global_peak = candidate_internal[candidate_global_index]
    internal_peak_error = abs(candidate_global_peak - ref_global_peak) / max(ref_global_peak, 1e-12)
    # Fine-node labels after removing the two external ports.
    internal_nodes = np.where(internal_mask)[0]
    ref_location = int(internal_nodes[ref_global_index[1]])
    candidate_location = int(internal_nodes[candidate_global_index[1]])
    location_distance = abs(candidate_location - ref_location)

    # Adjacent-turn/section voltage drop is a direct insulation-stress proxy.
    def branch_drop(voltage: np.ndarray) -> np.ndarray:
        primary = np.diff(np.c_[voltage[:, :n], np.zeros(len(voltage))], axis=1)
        secondary = np.diff(np.c_[voltage[:, n:], np.zeros(len(voltage))], axis=1)
        return np.abs(np.c_[primary, secondary])

    ref_drop = branch_drop(reference_voltage)
    candidate_drop = branch_drop(candidate_voltage)
    ref_drop_index = np.unravel_index(np.argmax(ref_drop), ref_drop.shape)
    candidate_drop_index = np.unravel_index(np.argmax(candidate_drop), candidate_drop.shape)
    stress_error = abs(candidate_drop[candidate_drop_index] - ref_drop[ref_drop_index]) / max(ref_drop[ref_drop_index], 1e-12)

    return {
        "boundaries": boundaries,
        "coarse_sections_per_winding": m,
        "port_relative_error": float(port_error),
        "mean_resonance_error": modal_error,
        "missing_resonances": modal_missing,
        "observed_resonance_error": observed_resonance_error,
        "observed_missing_resonances": observed_missing,
        "reference_resonance_count": int(ref_peak_index.size),
        "candidate_resonance_count": int(candidate_peak_index.size),
        "spurious_resonances": int(max(candidate_peak_index.size - ref_peak_index.size, 0)),
        "reference_key_modal_frequencies_hz": reference_modes.tolist(),
        "candidate_key_modal_frequencies_hz": candidate_modes.tolist(),
        "reference_modal_participation": reference_participation.tolist(),
        "candidate_modal_participation": candidate_participation.tolist(),
        "maximum_internal_peak_error": float(internal_peak_error),
        "reference_internal_peak_node": ref_location,
        "candidate_internal_peak_node": candidate_location,
        "internal_peak_location_distance": int(location_distance),
        "maximum_adjacent_voltage_stress_error": float(stress_error),
        "reference_stress_branch": int(ref_drop_index[1]),
        "candidate_stress_branch": int(candidate_drop_index[1]),
        "matrix_dimension": int(2 * m),
        "offline_reduction_seconds": float(reduction_elapsed),
        "runtime_seconds": float(elapsed),
        "candidate_y": candidate_y,
        "candidate_voltage": candidate_voltage,
    }


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    model = make_passive_reference(config)
    np.savez_compressed(
        RESULTS / "reference_model_matrices.npz",
        resistance=model.resistance,
        inductance=model.inductance,
        capacitance=model.capacitance,
        incidence=model.incidence,
    )
    frequencies = np.geomspace(config["frequency_min_hz"], config["frequency_max_hz"], config["frequency_points"])
    information_frequencies = np.geomspace(config["frequency_min_hz"], config["frequency_max_hz"], config["information_frequency_points"])

    reference_y = np.empty((len(frequencies), 2, 2), dtype=complex)
    reference_voltage = np.empty((len(frequencies), 2 * model.n), dtype=complex)
    reference_start = time.perf_counter()
    for k, f in enumerate(frequencies):
        y = fine_nodal_admittance(model, f)
        reference_y[k], reference_voltage[k] = kron_port_and_voltage(y, model.n, np.array([1.0, 0.0]))
    reference_runtime = time.perf_counter() - reference_start

    feature_start = time.perf_counter()
    information_features = build_information_features(model, information_frequencies)
    feature_runtime = time.perf_counter() - feature_start

    rows = []
    detailed = {}
    layer_sizes = list(config["physical_layer_sizes"])
    for m in config["coarse_sections_per_winding"]:
        partitions = {
            "uniform": uniform_boundaries(model.n, m),
            "physical": physical_boundaries(model.n, m, layer_sizes),
            "information_guided": information_boundaries(information_features, m),
        }
        for strategy, boundaries in partitions.items():
            result = evaluate_strategy(model, frequencies, reference_y, reference_voltage, boundaries)
            record = {
                "strategy": strategy,
                "coarse_sections_per_winding": m,
                "boundaries": "-".join(map(str, boundaries)),
                "port_error_percent": 100 * result["port_relative_error"],
                "resonance_error_percent": 100 * result["mean_resonance_error"],
                "missing_resonances": result["missing_resonances"],
                "observed_resonance_error_percent": 100 * result["observed_resonance_error"],
                "observed_missing_resonances": result["observed_missing_resonances"],
                "reference_resonance_count": result["reference_resonance_count"],
                "candidate_resonance_count": result["candidate_resonance_count"],
                "spurious_resonances": result["spurious_resonances"],
                "internal_peak_error_percent": 100 * result["maximum_internal_peak_error"],
                "internal_peak_reference_node": result["reference_internal_peak_node"],
                "internal_peak_candidate_node": result["candidate_internal_peak_node"],
                "internal_peak_location_distance": result["internal_peak_location_distance"],
                "adjacent_voltage_stress_error_percent": 100 * result["maximum_adjacent_voltage_stress_error"],
                "reference_stress_branch": result["reference_stress_branch"],
                "candidate_stress_branch": result["candidate_stress_branch"],
                "matrix_dimension": result["matrix_dimension"],
                "offline_reduction_seconds": result["offline_reduction_seconds"],
                "runtime_seconds": result["runtime_seconds"],
                "speedup_vs_reference": reference_runtime / max(result["runtime_seconds"], 1e-12),
            }
            record["meets_all_accuracy_targets"] = bool(
                record["port_error_percent"] <= config["port_error_target_percent"]
                and record["resonance_error_percent"] <= config["resonance_error_target_percent"]
                and record["internal_peak_error_percent"] <= config["internal_peak_error_target_percent"]
                and record["adjacent_voltage_stress_error_percent"]
                <= config["adjacent_voltage_stress_error_target_percent"]
            )
            rows.append(record)
            detailed[f"{strategy}_{m}"] = {
                **record,
                "port_y_real": result["candidate_y"].real.tolist(),
                "port_y_imag": result["candidate_y"].imag.tolist(),
            }

    columns = list(rows[0].keys())
    write_csv(RESULTS / "segmentation_metrics.csv", rows, columns)
    validation = {
        **validate_reference(model),
        "reference_runtime_seconds": reference_runtime,
        "information_feature_runtime_seconds": feature_runtime,
        "reference_matrix_dimension": 2 * model.n,
        "reference_reciprocity_residual": float(np.max(np.abs(reference_y[:, 0, 1] - reference_y[:, 1, 0]))),
        "experiment_scope": "Synthetic passive constant-parameter reference; no FEM or hardware calibration.",
    }
    (RESULTS / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (RESULTS / "detailed_results.json").write_text(json.dumps(detailed), encoding="utf-8")

    if plt is not None:
        styles = {"uniform": "o-", "physical": "s-", "information_guided": "^-"}
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
        for strategy, style in styles.items():
            selected = sorted((row for row in rows if row["strategy"] == strategy), key=lambda row: row["coarse_sections_per_winding"])
            x = [row["coarse_sections_per_winding"] for row in selected]
            axes[0, 0].semilogy(x, [row["port_error_percent"] for row in selected], style, label=strategy)
            axes[0, 1].semilogy(x, [row["resonance_error_percent"] for row in selected], style, label=strategy)
            axes[1, 0].semilogy(x, [row["internal_peak_error_percent"] for row in selected], style, label=strategy)
            axes[1, 1].semilogy(x, [max(row["adjacent_voltage_stress_error_percent"], 1e-8) for row in selected], style, label=strategy)
        axes[0, 0].axhline(config["port_error_target_percent"], color="k", ls="--", lw=0.9)
        axes[0, 1].axhline(config["resonance_error_target_percent"], color="k", ls="--", lw=0.9)
        axes[1, 0].axhline(config["internal_peak_error_target_percent"], color="k", ls="--", lw=0.9)
        axes[1, 1].axhline(config["adjacent_voltage_stress_error_target_percent"], color="k", ls="--", lw=0.9)
        titles = ["Port admittance error", "Internal-resonance error", "Global internal peak error", "Maximum adjacent-voltage-stress error"]
        ylabels = ["Error (%)", "Error (%)", "Error (%)", "Error (%)"]
        for axis, title, ylabel in zip(axes.ravel(), titles, ylabels):
            axis.set_title(title)
            axis.set_xlabel("Coarse sections per winding")
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        fig.savefig(FIGURES / "segmentation_comparison.png", dpi=220)
        fig.savefig(FIGURES / "segmentation_comparison.pdf")
        plt.close(fig)

        best = min(rows, key=lambda row: row["port_error_percent"] + row["internal_peak_error_percent"])
        key = f'{best["strategy"]}_{best["coarse_sections_per_winding"]}'
        candidate = np.asarray(detailed[key]["port_y_real"]) + 1j * np.asarray(detailed[key]["port_y_imag"])
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
        axes[0].loglog(frequencies, np.abs(reference_y[:, 0, 1]), label="turn-level reference")
        axes[0].loglog(frequencies, np.abs(candidate[:, 0, 1]), "--", label=key)
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel(r"$|Y_{12}|$ (S)")
        axes[0].grid(True, which="both", alpha=0.25)
        axes[0].legend()
        internal_mask = np.ones(2 * model.n, dtype=bool)
        internal_mask[[0, model.n]] = False
        ref_peak = np.max(np.abs(reference_voltage[:, internal_mask]), axis=1)
        chosen_boundaries = list(map(int, best["boundaries"].split("-")))
        result = evaluate_strategy(model, frequencies, reference_y, reference_voltage, chosen_boundaries)
        axes[1].semilogx(frequencies, ref_peak, label="turn-level reference")
        axes[1].semilogx(
            frequencies,
            np.max(np.abs(result["candidate_voltage"][:, internal_mask]), axis=1),
            "--",
            label=key,
        )
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Maximum internal voltage (p.u.)")
        axes[1].grid(True, which="both", alpha=0.25)
        axes[1].legend()
        fig.savefig(FIGURES / "best_model_response.png", dpi=220)
        fig.savefig(FIGURES / "best_model_response.pdf")
        plt.close(fig)

    minimum_qualifying = {}
    for strategy in ("uniform", "physical", "information_guided"):
        qualifying = [
            row["coarse_sections_per_winding"]
            for row in rows
            if row["strategy"] == strategy and row["meets_all_accuracy_targets"]
        ]
        minimum_qualifying[strategy] = min(qualifying) if qualifying else None

    summary = {
        "validation": validation,
        "accuracy_targets_percent": {
            "port": config["port_error_target_percent"],
            "resonance": config["resonance_error_target_percent"],
            "internal_peak": config["internal_peak_error_target_percent"],
            "adjacent_voltage_stress": config["adjacent_voltage_stress_error_target_percent"],
        },
        "minimum_qualifying_sections_per_winding": minimum_qualifying,
        "best_by_combined_port_internal_error": min(
            rows, key=lambda row: row["port_error_percent"] + row["internal_peak_error_percent"]
        ),
        "rows": rows,
    }
    (RESULTS / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["best_by_combined_port_internal_error"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
