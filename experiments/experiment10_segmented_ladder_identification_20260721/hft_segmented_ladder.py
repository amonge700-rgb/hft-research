"""Matrix-domain segmented high-frequency-transformer ladder model.

The implementation follows the general branch-impedance / nodal-admittance
idea used in distributed transformer ladder models, but defines a topology
suited to a two-winding high-frequency transformer:

* each winding is divided into ``n`` series regions;
* series R-L branches are magnetically coupled through a full inductance matrix;
* longitudinal, ground, and interwinding capacitances are stamped directly
  into a Maxwell-style nodal capacitance matrix;
* the internal nodes are eliminated by a Schur complement (Kron reduction).

Port convention
---------------
The first node of each winding is an external port.  The last series branch of
each winding terminates at the common mathematical reference.  This is the
standard two-port measurement convention; it does not assert that the two
physical winding returns are bonded in the hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class LadderParameters:
    n: int
    turns_ratio: float
    resistance: np.ndarray
    inductance: np.ndarray
    longitudinal_capacitance: np.ndarray
    ground_capacitance: np.ndarray
    interwinding_capacitance: np.ndarray
    dielectric_conductance: np.ndarray

    def copy_with(self, **changes: object) -> "LadderParameters":
        return replace(self, **changes)


@dataclass(frozen=True)
class FrequencySolution:
    nodal_admittance: np.ndarray
    port_admittance: np.ndarray
    internal_recovery: np.ndarray
    impedance_matrix: np.ndarray
    capacitance_matrix: np.ndarray
    conductance_matrix: np.ndarray


def _stamp_pair(matrix: np.ndarray, a: int, b: int | None, value: float) -> None:
    """Stamp a passive element between nodes ``a`` and ``b``.

    ``b=None`` denotes the reference node.
    """

    matrix[a, a] += value
    if b is not None:
        matrix[b, b] += value
        matrix[a, b] -= value
        matrix[b, a] -= value


def branch_incidence(n: int) -> np.ndarray:
    """Return node-by-series-branch incidence matrix.

    Unknown nodes are p0..p(n-1), s0..s(n-1).  The nth boundary of each
    winding is the reference.  Branch currents are oriented from the external
    terminal toward the reference.
    """

    node_count = 2 * n
    branch_count = 2 * n
    gamma = np.zeros((node_count, branch_count), dtype=float)
    for winding in range(2):
        node_offset = winding * n
        branch_offset = winding * n
        for k in range(n):
            node_a = node_offset + k
            node_b = node_offset + k + 1 if k + 1 < n else None
            branch = branch_offset + k
            gamma[node_a, branch] += 1.0
            if node_b is not None:
                gamma[node_b, branch] -= 1.0
    return gamma


def capacitance_matrix(params: LadderParameters) -> np.ndarray:
    n = params.n
    cmat = np.zeros((2 * n, 2 * n), dtype=float)

    # Longitudinal capacitance is parallel to each series winding region.
    for winding in range(2):
        node_offset = winding * n
        branch_offset = winding * n
        for k in range(n):
            a = node_offset + k
            b = node_offset + k + 1 if k + 1 < n else None
            _stamp_pair(cmat, a, b, params.longitudinal_capacitance[branch_offset + k])

    # Capacitance from every retained winding node to core/shield/reference.
    for node, value in enumerate(params.ground_capacitance):
        _stamp_pair(cmat, node, None, value)

    # First model: corresponding primary/secondary regions are coupled.
    for k, value in enumerate(params.interwinding_capacitance):
        _stamp_pair(cmat, k, n + k, value)
    return cmat


def conductance_matrix(params: LadderParameters) -> np.ndarray:
    gmat = np.zeros((2 * params.n, 2 * params.n), dtype=float)
    for node, value in enumerate(params.dielectric_conductance):
        _stamp_pair(gmat, node, None, value)
    return gmat


def validate_parameters(params: LadderParameters) -> dict[str, float]:
    n = params.n
    expected_branch = 2 * n
    expected_node = 2 * n
    if params.resistance.shape != (expected_branch,):
        raise ValueError("resistance must contain 2*n series-branch values")
    if params.inductance.shape != (expected_branch, expected_branch):
        raise ValueError("inductance must be a (2*n) x (2*n) matrix")
    if params.longitudinal_capacitance.shape != (expected_branch,):
        raise ValueError("longitudinal_capacitance must contain 2*n values")
    if params.ground_capacitance.shape != (expected_node,):
        raise ValueError("ground_capacitance must contain 2*n values")
    if params.interwinding_capacitance.shape != (n,):
        raise ValueError("interwinding_capacitance must contain n values")
    if params.dielectric_conductance.shape != (expected_node,):
        raise ValueError("dielectric_conductance must contain 2*n values")
    if np.any(params.resistance <= 0.0):
        raise ValueError("all series resistances must be positive")
    if np.any(params.longitudinal_capacitance < 0.0):
        raise ValueError("capacitances must be non-negative")
    symmetry_error = float(np.linalg.norm(params.inductance - params.inductance.T))
    eig_l = np.linalg.eigvalsh(0.5 * (params.inductance + params.inductance.T))
    if eig_l.min() <= 0.0:
        raise ValueError("inductance matrix must be positive definite")
    cmat = capacitance_matrix(params)
    eig_c = np.linalg.eigvalsh(0.5 * (cmat + cmat.T))
    if eig_c.min() < -1e-18:
        raise ValueError("capacitance matrix must be positive semidefinite")
    return {
        "inductance_symmetry_error": symmetry_error,
        "inductance_min_eigenvalue_h": float(eig_l.min()),
        "capacitance_min_eigenvalue_f": float(eig_c.min()),
    }


def solve_frequency(frequency_hz: float, params: LadderParameters) -> FrequencySolution:
    """Solve one frequency and Kron-reduce internal ladder nodes."""

    omega = 2.0 * np.pi * float(frequency_hz)
    gamma = branch_incidence(params.n)
    zmat = np.diag(params.resistance) + 1j * omega * params.inductance
    cmat = capacitance_matrix(params)
    gmat = conductance_matrix(params)

    # Yz = Gamma * inv(Z) * Gamma.T, evaluated with a linear solve.
    z_inverse_gamma_t = np.linalg.solve(zmat, gamma.T)
    ynodal = gamma @ z_inverse_gamma_t + gmat + 1j * omega * cmat

    port = np.array([0, params.n], dtype=int)
    internal = np.array([k for k in range(2 * params.n) if k not in port], dtype=int)
    ypp = ynodal[np.ix_(port, port)]
    ypi = ynodal[np.ix_(port, internal)]
    yip = ynodal[np.ix_(internal, port)]
    yii = ynodal[np.ix_(internal, internal)]
    recovery = -np.linalg.solve(yii, yip)
    yport = ypp + ypi @ recovery

    return FrequencySolution(
        nodal_admittance=ynodal,
        port_admittance=yport,
        internal_recovery=recovery,
        impedance_matrix=zmat,
        capacitance_matrix=cmat,
        conductance_matrix=gmat,
    )


def solve_sweep(frequency_hz: np.ndarray, params: LadderParameters) -> dict[str, np.ndarray]:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    port_y = np.empty((frequency_hz.size, 2, 2), dtype=complex)
    recovery = np.empty((frequency_hz.size, 2 * params.n - 2, 2), dtype=complex)
    for index, freq in enumerate(frequency_hz):
        solution = solve_frequency(float(freq), params)
        port_y[index] = solution.port_admittance
        recovery[index] = solution.internal_recovery
    return {"frequency_hz": frequency_hz, "port_admittance": port_y, "internal_recovery": recovery}


def recover_all_node_voltages(
    frequency_hz: float, params: LadderParameters, port_voltage: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover all node voltages and external port currents."""

    solution = solve_frequency(frequency_hz, params)
    port = np.array([0, params.n], dtype=int)
    internal = np.array([k for k in range(2 * params.n) if k not in port], dtype=int)
    voltage = np.zeros(2 * params.n, dtype=complex)
    voltage[port] = np.asarray(port_voltage, dtype=complex)
    voltage[internal] = solution.internal_recovery @ voltage[port]
    current = solution.nodal_admittance @ voltage
    return voltage, current[port]


def make_four_section_baseline() -> LadderParameters:
    """Construct a physically passive four-region demonstration model.

    Values are intentionally transparent pilot values, not claimed as measured
    parameters of a specific hardware transformer.
    """

    n = 4
    ratio = 4.0
    rs_primary_total = 0.18
    rs_secondary_total = rs_primary_total / ratio**2
    resistance = np.r_[
        np.full(n, rs_primary_total / n),
        np.full(n, rs_secondary_total / n),
    ]

    leakage_primary_total = 3.6e-6
    leakage_secondary_total = leakage_primary_total / ratio**2
    leakage_diag = np.r_[
        np.full(n, leakage_primary_total / n),
        np.full(n, leakage_secondary_total / n),
    ]

    # Low-rank common-flux contribution plus strictly positive leakage.
    # The sign follows the selected current references at the dotted terminals.
    lm_primary = 220e-6
    flux_vector = np.r_[np.ones(n), -np.ones(n) / ratio]
    magnetic = (lm_primary / n**2) * np.outer(flux_vector, flux_vector)
    inductance = np.diag(leakage_diag) + magnetic

    longitudinal_capacitance = np.r_[
        np.full(n, 72e-12),
        np.full(n, 42e-12),
    ]
    ground_capacitance = np.r_[
        np.array([7.0, 6.0, 5.0, 4.0]) * 1e-12,
        np.array([5.0, 4.5, 4.0, 3.5]) * 1e-12,
    ]
    interwinding_capacitance = np.full(n, 28.36e-12 / n)
    dielectric_conductance = np.zeros(2 * n)

    params = LadderParameters(
        n=n,
        turns_ratio=ratio,
        resistance=resistance,
        inductance=inductance,
        longitudinal_capacitance=longitudinal_capacitance,
        ground_capacitance=ground_capacitance,
        interwinding_capacitance=interwinding_capacitance,
        dielectric_conductance=dielectric_conductance,
    )
    validate_parameters(params)
    return params


def perturb_local_cps(
    params: LadderParameters, section: int, relative_change: float
) -> LadderParameters:
    cps = params.interwinding_capacitance.copy()
    cps[section] *= 1.0 + relative_change
    return params.copy_with(interwinding_capacitance=cps)

