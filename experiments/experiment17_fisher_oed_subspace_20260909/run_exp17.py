"""EXP-017: Fisher/OED and identifiable parameter subspaces for EXP-016.

This is a synthetic, local-identifiability experiment.  The forward model is
the exact EXP-015 matrix/Kron solver.  It does not use measurements, FEM, or a
DAB switching model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
EXPERIMENTS = ROOT.parent
EXP16 = EXPERIMENTS / "experiment16_falcon_hft_gnn_20260909"
sys.path.insert(0, str(EXP16))
from run_exp16 import PARAM_NAMES, make_template, response


PORT_NAMES = ["Y11", "Y22", "Y12"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_rows(freq_ids, port_ids, nfreq):
    """Rows of flattened [frequency, six response channels]."""
    rows = []
    for fi in freq_ids:
        for block in (0, 1):  # log magnitude, phase
            for pi in port_ids:
                rows.append(fi * 6 + block * 3 + pi)
    return torch.tensor(rows, dtype=torch.long)


def jacobian_at(template, z, freqs):
    return torch.autograd.functional.jacobian(
        lambda q: response(template, q, freqs).reshape(-1), z,
        vectorize=True,
    )


def fisher_metrics(j):
    sv = torch.linalg.svdvals(j)
    tol = sv.max() * max(j.shape) * torch.finfo(j.dtype).eps
    rank = int((sv > tol).sum().item())
    cond = float((sv.max() / sv.min().clamp_min(1e-30)).item())
    fim_eigs = torch.linalg.eigvalsh(j.T @ j)
    return {
        "rank": rank,
        "singular_values": sv.detach().cpu().tolist(),
        "minimum_singular_value": float(sv.min().item()),
        "condition_number": cond,
        "fisher_eigenvalues": fim_eigs.detach().cpu().tolist(),
    }


def normalized_jacobian(j):
    # Unit-column scaling prevents OED from selecting frequencies merely for
    # the largest response unit.  The same scaling is used in all comparisons.
    scale = torch.linalg.vector_norm(j, dim=0).clamp_min(1e-14)
    return j / scale, scale


def greedy_logdet_frequency_selection(j, k, nfreq, port_ids, ridge=1e-6):
    chosen = []
    remaining = list(range(nfreq))
    eye = torch.eye(j.shape[1], dtype=j.dtype, device=j.device)
    trace = []
    for _ in range(k):
        best = None
        for fi in remaining:
            ids = selected_rows(chosen + [fi], port_ids, nfreq).to(j.device)
            f = j[ids].T @ j[ids] + ridge * eye
            score = float(torch.linalg.slogdet(f)[1].item())
            mineig = float(torch.linalg.eigvalsh(f).min().item())
            candidate = (score, mineig, fi)
            if best is None or candidate > best:
                best = candidate
        chosen.append(best[2]); remaining.remove(best[2])
        trace.append({"step": len(chosen), "frequency_index": best[2],
                      "logdet": best[0], "minimum_eigenvalue": best[1]})
    return sorted(chosen), trace


def exhaustive_parameter_subset(j, q=4):
    import itertools
    best = None
    for cols in itertools.combinations(range(j.shape[1]), q):
        js = j[:, list(cols)]
        f = js.T @ js + 1e-9 * torch.eye(q, dtype=j.dtype, device=j.device)
        score = float(torch.linalg.slogdet(f)[1].item())
        candidate = (score, cols)
        if best is None or candidate > best:
            best = candidate
    return list(best[1]), best[0]


def solve_linear(j, residual, ridge=1e-5):
    # Tikhonov-regularized local Gauss-Newton step.
    eye = torch.eye(j.shape[1], dtype=j.dtype, device=j.device)
    return torch.linalg.solve(j.T @ j + ridge * eye, j.T @ residual)


def inversion_one(j, residual, mode, selected_cols, basis, column_scale):
    """Return log-parameter increments in the original dimensionless basis."""
    p = j.shape[1]
    if mode == "full8":
        return solve_linear(j, residual, ridge=2e-5)
    if mode == "selected4":
        d = torch.zeros(p, dtype=j.dtype, device=j.device)
        d[selected_cols] = solve_linear(j[:, selected_cols], residual, ridge=2e-5)
        return d
    if mode == "lowrank4":
        return basis @ solve_linear(j @ basis, residual, ridge=2e-5)
    raise ValueError(mode)


def monte_carlo(template, freqs, j_full, h0, freq_ids, port_ids,
                selected_cols, basis, column_scale, cases, noise_sigma, seed):
    gen = torch.Generator(device=freqs.device).manual_seed(seed)
    ids = selected_rows(freq_ids, port_ids, len(freqs)).to(freqs.device)
    j = j_full[ids]
    records = []
    modes = ("full8", "selected4", "lowrank4")
    for case in range(cases):
        # Small perturbations keep the local Fisher/Gauss-Newton experiment
        # honest.  The range is +/-8.3% in positive scale space.
        true_z = (torch.rand(8, generator=gen, dtype=torch.float64,
                             device=freqs.device) * 0.16 - 0.08)
        ht = response(template, true_z, freqs).reshape(-1)
        noise = noise_sigma * torch.randn(len(ids), generator=gen,
                                           dtype=torch.float64, device=freqs.device)
        observed = ht[ids] + noise
        residual = observed - h0[ids]
        for mode in modes:
            zhat = inversion_one(j, residual, mode, selected_cols, basis, column_scale)
            hp = response(template, zhat, freqs).reshape(-1)
            scale_error = (torch.exp(zhat - true_z) - 1).abs() * 100
            records.append({
                "case": case, "mode": mode,
                "parameter_mape_percent": float(scale_error.mean().item()),
                "observed_response_mae": float((hp[ids] - ht[ids]).abs().mean().item()),
                "full_response_mae": float((hp - ht).abs().mean().item()),
                **{f"error_{name}_percent": float(scale_error[i].item())
                   for i, name in enumerate(PARAM_NAMES)},
            })
    return records


def summarize(records):
    out = {}
    for mode in sorted({r["mode"] for r in records}):
        rr = [r for r in records if r["mode"] == mode]
        out[mode] = {}
        for key in ("parameter_mape_percent", "observed_response_mae", "full_response_mae"):
            a = np.asarray([r[key] for r in rr])
            out[mode][key] = {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
                              "p95": float(np.percentile(a, 95))}
        out[mode]["per_parameter_mean_error_percent"] = {
            name: float(np.mean([r[f"error_{name}_percent"] for r in rr]))
            for name in PARAM_NAMES
        }
    return out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1709)
    ap.add_argument("--noise", type=float, default=0.003)
    args = ap.parse_args(); seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out = ROOT / "results"; out.mkdir(parents=True, exist_ok=True)
    freqs = torch.logspace(4, 7, 12, dtype=torch.float64, device=device)
    t = make_template(8, device)
    z0 = torch.zeros(8, dtype=torch.float64, device=device, requires_grad=True)
    h0 = response(t, z0, freqs).reshape(-1).detach()
    tic = time.time()
    j_raw = jacobian_at(t, z0, freqs).detach()
    j_normalized, col_scale = normalized_jacobian(j_raw)
    # z consists of dimensionless log-relative scales.  Hence the raw column
    # norms are meaningful sensitivity information and must remain in F/OED.
    j = j_raw

    all_f = list(range(len(freqs))); all_p = [0, 1, 2]
    uniform_f = np.linspace(0, len(freqs)-1, 6).round().astype(int).tolist()
    oed_f, oed_trace = greedy_logdet_frequency_selection(j, 6, len(freqs), all_p)
    selected_cols, subset_score = exhaustive_parameter_subset(
        j[selected_rows(oed_f, all_p, len(freqs)).to(device)], 4)

    # Dominant right-singular vectors are the locally observable low-rank
    # parameter coordinates.  They are fixed at the nominal model.
    joed = j[selected_rows(oed_f, all_p, len(freqs)).to(device)]
    _, _, vh = torch.linalg.svd(joed, full_matrices=False)
    basis = vh[:4].T.contiguous()

    diagnostics = {"all_ports_all_frequencies": fisher_metrics(j)}
    for label, ports in (("Y11", [0]), ("Y22", [1]), ("Y12", [2]),
                         ("self_ports", [0, 1]), ("all_ports", all_p)):
        diagnostics[label] = fisher_metrics(j[selected_rows(all_f, ports, len(freqs)).to(device)])
    diagnostics["uniform6"] = fisher_metrics(j[selected_rows(uniform_f, all_p, len(freqs)).to(device)])
    diagnostics["oed6"] = fisher_metrics(joed)

    all_records = []
    summaries = {}
    for si, (name, fids) in enumerate((("uniform6", uniform_f), ("oed6", oed_f))):
        rec = monte_carlo(t, freqs, j, h0, fids, all_p, selected_cols, basis, col_scale,
                          args.cases, args.noise, args.seed + 1000 * si)
        for r in rec: r["frequency_design"] = name
        all_records.extend(rec); summaries[name] = summarize(rec)

    metrics = {
        "experiment": "EXP-017", "synthetic_only": True,
        "local_linearization": True, "dab_windows_used": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": args.seed, "monte_carlo_cases_per_design": args.cases,
        "noise_sigma_normalized_response": args.noise,
        "frequencies_hz": freqs.detach().cpu().tolist(),
        "uniform_frequency_indices": uniform_f,
        "uniform_frequencies_hz": freqs[uniform_f].detach().cpu().tolist(),
        "oed_frequency_indices": oed_f,
        "oed_frequencies_hz": freqs[oed_f].detach().cpu().tolist(),
        "oed_trace": oed_trace,
        "fisher_selected_parameter_indices": selected_cols,
        "fisher_selected_parameters": [PARAM_NAMES[i] for i in selected_cols],
        "parameter_subset_logdet": subset_score,
        "jacobian_column_norms": dict(zip(PARAM_NAMES, col_scale.detach().cpu().tolist())),
        "diagnostics": diagnostics,
        "monte_carlo": summaries,
        "runtime_seconds": time.time() - tic,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "monte_carlo_records.csv", all_records)
    np.savez(out / "fisher_intermediates.npz",
             frequencies_hz=freqs.detach().cpu().numpy(),
             jacobian_raw=j_raw.cpu().numpy(), jacobian_normalized=j_normalized.cpu().numpy(),
             column_scales=col_scale.cpu().numpy(), lowrank_basis=basis.cpu().numpy(),
             uniform_indices=np.asarray(uniform_f), oed_indices=np.asarray(oed_f))

    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    contrib = torch.linalg.matrix_norm(j.reshape(len(freqs), 6, 8), dim=(1, 2)).cpu().numpy()
    ax.semilogx(freqs.cpu(), contrib, "o-", label="Jacobian Frobenius norm")
    ax.scatter(freqs[oed_f].cpu(), contrib[oed_f], s=70, facecolors="none", edgecolors="C3", label="log-det OED")
    ax.scatter(freqs[uniform_f].cpu(), contrib[uniform_f], marker="x", s=55, color="C2", label="uniform")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Normalized sensitivity contribution")
    ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "frequency_selection.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, fids, style in (("all 12", all_f, "o-"), ("uniform 6", uniform_f, "s--"), ("OED 6", oed_f, "^--")):
        sv = torch.linalg.svdvals(j[selected_rows(fids, all_p, len(freqs)).to(device)]).cpu()
        ax.semilogy(range(1, 9), sv, style, label=name)
    ax.set_xlabel("Singular-value index"); ax.set_ylabel("Singular value")
    ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "fisher_spectrum_comparison.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1))
    modes = ["full8", "selected4", "lowrank4"]
    for di, design in enumerate(("uniform6", "oed6")):
        data = [[r["parameter_mape_percent"] for r in all_records
                 if r["frequency_design"] == design and r["mode"] == m] for m in modes]
        axes[di].boxplot(data, tick_labels=modes, showfliers=False)
        axes[di].set_title(design); axes[di].set_ylabel("Parameter MAPE (%)")
        axes[di].tick_params(axis="x", rotation=20); axes[di].grid(True, axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "monte_carlo_parameter_error.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(np.abs(basis.cpu().numpy()), aspect="auto", cmap="viridis")
    ax.set_yticks(range(8), PARAM_NAMES); ax.set_xticks(range(4), [f"mode {i+1}" for i in range(4)])
    fig.colorbar(im, ax=ax, label="Absolute loading"); fig.tight_layout()
    fig.savefig(out / "identifiable_lowrank_basis.png", dpi=220); plt.close(fig)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
