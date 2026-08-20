"""Experiment 9: CUDA physics-anchored spectral residual learning pilot.

This experiment deliberately starts small.  A cheap richer frequency-domain
forward model creates nonuniform multiport spectral tokens.  A fixed compact
physics inverse estimates Rs, Ls and Cps.  Neural models predict only the
remaining log-parameter residuals for Ls and Cps.

The dataset contains 512 independent physical states and four measurement
realisations per state.  Splits are made by physical state, never by repeat.
The shifted test split contains nuisance ranges not seen during training.

This remains synthetic model-level evidence.  It is not SPICE, FEM or hardware
validation and must not be reported as hardware accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import qmc
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 20260716
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiment9_ai_residual_pilot_20260716"
DATA_DIR = OUTPUT / "data"
MODEL_DIR = OUTPUT / "models"
FIGURE_DIR = OUTPUT / "figures"
NOTE_DIR = OUTPUT / "notes"


@dataclass
class Config:
    base_train: int = 320
    base_val: int = 64
    base_test_interp: int = 64
    base_test_shift: int = 64
    repeats: int = 4
    n_frequency: int = 64
    f_low_hz: float = 8.0e4
    f_high_hz: float = 1.0e7
    batch_size: int = 128
    max_epochs: int = 160
    patience: int = 28
    learning_rate: float = 2.0e-3
    weight_decay: float = 2.0e-4
    lambda_physics: float = 0.01
    lambda_residual: float = 2.0e-4
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def complex_channel(freq: np.ndarray, gain: float, fc_hz: float, delay_s: float) -> np.ndarray:
    return gain * np.exp(-1j * 2.0 * np.pi * freq * delay_s) / (1.0 + 1j * freq / fc_hz)


def richer_y(freq: np.ndarray, p: dict) -> np.ndarray:
    freq = np.asarray(freq, dtype=float)
    s = 1j * 2.0 * np.pi * freq
    ratio = np.maximum(freq / 1.0e6, 1.0e-12)
    rs_f = p["rs"] * (1.0 + p["skin"] * np.sqrt(ratio))
    ls_f = p["ls"] * np.clip(1.0 - p["dispersion"] * np.log1p(ratio), 0.70, 1.10)
    yl = 1.0 / (rs_f + s * ls_f)
    ym = 1.0 / p["rm"] + 1.0 / (s * p["lm"])
    ycross = s * p["cps"] + 2.0 * np.pi * freq * p["cps"] * p["tan_delta"]
    if p["mode_enabled"]:
        zmode = p["mode_r"] + s * p["mode_l"] + 1.0 / (s * p["mode_c"])
        ycross = ycross + 1.0 / zmode

    y = np.empty((freq.size, 2, 2), dtype=np.complex128)
    y[:, 0, 0] = p["n"] ** 2 * yl + ym + s * (p["cp"] + p["cpg"]) + ycross
    y[:, 0, 1] = -p["n"] * yl - ycross
    y[:, 1, 0] = y[:, 0, 1]
    y[:, 1, 1] = yl + s * (p["cs"] + p["csg"]) + ycross
    return y


def measured_y(freq: np.ndarray, truth: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    hv1 = complex_channel(freq, p["v1_gain"], p["v1_fc"], p["v1_delay"])
    hv2 = complex_channel(freq, p["v2_gain"], p["v2_fc"], p["v2_delay"])
    hi1 = complex_channel(freq, p["i1_gain"], p["i1_fc"], p["i1_delay"])
    hi2 = complex_channel(freq, p["i2_gain"], p["i2_fc"], p["i2_delay"])
    hv = np.stack([hv1, hv2], axis=1)
    hi = np.stack([hi1, hi2], axis=1)
    y = truth * hi[:, :, None] / hv[:, None, :]
    rel = p["noise_rel"]
    floor = 2.0e-7
    scale = rel * np.maximum(np.abs(y), floor)
    noise = (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)) * scale / math.sqrt(2.0)
    return y + noise


def simplified_y12(freq: np.ndarray, rs: float, ls: float, cps: float, n: float = 4.0) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freq
    return -n / (rs + s * ls) - s * cps


def physical_fit(freq: np.ndarray, y12: np.ndarray) -> dict:
    x0 = np.log([0.12, 96.0e-6, 32.0e-12])
    lower = np.log([0.035, 60.0e-6, 15.0e-12])
    upper = np.log([0.45, 145.0e-6, 75.0e-12])

    def residual(x: np.ndarray) -> np.ndarray:
        rs, ls, cps = np.exp(x)
        model = simplified_y12(freq, rs, ls, cps)
        scale = np.maximum(np.abs(y12), 2.0e-4)
        err = (model - y12) / scale
        return np.concatenate([err.real, err.imag])

    result = least_squares(residual, x0=x0, bounds=(lower, upper), max_nfev=180)
    rs, ls, cps = np.exp(result.x)
    return {
        "rs": float(rs),
        "ls": float(ls),
        "cps": float(cps),
        "cost": float(np.sqrt(np.mean(residual(result.x) ** 2))),
        "success": bool(result.success),
    }


def scale(u: float, low: float, high: float) -> float:
    return low + u * (high - low)


def make_state(u: np.ndarray, shifted: bool, state_id: int) -> dict:
    if shifted:
        temp = scale(u[0], 90.0, 115.0)
        load = scale(u[1], 1000.0, 1450.0)
        skin = scale(u[9], 0.18, 0.30)
        dispersion = scale(u[10], 0.020, 0.042)
        delay_scale = 1.7
        mode_c_range = (550.0e-12, 1100.0e-12)
        noise_range = (7.0e-4, 2.0e-3)
    else:
        temp = scale(u[0], 25.0, 90.0)
        load = scale(u[1], 150.0, 1000.0)
        skin = scale(u[9], 0.0, 0.18)
        dispersion = scale(u[10], 0.0, 0.020)
        delay_scale = 1.0
        mode_c_range = (120.0e-12, 700.0e-12)
        noise_range = (2.0e-4, 1.1e-3)

    cps = scale(u[2], 22.0e-12, 52.0e-12)
    ls = scale(u[3], 78.0e-6, 116.0e-6)
    rs = 0.095 * (1.0 + 0.00393 * (temp - 25.0)) * scale(u[4], 0.94, 1.06)
    mode_enabled = bool(u[13] > 0.36)
    phase = scale(u[5], 4.0, 52.0)
    cpg = scale(u[6], 5.0e-12, 50.0e-12)
    csg = scale(u[7], 15.0e-12, 110.0e-12)
    tan_delta = scale(u[8], 0.0, 0.04)

    return {
        "state_id": state_id,
        "n": 4.0,
        "lm": scale(u[11], 42.0e-3, 52.0e-3),
        "rm": scale(u[12], 18.0e3, 35.0e3),
        "ls": ls,
        "rs": rs,
        "cp": scale(u[14], 4.0e-12, 7.0e-12),
        "cs": scale(u[15], 175.0e-12, 270.0e-12),
        "cps": cps,
        "cpg": cpg,
        "csg": csg,
        "skin": skin,
        "dispersion": dispersion,
        "tan_delta": tan_delta,
        "mode_enabled": mode_enabled,
        "mode_r": scale(u[16], 3.5e3, 22.0e3),
        "mode_l": scale(u[17], 2.0e-6, 11.0e-6),
        "mode_c": scale(u[18], *mode_c_range),
        "load_w": load,
        "temperature_c": temp,
        "phase_deg": phase,
        "v1_gain": scale(u[19], 0.996, 1.004),
        "v2_gain": scale(u[20], 0.994, 1.007),
        "i1_gain": scale(u[21], 0.993, 1.006),
        "i2_gain": scale(u[22], 0.992, 1.008),
        "v1_fc": scale(u[23], 18.0e6, 35.0e6),
        "v2_fc": scale(u[24], 12.0e6, 28.0e6),
        "i1_fc": scale(u[25], 10.0e6, 24.0e6),
        "i2_fc": scale(u[26], 9.0e6, 22.0e6),
        "v1_delay": delay_scale * scale(u[27], -1.0e-9, 2.0e-9),
        "v2_delay": delay_scale * scale(u[28], 0.0e-9, 5.0e-9),
        "i1_delay": delay_scale * scale(u[29], 1.0e-9, 7.0e-9),
        "i2_delay": delay_scale * scale(u[30], -2.0e-9, 4.0e-9),
        "noise_rel": scale(u[31], *noise_range),
    }


def tokenise(freq: np.ndarray, y: np.ndarray) -> np.ndarray:
    logf = np.log10(freq)
    logf = 2.0 * (logf - logf.min()) / (logf.max() - logf.min()) - 1.0
    features = [logf]
    for i, j in ((0, 0), (0, 1), (1, 1)):
        z = y[:, i, j]
        features.extend([np.log10(np.abs(z) + 1.0e-10), np.sin(np.angle(z)), np.cos(np.angle(z))])
    return np.stack(features, axis=1).astype(np.float32)


def generate_dataset(cfg: Config) -> dict[str, np.ndarray]:
    counts = {
        "train": cfg.base_train,
        "val": cfg.base_val,
        "test_interp": cfg.base_test_interp,
        "test_shift": cfg.base_test_shift,
    }
    freq = np.geomspace(cfg.f_low_hz, cfg.f_high_hz, cfg.n_frequency)
    all_rows: list[dict] = []
    token_rows: list[np.ndarray] = []
    global_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    y12_rows: list[np.ndarray] = []
    split_rows: list[str] = []
    base_rows: list[int] = []
    rng = np.random.default_rng(SEED + 33)
    next_state = 0

    for split, count in counts.items():
        sampler = qmc.Sobol(d=32, scramble=True, seed=SEED + next_state + (900 if split == "test_shift" else 0))
        # The grouped split sizes are intentionally unequal.  Scrambled Sobol
        # still gives a useful low-discrepancy pilot design at arbitrary n,
        # although the strict balance guarantee only holds at powers of two.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            u_states = sampler.random(count)
        for local_idx, u in enumerate(u_states):
            state_id = next_state + local_idx
            p = make_state(u, split == "test_shift", state_id)
            truth = richer_y(freq, p)
            for repeat in range(cfg.repeats):
                local_rng = np.random.default_rng(SEED + state_id * 1009 + repeat * 97)
                ym = measured_y(freq, truth, p, local_rng)
                fit = physical_fit(freq, ym[:, 0, 1])
                if not fit["success"]:
                    raise RuntimeError(f"Physical fit failed for state {state_id}")
                target = np.asarray([
                    np.log(p["ls"] / fit["ls"]),
                    np.log(p["cps"] / fit["cps"]),
                ], dtype=np.float32)
                global_feature = np.asarray([
                    np.log(fit["rs"]), np.log(fit["ls"]), np.log(fit["cps"]),
                    np.log(fit["cost"] + 1.0e-8), p["load_w"], p["temperature_c"],
                    p["phase_deg"], p["noise_rel"],
                ], dtype=np.float32)
                token_rows.append(tokenise(freq, ym))
                global_rows.append(global_feature)
                target_rows.append(target)
                y12_rows.append(np.stack([ym[:, 0, 1].real, ym[:, 0, 1].imag], axis=1).astype(np.float32))
                split_rows.append(split)
                base_rows.append(state_id)
                all_rows.append({
                    "state_id": state_id,
                    "split": split,
                    "repeat": repeat,
                    "true_ls_uh": p["ls"] * 1.0e6,
                    "true_cps_pf": p["cps"] * 1.0e12,
                    "physical_ls_uh": fit["ls"] * 1.0e6,
                    "physical_cps_pf": fit["cps"] * 1.0e12,
                    "physical_cps_error_pct": 100.0 * (fit["cps"] - p["cps"]) / p["cps"],
                    "physical_fit_cost": fit["cost"],
                    "load_w": p["load_w"],
                    "temperature_c": p["temperature_c"],
                    "phase_deg": p["phase_deg"],
                    "cpg_pf": p["cpg"] * 1.0e12,
                    "csg_pf": p["csg"] * 1.0e12,
                    "skin_coeff": p["skin"],
                    "leakage_dispersion": p["dispersion"],
                    "tan_delta": p["tan_delta"],
                    "mode_enabled": int(p["mode_enabled"]),
                    "noise_rel": p["noise_rel"],
                })
        next_state += count

    return {
        "tokens": np.stack(token_rows),
        "global": np.stack(global_rows),
        "target": np.stack(target_rows),
        "y12": np.stack(y12_rows),
        "split": np.asarray(split_rows),
        "state_id": np.asarray(base_rows, dtype=np.int32),
        "frequency": freq.astype(np.float32),
        "rows": all_rows,
    }


class SpectralDataset(Dataset):
    def __init__(self, arrays: dict, indices: np.ndarray, norm: dict):
        self.tokens = torch.from_numpy((arrays["tokens"][indices] - norm["token_mean"]) / norm["token_std"])
        self.global_x = torch.from_numpy((arrays["global"][indices] - norm["global_mean"]) / norm["global_std"])
        self.target = torch.from_numpy((arrays["target"][indices] - norm["target_mean"]) / norm["target_std"])
        self.target_raw = torch.from_numpy(arrays["target"][indices])
        self.y12 = torch.from_numpy(arrays["y12"][indices])
        self.phys_log_rs = torch.from_numpy(arrays["global"][indices, 0])
        self.phys_log_ls = torch.from_numpy(arrays["global"][indices, 1])
        self.phys_log_cps = torch.from_numpy(arrays["global"][indices, 2])

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int):
        return (
            self.tokens[index], self.global_x[index], self.target[index], self.target_raw[index],
            self.y12[index], self.phys_log_rs[index], self.phys_log_ls[index], self.phys_log_cps[index],
        )


class MLPResidual(nn.Module):
    def __init__(self, n_freq: int, token_dim: int, global_dim: int):
        super().__init__()
        inp = n_freq * token_dim + global_dim
        self.net = nn.Sequential(
            nn.Linear(inp, 256), nn.GELU(), nn.Dropout(0.08),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2),
        )

    def forward(self, tokens: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([tokens.flatten(1), global_x], dim=1))


class SpectralPerceiver(nn.Module):
    def __init__(self, token_dim: int, global_dim: int, width: int = 64, n_latent: int = 8):
        super().__init__()
        self.token_embed = nn.Sequential(nn.Linear(token_dim, width), nn.GELU(), nn.Linear(width, width))
        self.global_embed = nn.Sequential(nn.Linear(global_dim, width), nn.GELU(), nn.Linear(width, width))
        self.latents = nn.Parameter(torch.randn(n_latent, width) * 0.02)
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.05)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 2, dropout=0.05,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width * 2, 96), nn.GELU(), nn.Dropout(0.05), nn.Linear(96, 2))

    def forward(self, tokens: torch.Tensor, global_x: torch.Tensor) -> torch.Tensor:
        token = self.token_embed(tokens)
        global_e = self.global_embed(global_x)
        latent = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1) + global_e.unsqueeze(1)
        update, _ = self.cross(latent, token, token, need_weights=False)
        latent = self.encoder(latent + update)
        pooled = self.norm(latent).mean(dim=1)
        return self.head(torch.cat([pooled, global_e], dim=1))


def physics_loss(
    pred_norm: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    phys_log_rs: torch.Tensor,
    phys_log_ls: torch.Tensor,
    phys_log_cps: torch.Tensor,
    y12: torch.Tensor,
    freq: torch.Tensor,
) -> torch.Tensor:
    residual = pred_norm * target_std + target_mean
    rs = torch.exp(phys_log_rs)
    ls = torch.exp(phys_log_ls + residual[:, 0])
    cps = torch.exp(phys_log_cps + residual[:, 1])
    omega = 2.0 * torch.pi * freq.unsqueeze(0)
    denom_real = rs.unsqueeze(1)
    denom_imag = omega * ls.unsqueeze(1)
    denom_sq = denom_real.square() + denom_imag.square()
    real = -4.0 * denom_real / denom_sq
    imag = 4.0 * denom_imag / denom_sq - omega * cps.unsqueeze(1)
    model = torch.stack([real, imag], dim=-1)
    scale = torch.clamp(torch.linalg.vector_norm(y12, dim=-1), min=2.0e-4)
    return torch.mean(torch.sum((model - y12).square(), dim=-1) / scale.square())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, norm: dict) -> dict:
    model.eval()
    pred_all, raw_all, pls_all, pcps_all = [], [], [], []
    with torch.no_grad():
        for tokens, gx, _, raw, _, _, pls, pcps in loader:
            pred = model(tokens.to(device), gx.to(device)).float().cpu()
            pred_raw = pred * torch.from_numpy(norm["target_std"]) + torch.from_numpy(norm["target_mean"])
            pred_all.append(pred_raw)
            raw_all.append(raw)
            pls_all.append(pls)
            pcps_all.append(pcps)
    pred = torch.cat(pred_all).numpy()
    true_res = torch.cat(raw_all).numpy()
    phys_ls = np.exp(torch.cat(pls_all).numpy())
    phys_cps = np.exp(torch.cat(pcps_all).numpy())
    true_ls = phys_ls * np.exp(true_res[:, 0])
    true_cps = phys_cps * np.exp(true_res[:, 1])
    corrected_ls = phys_ls * np.exp(pred[:, 0])
    corrected_cps = phys_cps * np.exp(pred[:, 1])
    return {
        "physics_ls_mae_pct": float(np.mean(np.abs(phys_ls - true_ls) / true_ls) * 100.0),
        "corrected_ls_mae_pct": float(np.mean(np.abs(corrected_ls - true_ls) / true_ls) * 100.0),
        "physics_cps_mae_pct": float(np.mean(np.abs(phys_cps - true_cps) / true_cps) * 100.0),
        "corrected_cps_mae_pct": float(np.mean(np.abs(corrected_cps - true_cps) / true_cps) * 100.0),
        "true_cps": true_cps,
        "physics_cps": phys_cps,
        "corrected_cps": corrected_cps,
    }


def train_model(name: str, model: nn.Module, loaders: dict, norm: dict, cfg: Config, device: torch.device, freq: np.ndarray):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    target_mean = torch.from_numpy(norm["target_mean"]).to(device)
    target_std = torch.from_numpy(norm["target_std"]).to(device)
    freq_t = torch.from_numpy(freq).to(device)
    best_loss = float("inf")
    best_state = None
    patience_left = cfg.patience
    history = []
    started = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses = []
        for tokens, gx, target, _, y12, prs, pls, pcps in loaders["train"]:
            tokens, gx, target = tokens.to(device), gx.to(device), target.to(device)
            y12, prs, pls, pcps = y12.to(device), prs.to(device), pls.to(device), pcps.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                pred = model(tokens, gx)
                l_param = torch.mean((pred - target).square())
                l_phys = physics_loss(pred.float(), target_mean, target_std, prs, pls, pcps, y12, freq_t)
                raw_res = pred.float() * target_std + target_mean
                l_reg = torch.mean(raw_res.square())
                loss = l_param + cfg.lambda_physics * l_phys + cfg.lambda_residual * l_reg
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for tokens, gx, target, _, _, _, _, _ in loaders["val"]:
                pred = model(tokens.to(device), gx.to(device)).float()
                val_losses.append(float(torch.mean((pred - target.to(device)).square()).cpu()))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        if row["val_loss"] < best_loss - 1.0e-6:
            best_loss = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "normalization": norm, "config": asdict(cfg), "name": name}, MODEL_DIR / f"{name}_best.pt")
    metrics = {
        split: evaluate(model, loaders[split], device, norm)
        for split in ("val", "test_interp", "test_shift")
    }
    metrics["train"] = evaluate(model, loaders["train_eval"], device, norm)
    return model, history, metrics, time.perf_counter() - started


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    cfg = Config()
    seed_everything(SEED)
    for directory in (DATA_DIR, MODEL_DIR, FIGURE_DIR, NOTE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment but torch.cuda.is_available() is False")
    device = torch.device("cuda:0")
    dataset_path = DATA_DIR / "ai_residual_pilot_dataset.npz"
    metadata_path = DATA_DIR / "ai_residual_pilot_metadata.csv"

    generation_started = time.perf_counter()
    if args.regenerate or not dataset_path.exists():
        arrays = generate_dataset(cfg)
        np.savez_compressed(
            dataset_path,
            tokens=arrays["tokens"], global_features=arrays["global"], targets=arrays["target"],
            y12=arrays["y12"], split=arrays["split"], state_id=arrays["state_id"], frequency=arrays["frequency"],
        )
        write_csv(metadata_path, arrays["rows"])
    else:
        saved = np.load(dataset_path, allow_pickle=False)
        arrays = {
            "tokens": saved["tokens"], "global": saved["global_features"], "target": saved["targets"],
            "y12": saved["y12"], "split": saved["split"], "state_id": saved["state_id"],
            "frequency": saved["frequency"],
        }
    generation_seconds = time.perf_counter() - generation_started

    train_idx = np.flatnonzero(arrays["split"] == "train")
    norm = {
        "token_mean": arrays["tokens"][train_idx].mean(axis=(0, 1), keepdims=True).astype(np.float32),
        "token_std": (arrays["tokens"][train_idx].std(axis=(0, 1), keepdims=True) + 1.0e-6).astype(np.float32),
        "global_mean": arrays["global"][train_idx].mean(axis=0, keepdims=True).astype(np.float32),
        "global_std": (arrays["global"][train_idx].std(axis=0, keepdims=True) + 1.0e-6).astype(np.float32),
        "target_mean": arrays["target"][train_idx].mean(axis=0).astype(np.float32),
        "target_std": (arrays["target"][train_idx].std(axis=0) + 1.0e-6).astype(np.float32),
    }
    loaders = {}
    for split in ("train", "val", "test_interp", "test_shift"):
        idx = np.flatnonzero(arrays["split"] == split)
        ds = SpectralDataset(arrays, idx, norm)
        loaders[split] = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=(split == "train"), num_workers=cfg.num_workers,
            pin_memory=True, drop_last=False,
        )
    loaders["train_eval"] = DataLoader(
        SpectralDataset(arrays, train_idx, norm), batch_size=cfg.batch_size, shuffle=False, pin_memory=True,
    )

    token_dim = arrays["tokens"].shape[-1]
    global_dim = arrays["global"].shape[-1]
    models = {
        "mlp_residual": MLPResidual(cfg.n_frequency, token_dim, global_dim),
        "spectral_perceiver": SpectralPerceiver(token_dim, global_dim),
    }
    histories, all_metrics, runtimes = {}, {}, {}
    for name, model in models.items():
        trained, history, metrics, runtime = train_model(name, model, loaders, norm, cfg, device, arrays["frequency"])
        histories[name] = history
        all_metrics[name] = metrics
        runtimes[name] = runtime
        write_csv(DATA_DIR / f"{name}_training_history.csv", history)

    metric_rows = []
    for model_name, split_metrics in all_metrics.items():
        for split, m in split_metrics.items():
            metric_rows.append({
                "model": model_name,
                "split": split,
                "physics_ls_mae_pct": m["physics_ls_mae_pct"],
                "corrected_ls_mae_pct": m["corrected_ls_mae_pct"],
                "physics_cps_mae_pct": m["physics_cps_mae_pct"],
                "corrected_cps_mae_pct": m["corrected_cps_mae_pct"],
                "cps_error_reduction_pct": 100.0 * (1.0 - m["corrected_cps_mae_pct"] / max(m["physics_cps_mae_pct"], 1.0e-12)),
            })
    write_csv(DATA_DIR / "ai_residual_pilot_metrics.csv", metric_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for name, history in histories.items():
        axes[0].semilogy([r["epoch"] for r in history], [r["val_loss"] for r in history], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Validation normalized MSE", title="CUDA training convergence")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    splits = ["val", "test_interp", "test_shift"]
    x = np.arange(len(splits))
    width = 0.24
    physics = [all_metrics["spectral_perceiver"][s]["physics_cps_mae_pct"] for s in splits]
    mlp = [all_metrics["mlp_residual"][s]["corrected_cps_mae_pct"] for s in splits]
    perc = [all_metrics["spectral_perceiver"][s]["corrected_cps_mae_pct"] for s in splits]
    axes[1].bar(x - width, physics, width, label="physics only")
    axes[1].bar(x, mlp, width, label="MLP residual")
    axes[1].bar(x + width, perc, width, label="Spectral Perceiver")
    axes[1].set_xticks(x, splits)
    axes[1].set(ylabel="Cps MAE (%)", title="Grouped-state generalization")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ai_residual_pilot_training_and_metrics.png", dpi=220)
    plt.close(fig)

    best = all_metrics["spectral_perceiver"]["test_interp"]
    true_pf = best["true_cps"] * 1.0e12
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    ax.scatter(true_pf, best["physics_cps"] * 1.0e12, s=13, alpha=0.45, label="physics")
    ax.scatter(true_pf, best["corrected_cps"] * 1.0e12, s=13, alpha=0.45, label="corrected")
    lo, hi = float(true_pf.min()), float(true_pf.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1)
    ax.set(xlabel="True Cps (pF)", ylabel="Estimated Cps (pF)", title="Interpolation test: physics anchor and correction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ai_residual_pilot_cps_scatter.png", dpi=220)
    plt.close(fig)

    config = {
        "seed": SEED,
        "config": asdict(cfg),
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "capability": list(torch.cuda.get_device_capability(0)),
            "amp": "float16",
        },
        "dataset": {
            "independent_states": int(sum((cfg.base_train, cfg.base_val, cfg.base_test_interp, cfg.base_test_shift))),
            "records": int(arrays["tokens"].shape[0]),
            "frequency_tokens": cfg.n_frequency,
            "split_by_state": True,
        },
        "runtime_seconds": {"generation_or_load": generation_seconds, **runtimes},
        "boundaries": [
            "Synthetic richer frequency-domain forward model; not SPICE, FEM, or hardware truth.",
            "The network predicts log residuals around a fixed compact physics inverse.",
            "All repeats of one physical state remain in the same split.",
            "The shifted test split uses nuisance ranges outside the training domain.",
        ],
    }
    with (DATA_DIR / "ai_residual_pilot_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    def m(model: str, split: str, key: str) -> float:
        return all_metrics[model][split][key]

    note = f"""# 实验九：CUDA 物理锚定频谱残差学习先导实验

## 目的

用 512 个独立物理状态验证 AI 数据接口、分组划分、物理残差标签和 CUDA 训练闭环。网络不从零预测参数，而是修正固定三参数物理反演得到的 `Delta log(Ls)` 与 `Delta log(Cps)`。

## 数据

- 独立物理状态：512。
- 每状态测量扰动：4。
- 总记录：2048。
- 频率令牌：64 个，80 kHz--10 MHz。
- 划分：320 train / 64 val / 64 interpolation test / 64 shifted test，按物理状态分组，无重复窗口泄漏。

## CUDA

- GPU：{torch.cuda.get_device_name(0)}。
- PyTorch：{torch.__version__}，CUDA runtime：{torch.version.cuda}。
- 混合精度：float16 AMP。

## Cps MAE

| 测试集 | 纯物理 (%) | MLP 修正 (%) | Spectral Perceiver 修正 (%) |
|---|---:|---:|---:|
| validation | {m('spectral_perceiver','val','physics_cps_mae_pct'):.3f} | {m('mlp_residual','val','corrected_cps_mae_pct'):.3f} | {m('spectral_perceiver','val','corrected_cps_mae_pct'):.3f} |
| interpolation test | {m('spectral_perceiver','test_interp','physics_cps_mae_pct'):.3f} | {m('mlp_residual','test_interp','corrected_cps_mae_pct'):.3f} | {m('spectral_perceiver','test_interp','corrected_cps_mae_pct'):.3f} |
| shifted test | {m('spectral_perceiver','test_shift','physics_cps_mae_pct'):.3f} | {m('mlp_residual','test_shift','corrected_cps_mae_pct'):.3f} | {m('spectral_perceiver','test_shift','corrected_cps_mae_pct'):.3f} |

## 边界

这是便宜频域复杂前向模型上的 AI 先导实验，不是独立 SPICE、FEM 或样机精度。shifted test 只能检验预设分布偏移，不能替代跨求解器和硬件验证。后续是否扩充数据，应先看学习曲线和跨域误差，而不是盲目生成一万组同源样本。
"""
    (NOTE_DIR / "ai_residual_pilot_record.md").write_text(note, encoding="utf-8")
    print(json.dumps({"cuda": config["cuda"], "runtime_seconds": config["runtime_seconds"], "metrics": metric_rows}, indent=2))


if __name__ == "__main__":
    main()
