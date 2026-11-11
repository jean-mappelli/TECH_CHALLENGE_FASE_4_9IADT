from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from ..anomaly.common import group_events
from .findings import Finding

MIN_CONF = 0.3


#repetições


def _dominant_joint(angles: pd.DataFrame, min_rom: float) -> str | None:
    best, best_rom = None, 0.0
    for col in angles.columns:
        x = angles[col].dropna()
        if len(x) < 10:
            continue
        rom = float(np.percentile(x, 95) - np.percentile(x, 5))
        if rom > best_rom:
            best, best_rom = col, rom
    return best if best_rom >= min_rom else None


def _half_cycles(x: np.ndarray, fps: float, prominence: float) -> list[tuple[int, int, float]]:
    """Meio-ciclos (i_inicio, i_fim, amplitude) entre extremos alternados do sinal já suavizado."""
    dist = max(1, int(0.3 * fps))
    peaks, _ = find_peaks(x, prominence=prominence, distance=dist)
    valleys, _ = find_peaks(-x, prominence=prominence, distance=dist)
    ext = sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in valleys])
    clean: list[tuple[int, int]] = []
    for idx, typ in ext:  # força alternância pico/vale (mantém o mais extremo)
        if clean and clean[-1][1] == typ:
            if (x[idx] - x[clean[-1][0]]) * typ > 0:
                clean[-1] = (idx, typ)
            continue
        clean.append((idx, typ))
    return [(a[0], b[0], float(abs(x[a[0]] - x[b[0]]))) for a, b in zip(clean, clean[1:])]


def analyze_repetitions(angles: pd.DataFrame, fps: float, incomplete_ratio: float = 0.7,
                        min_rom: float = 25.0, min_half_cycles: int = 4) -> tuple[list[Finding], dict]:
    """Retorna (achados de repetição incompleta, resumo da sessão)."""
    summary: dict = {"joint": None, "half_cycles": 0, "repetitions": 0}
    if angles is None or angles.empty or fps <= 0:
        return [], summary
    filled = angles.interpolate(limit=int(fps), limit_direction="both")
    joint = _dominant_joint(filled, min_rom)
    if joint is None:
        return [], summary
    x = filled[joint].ffill().bfill().to_numpy(dtype=float)
    x = median_filter(x, size=max(3, int(0.5 * fps) | 1), mode="nearest")  # remove jitter/picos de 1 quadro
    rom = float(np.percentile(x, 95) - np.percentile(x, 5))
    cycles = _half_cycles(x, fps, prominence=0.12 * rom)
    summary.update(joint=joint, half_cycles=len(cycles), repetitions=len(cycles) // 2)
    if len(cycles) < min_half_cycles:
        return [], summary

    amps = np.array([c[2] for c in cycles])
    ref = float(np.percentile(amps, 75))
    summary.update(typical_amplitude_deg=round(ref, 1), min_amplitude_deg=round(float(amps.min()), 1))
    findings: list[Finding] = []
    if ref < min_rom:
        return findings, summary
    flagged = [(n, c) for n, c in enumerate(cycles, start=1) if c[2] < incomplete_ratio * ref]
    groups: list[list[tuple[int, tuple[int, int, float]]]] = []
    for item in flagged:                      # descida+subida da mesma repetição viram um único achado
        if groups and item[0] - groups[-1][-1][0] == 1:
            groups[-1].append(item)
        else:
            groups.append([item])
    for g in groups:
        i0, i1 = g[0][1][0], g[-1][1][1]
        amp = min(c[2] for _, c in g)
        findings.append(Finding(
            "incomplete_repetition", "WARNING", i0 / fps, i1 / fps,
            f"{joint}: repetição incompleta (amplitude {amp:.0f}° = {amp / ref:.0%} da típica de ~{ref:.0f}°)",
            {"joint": joint, "amplitude_deg": amp, "typical_amplitude_deg": ref, "half_cycle": g[0][0]}))
    return findings, summary


#equilíbrio 


def _xy(kp: dict | None, name: str) -> np.ndarray | None:
    if kp and name in kp and (len(kp[name]) < 3 or kp[name][2] >= MIN_CONF):
        return np.asarray(kp[name][:2], dtype=float)
    return None


def balance_series(frames: list[dict | None], fps: float) -> pd.DataFrame:
    """Inclinação do tronco (graus da vertical, com sinal) e das linhas de ombros/quadris (graus da horizontal)."""
    rows = []
    for kp in frames:
        ls, rs, lh, rh = (_xy(kp, n) for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
        row = {"trunk_tilt": np.nan, "shoulder_tilt": np.nan, "hip_tilt": np.nan}
        if ls is not None and rs is not None and lh is not None and rh is not None:
            trunk = (ls + rs) / 2 - (lh + rh) / 2           # y cresce para baixo: "para cima" = -y
            row["trunk_tilt"] = float(np.degrees(np.arctan2(trunk[0], -trunk[1])))
            torso = float(np.linalg.norm(trunk))
            for key, a, b in (("shoulder_tilt", ls, rs), ("hip_tilt", lh, rh)):
                d = b - a
                if torso < 1e-6 or np.linalg.norm(d) < 0.5 * torso:
                    continue        # linha curta (vista lateral): a inclinação seria só ruído
                ang = float(np.degrees(np.arctan2(d[1], d[0])))
                row[key] = ang - 180.0 if ang > 90 else ang + 180.0 if ang < -90 else ang
        rows.append(row)
    df = pd.DataFrame(rows)
    df.index = np.arange(len(df)) / fps
    df.index.name = "t"
    return df.dropna(axis=1, how="all")


_BALANCE_LABEL = {"trunk_tilt": "inclinação do tronco", "shoulder_tilt": "linha dos ombros",
                  "hip_tilt": "linha do quadril"}
_BALANCE_MIN_DEG = {"trunk_tilt": 10.0, "shoulder_tilt": 8.0, "hip_tilt": 8.0}


def analyze_balance(frames: list[dict | None], fps: float, k_scale: float = 3.0,
                    min_duration_s: float = 0.3) -> list[Finding]:
    series = balance_series(frames, fps)
    if series.empty or len(series) < int(2 * fps):
        return []
    findings: list[Finding] = []
    for col in series.columns:
        raw = series[col].interpolate(limit=int(fps), limit_direction="both")
        if raw.isna().all() or raw.notna().sum() < 10:
            continue
        x = median_filter(raw.ffill().bfill().to_numpy(dtype=float), size=max(3, int(0.3 * fps) | 1), mode="nearest")
        med = float(np.median(x))
        scale = 1.4826 * float(np.median(np.abs(x - med)))
        thr = max(_BALANCE_MIN_DEG[col], k_scale * scale)
        dev = np.abs(x - med)
        min_len = max(2, int(round(min_duration_s * fps)))
        for a, b in group_events(dev > thr, max_gap=1):
            if b - a + 1 < min_len:
                continue
            peak = float(dev[a:b + 1].max())
            findings.append(Finding(
                "balance_loss", "WARNING", a / fps, (b + 1) / fps,
                f"possível desequilíbrio: {_BALANCE_LABEL[col]} desviou {peak:.0f}° do padrão da sessão "
                f"(limite {thr:.0f}°) por {(b - a + 1) / fps:.1f}s",
                {"metric": col, "peak_deviation_deg": peak, "threshold_deg": thr, "baseline_deg": med}))
    findings.sort(key=lambda f: f.t_start)
    merged: list[Finding] = []                # tronco e ombros se movem juntos: um achado por evento
    for f in findings:
        if merged and f.t_start <= merged[-1].t_end:
            m = merged[-1]
            m.t_end = max(m.t_end, f.t_end)
            m.evidence.setdefault("metrics", [m.evidence["metric"]]).append(f.evidence["metric"])
            if f.evidence["peak_deviation_deg"] > m.evidence["peak_deviation_deg"]:
                m.message, m.evidence["peak_deviation_deg"] = f.message, f.evidence["peak_deviation_deg"]
        else:
            merged.append(f)
    return merged


def analyze_exercise(frames: list[dict | None], fps: float, angles: pd.DataFrame
                     ) -> tuple[list[Finding], dict]:
    rep_findings, summary = analyze_repetitions(angles, fps)
    return rep_findings + analyze_balance(frames, fps), summary
