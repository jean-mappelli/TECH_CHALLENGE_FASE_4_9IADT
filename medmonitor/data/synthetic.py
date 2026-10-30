"""Geradores de dados sintéticos (reprodutíveis) para testes e demonstração.

Todos usam semente fixa. Os dados NÃO representam pacientes reais: servem para
exercitar o pipeline ponta a ponta com anomalias de posição/instante conhecidos
(ground truth), o que também permite medir detecção nos testes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile

# --------------------------------------------------------------- vitais -----


def generate_vitals(start: str = "2026-10-09 00:00", hours: int = 24, step_min: int = 5,
                    event_index: int | None = 200, seed: int = 7) -> pd.DataFrame:
    """Sinais vitais com ritmo circadiano leve + uma deterioração clínica injetada
    (queda de SpO2, taquicardia, taquipneia, queda de pressão)."""
    rng = np.random.default_rng(seed)
    n = hours * 60 // step_min
    ts = pd.date_range(start, periods=n, freq=f"{step_min}min")
    t = np.arange(n) / n * 2 * np.pi
    df = pd.DataFrame({
        "timestamp": ts,
        "heart_rate": 72 + 4 * np.sin(t - 1.5) + rng.normal(0, 1.8, n),
        "spo2": 97.3 + rng.normal(0, 0.5, n),
        "systolic_bp": 120 + 5 * np.sin(t - 1.0) + rng.normal(0, 2.5, n),
        "diastolic_bp": 78 + 3 * np.sin(t - 1.0) + rng.normal(0, 1.8, n),
        "temperature": 36.6 + 0.2 * np.sin(t - 2.0) + rng.normal(0, 0.06, n),
        "resp_rate": 16 + rng.normal(0, 0.8, n),
    })
    if event_index is not None:
        L = 8
        ramp = np.sin(np.linspace(0, np.pi, L)) ** 0.7
        sl = slice(event_index, event_index + L - 1)  # .loc é inclusivo nos dois extremos
        df.loc[sl, "spo2"] = df.loc[sl, "spo2"].to_numpy() - 11 * ramp
        df.loc[sl, "heart_rate"] = df.loc[sl, "heart_rate"].to_numpy() + 52 * ramp
        df.loc[sl, "resp_rate"] = df.loc[sl, "resp_rate"].to_numpy() + 12 * ramp
        df.loc[sl, "systolic_bp"] = df.loc[sl, "systolic_bp"].to_numpy() - 32 * ramp
    return df


# ---------------------------------------------------------- prescrições -----


def generate_prescriptions(day: str = "2026-10-09", with_anomalies: bool = True) -> pd.DataFrame:
    d = pd.Timestamp(day)
    rows = [
        (d - pd.Timedelta(days=2) + pd.Timedelta(hours=8), "dipirona", 500, 4),
        (d - pd.Timedelta(days=2) + pd.Timedelta(hours=8), "omeprazol", 20, 1),
        (d - pd.Timedelta(days=2) + pd.Timedelta(hours=9), "enalapril", 10, 2),
        (d - pd.Timedelta(days=1) + pd.Timedelta(hours=8), "dipirona", 500, 4),   # sem mudança
        (d - pd.Timedelta(days=1) + pd.Timedelta(hours=8), "enalapril", 10, 2),   # sem mudança
        (d + pd.Timedelta(hours=8), "omeprazol", 20, 1),                          # sem mudança
    ]
    if with_anomalies:
        rows += [
            (d + pd.Timedelta(hours=15, minutes=30), "dipirona", 1000, 6),        # salto de dose
            (d + pd.Timedelta(hours=15, minutes=40), "enalapril", 40, 2),         # x4
            (d + pd.Timedelta(hours=15, minutes=45), "furosemida", 40, 2),        # rajada de novos
            (d + pd.Timedelta(hours=16, minutes=0), "morfina", 5, 4),
        ]
    return pd.DataFrame(rows, columns=["timestamp", "drug", "dose_mg", "frequency_per_day"])


# ---------------------------------------------------------- movimentação ----


def generate_movement(start: str = "2026-10-09 00:00", hours: int = 24, seed: int = 11,
                      fall_at: str | None = "2026-10-09 16:35",
                      inactivity: tuple[str, str] | None = ("2026-10-09 09:30", "2026-10-09 12:00")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=hours * 60, freq="1min")
    h = ts.hour.to_numpy()
    day = (h >= 7) & (h < 21)
    act = np.where(day, np.abs(rng.normal(0.55, 0.25, len(ts))), np.abs(rng.normal(0.03, 0.02, len(ts))))
    df = pd.DataFrame({"timestamp": ts, "activity": act})
    if inactivity:
        m = (df.timestamp >= inactivity[0]) & (df.timestamp < inactivity[1])
        df.loc[m, "activity"] = np.abs(rng.normal(0.01, 0.005, m.sum()))
    if fall_at:
        i = int(np.flatnonzero(df.timestamp >= fall_at)[0])
        df.loc[i : i + 1, "activity"] = [5.0, 4.2]
        df.loc[i + 2 : i + 30, "activity"] = np.abs(rng.normal(0.01, 0.005, 29))
    return df


# ----------------------------------------------------------------- áudio ----

SAMPLE_RATE = 16000
TRANSCRIPTS = {
    "normal": "Estou me sentindo bem, tranquilo, só um pouco de sono.",
    "fatigued": "Estou muito cansado, com fraqueza e tontura desde ontem.",
    "breathless": "Estou com falta de ar e dor no peito, não consigo respirar direito.",
}


def _syllable(duration: float, f0: float, amp: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    vib = 1 + 0.01 * np.sin(2 * np.pi * 5 * t)
    sig = sum(np.sin(2 * np.pi * f0 * k * vib * t) / k for k in range(1, 7))
    env = np.sin(np.pi * t / duration) ** 0.5
    return amp * env * sig / 2.5


def generate_voice(kind: str = "normal", seconds: float = 20.0, seed: int = 3) -> np.ndarray:
    """Sinal tipo-fala sintético (sílabas harmônicas) com 3 perfis:
    normal | fatigued (baixo volume, lento, muitas pausas) | breathless (frases curtas, pausas frequentes)."""
    rng = np.random.default_rng(seed)
    cfg = {
        "normal":     dict(amp=0.30, syl=0.20, gap=0.04, per_phrase=(16, 22), pause=0.5),
        "fatigued":   dict(amp=0.025, syl=0.40, gap=0.25, per_phrase=(3, 4), pause=1.3),
        "breathless": dict(amp=0.25, syl=0.20, gap=0.04, per_phrase=(3, 4), pause=0.7),
    }[kind]
    parts: list[np.ndarray] = []
    total = 0.0
    while total < seconds:
        n_syl = int(rng.integers(cfg["per_phrase"][0], cfg["per_phrase"][1] + 1))
        for _ in range(n_syl):
            f0 = 120 * (1 + rng.normal(0, 0.03))
            parts.append(_syllable(cfg["syl"] * (1 + rng.normal(0, 0.05)), f0, cfg["amp"], rng))
            parts.append(np.zeros(int(cfg["gap"] * SAMPLE_RATE)))
        parts.append(np.zeros(int(cfg["pause"] * SAMPLE_RATE)))
        total = sum(len(p) for p in parts) / SAMPLE_RATE
    x = np.concatenate(parts)[: int(seconds * SAMPLE_RATE)]
    return x + rng.normal(0, 0.0015, len(x))


def write_voice_sample(path: str | Path, kind: str = "normal", seconds: float = 20.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = generate_voice(kind, seconds)
    wavfile.write(str(path), SAMPLE_RATE, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    path.with_suffix(".txt").write_text(TRANSCRIPTS[kind], encoding="utf-8")  # transcrição p/ modo mock
    return path


# ----------------------------------------------------------------- vídeo ----


def synthetic_keypoints(duration_s: float = 20.0, fps: float = 10.0, freeze: tuple[float, float] | None = (8.0, 12.0),
                        jerk_at: float | None = 15.0, left_scale: float = 1.0) -> list[dict]:
    """Keypoints de flexão/extensão de cotovelos (exercício de fisioterapia) com anomalias conhecidas."""
    L, frames = 60.0, []
    for i in range(int(duration_s * fps)):
        t = i / fps
        a = 90 + 60 * np.sin(2 * np.pi * t / 4)
        if freeze and freeze[0] <= t < freeze[1]:
            a = 90.0
        if jerk_at is not None and abs(t - jerk_at) < 0.5 / fps:
            a += 50
        kp = {}
        for side, sx, scale in (("left", -40, left_scale), ("right", 40, 1.0)):
            ang = 90 + (a - 90) * scale
            sh = np.array([sx, 0.0])
            el = sh + np.array([0.0, L])
            wr = el + L * np.array([np.sin(np.radians(ang)), -np.cos(np.radians(ang))])
            kp[f"{side}_shoulder"], kp[f"{side}_elbow"], kp[f"{side}_wrist"] = (*sh, 1.0), (*el, 1.0), (*wr, 1.0)
        frames.append(kp)
    return frames


def write_physio_video(path: str | Path, duration_s: float = 20.0, fps: int = 30,
                       freeze: tuple[float, float] | None = (8.0, 12.0),
                       jerk_at: float | None = None, seed: int = 5) -> Path:
    """Vídeo sintético (boneco de palitos fazendo elevação de braços) com congelamento.

    `jerk_at` injeta agitação rápida, mas a análise por fluxo óptico NÃO a detecta de forma confiável
    (por isso o padrão é None); movimento brusco é testado via keypoints em `synthetic_keypoints`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = 320, 240
    bg = (rng.normal(120, 18, (h, w))).clip(0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("não foi possível criar o vídeo sintético (codec MJPG indisponível)")
    for i in range(int(duration_s * fps)):
        t = i / fps
        theta = 80 + 70 * np.sin(2 * np.pi * t / 4)  # elevação dos braços (graus)
        if freeze and freeze[0] <= t < freeze[1]:
            theta = 80.0
        if jerk_at is not None and jerk_at <= t < jerk_at + 0.6:
            theta = 80 + 70 * np.sin(2 * np.pi * (t * 3.5) / 1.0)  # agitação rápida
        frame = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        cv2.circle(frame, (160, 70), 18, (30, 30, 30), 3)
        cv2.line(frame, (160, 88), (160, 170), (30, 30, 30), 4)
        cv2.line(frame, (160, 170), (140, 225), (30, 30, 30), 4)
        cv2.line(frame, (160, 170), (180, 225), (30, 30, 30), 4)
        for sign in (-1, 1):
            r = np.radians(theta)
            tip = (int(160 + sign * 70 * np.sin(r)), int(100 - 70 * np.cos(r)))
            cv2.line(frame, (160, 100), tip, (20, 20, 160), 4)
        out.write(frame)
    out.release()
    return path
