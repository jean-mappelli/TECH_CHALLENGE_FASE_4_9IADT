"""Características acústicas da voz e indícios de fadiga/dificuldade respiratória.

Tudo é calculado com numpy/scipy (sem modelos pesados). As regras de decisão
são HEURÍSTICAS de triagem, não diagnóstico: servem para priorizar a revisão
humana e devem ser calibradas com dados reais e validação clínica.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, medfilt


@dataclass
class VoiceFeatures:
    duration_s: float
    mean_level_db: float          # nível RMS médio das frames com fala (dBFS)
    speech_ratio: float           # fração do tempo com fala
    pause_count: int              # pausas > min_pause_s
    pauses_per_min: float
    longest_pause_s: float
    median_segment_s: float       # duração mediana dos trechos contínuos de fala
    onset_rate_hz: float          # proxy de taxa de fala (picos de energia/seg)
    pitch_median_hz: float | None
    pitch_std_hz: float | None
    level_cv: float               # variação do volume (coef. de variação)

    def to_dict(self) -> dict:
        return asdict(self)


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    # normaliza para [-1, 1] ANTES de misturar canais (a média de inteiros vira float e perderia a escala)
    if data.dtype == np.uint8:  # WAV 8-bit é sem sinal
        data = (data.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return int(sr), data


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)


def _pitch_autocorr(frame: np.ndarray, sr: int, fmin: int = 60, fmax: int = 400) -> float | None:
    frame = frame - frame.mean()
    if np.max(np.abs(frame)) < 1e-4:
        return None
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[0] <= 0 or ac[lag] / ac[0] < 0.3:  # pouco periódico = não vozeado
        return None
    return sr / lag


def extract_features(x: np.ndarray, sr: int, min_pause_s: float = 0.3) -> VoiceFeatures:
    x = np.asarray(x, dtype=np.float64)
    duration = len(x) / sr
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    rms = _frame_rms(x, frame, hop)
    ref = np.percentile(rms, 95)
    noise = np.percentile(rms, 10)
    # limiar adaptativo ao ruído de fundo (funciona para voz fraca e para voz forte)
    thr = noise + 0.2 * (ref - noise)
    speech = rms > thr if ref > 3 * noise else np.zeros_like(rms, dtype=bool)
    speech = medfilt(speech.astype(float), 5) > 0.5  # remove ruídos curtos
    hop_s = hop / sr

    # corridas de fala/silêncio -> pausas reais e "frases" (fala entre pausas reais)
    runs, run, state = [], 0, bool(speech[0])
    for s in speech:
        if bool(s) == state:
            run += 1
        else:
            runs.append((state, run * hop_s))
            state, run = bool(s), 1
    runs.append((state, run * hop_s))

    pauses, segments, cur, seen = [], [], 0.0, False
    for i, (is_speech, dur) in enumerate(runs):
        if is_speech:
            cur += dur
            seen = True
        elif not seen:
            continue  # silêncio inicial
        elif dur >= min_pause_s and i < len(runs) - 1:
            pauses.append(dur)  # pausa real no meio da fala
            segments.append(cur)
            cur, seen = 0.0, False
        else:
            cur += dur  # lacuna curta dentro da frase (ou silêncio final)
    if seen and cur > 0:
        segments.append(cur)
    real_pauses = pauses

    voiced_rms = rms[speech]
    mean_db = float(20 * np.log10(np.sqrt(np.mean(voiced_rms**2)) + 1e-12)) if voiced_rms.size else -120.0
    level_cv = float(np.std(voiced_rms) / (np.mean(voiced_rms) + 1e-12)) if voiced_rms.size else 0.0

    # taxa de "sílabas" por picos do envelope suavizado
    env = np.convolve(rms, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(env, height=0.3 * ref, distance=max(1, int(0.12 / hop_s)), prominence=0.1 * ref)
    speech_time = max(speech.sum() * hop_s, 1e-6)
    onset_rate = len(peaks) / speech_time

    # pitch em até 150 frames vozeadas (40 ms)
    pitches = []
    pframe = int(0.04 * sr)
    voiced_idx = np.flatnonzero(speech)
    if voiced_idx.size:
        for i in np.linspace(0, voiced_idx.size - 1, min(150, voiced_idx.size)).astype(int):
            start = voiced_idx[i] * hop
            if start + pframe <= len(x):
                p = _pitch_autocorr(x[start : start + pframe], sr)
                if p:
                    pitches.append(p)

    return VoiceFeatures(
        duration_s=float(duration),
        mean_level_db=mean_db,
        speech_ratio=float(speech.mean()),
        pause_count=len(real_pauses),
        pauses_per_min=float(len(real_pauses) / (duration / 60)) if duration > 0 else 0.0,
        longest_pause_s=float(max(real_pauses)) if real_pauses else 0.0,
        median_segment_s=float(np.median(segments)) if segments else 0.0,
        onset_rate_hz=float(onset_rate),
        pitch_median_hz=float(np.median(pitches)) if len(pitches) >= 5 else None,
        pitch_std_hz=float(np.std(pitches)) if len(pitches) >= 5 else None,
        level_cv=level_cv,
    )


@dataclass
class VoiceFinding:
    kind: str
    severity: str  # "WARNING" | "CRITICAL"
    message: str
    evidence: dict


def assess_voice(f: VoiceFeatures, baseline: VoiceFeatures | None = None) -> list[VoiceFinding]:
    """Regras de triagem. Se houver `baseline` do paciente, compara com ele."""
    findings: list[VoiceFinding] = []

    # Fadiga: voz fraca, fala lenta e/ou muitas pausas
    fatigue_signals = {
        "volume_baixo": f.mean_level_db < -30,
        "fala_lenta": f.onset_rate_hz < 2.5,
        "muitas_pausas": f.speech_ratio < 0.55,
    }
    if baseline:
        fatigue_signals["queda_vs_baseline"] = (
            f.mean_level_db < baseline.mean_level_db - 6 or f.onset_rate_hz < 0.7 * baseline.onset_rate_hz
        )
    n = sum(fatigue_signals.values())
    if n >= 2:
        findings.append(VoiceFinding(
            "vocal_fatigue", "WARNING",
            f"indícios de cansaço vocal ({n} sinais: {', '.join(k for k, v in fatigue_signals.items() if v)})",
            {**fatigue_signals, "mean_level_db": f.mean_level_db, "onset_rate_hz": f.onset_rate_hz,
             "speech_ratio": f.speech_ratio}))

    # Possível dificuldade respiratória: frases curtas e pausas frequentes
    if f.median_segment_s < 1.0 and f.pauses_per_min >= 12:
        findings.append(VoiceFinding(
            "breathing_difficulty_pattern", "WARNING",
            f"fala fragmentada (trechos de {f.median_segment_s:.1f}s, {f.pauses_per_min:.0f} pausas/min), "
            "padrão compatível com dificuldade respiratória",
            {"median_segment_s": f.median_segment_s, "pauses_per_min": f.pauses_per_min}))

    # Disartria (proxy): fala muito lenta com variação de volume alta
    if f.onset_rate_hz < 1.8 and f.level_cv > 0.8:
        findings.append(VoiceFinding(
            "possible_dysarthria_pattern", "WARNING",
            "fala muito lenta com volume irregular (possível alteração de articulação; requer avaliação)",
            {"onset_rate_hz": f.onset_rate_hz, "level_cv": f.level_cv}))

    if f.longest_pause_s >= 4.0:
        findings.append(VoiceFinding(
            "long_pause", "WARNING", f"pausa longa de {f.longest_pause_s:.1f}s durante a fala",
            {"longest_pause_s": f.longest_pause_s}))
    return findings
