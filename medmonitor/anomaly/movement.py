"""Anomalias em padrões de movimentação do paciente durante a internação.

Entrada: DataFrame com `timestamp` e `activity` (0 = imóvel; ex.: magnitude de
acelerômetro/pulseira ou contagem de movimento por amostra), amostragem regular.

Padrões detectados:
- possível queda: pico de atividade seguido de imobilidade;
- imobilidade prolongada em horário diurno (risco de lesão por pressão);
- agitação noturna acima da linha de base.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..alerts import Alert, Severity
from .common import group_events, rolling_robust_z


def detect_movement_anomalies(
    df: pd.DataFrame,
    patient_id: str,
    sample_minutes: float = 1.0,
    spike_z: float = 8.0,
    min_peak_ratio: float = 3.0,
    post_fall_minutes: float = 10.0,
    inactivity_threshold: float = 0.05,
    max_inactive_minutes: float = 120.0,
    day_hours: tuple[int, int] = (8, 20),
    night_to_day_ratio: float = 0.8,
) -> list[Alert]:
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    act = data["activity"].astype(float).to_numpy()
    alerts: list[Alert] = []
    if len(act) < 20:
        return alerts

    # 1) possível queda: pico em relação à linha de base LOCAL (a atividade tem ciclo dia/noite,
    #    então uma referência global seria bimodal) E bem acima do nível ativo típico.
    win = max(20, int(60 / sample_minutes))
    z = rolling_robust_z(pd.Series(act), window=win, min_periods=max(10, win // 3)).fillna(0).to_numpy()
    after_n = max(1, int(post_fall_minutes / sample_minutes))
    typical_active = np.percentile(act, 75)
    spike = (z > spike_z) & (act > min_peak_ratio * typical_active)
    for start, end in group_events(spike, max_gap=1):
        after = act[end + 1 : end + 1 + after_n]
        if len(after) >= max(3, after_n // 2) and after.mean() < 0.25 * max(typical_active, 1e-6):
            ts = data["timestamp"].iloc[start]
            alerts.append(
                Alert(patient_id, "movement", "possible_fall", Severity.CRITICAL,
                      f"pico de movimento seguido de imobilidade (possível queda) às {ts:%d/%m %H:%M}",
                      timestamp=ts.isoformat(),
                      details={"peak": float(act[start : end + 1].max()), "mean_after": float(after.mean())})
            )

    # 2) imobilidade prolongada durante o dia
    hours = data["timestamp"].dt.hour.to_numpy()
    daytime = (hours >= day_hours[0]) & (hours < day_hours[1])
    inactive = (act < inactivity_threshold) & daytime
    for start, end in group_events(inactive, max_gap=0):
        minutes = (end - start + 1) * sample_minutes
        if minutes >= max_inactive_minutes:
            ts = data["timestamp"].iloc[start]
            alerts.append(
                Alert(patient_id, "movement", "prolonged_inactivity", Severity.WARNING,
                      f"imobilidade de {minutes:.0f} min durante o dia a partir de {ts:%d/%m %H:%M}",
                      timestamp=ts.isoformat(), details={"minutes": float(minutes)})
            )

    # 3) agitação noturna
    night = (hours >= 0) & (hours < 5)
    if night.sum() >= 10 and (~night).sum() >= 10:
        day_mean = float(act[~night].mean())
        night_mean = float(act[night].mean())
        if day_mean > 0 and night_mean >= night_to_day_ratio * day_mean:
            ts = data["timestamp"].iloc[np.flatnonzero(night)[0]]
            alerts.append(
                Alert(patient_id, "movement", "night_agitation", Severity.WARNING,
                      f"atividade noturna ({night_mean:.2f}) elevada em relação ao restante do dia ({day_mean:.2f})",
                      timestamp=ts.isoformat(), details={"night_mean": night_mean, "day_mean": day_mean})
            )
    return alerts
