"""Carregadores para dados reais (ex.: exportações CSV do PhysioNet/MIMIC/BIDMC).

O formato interno de sinais vitais é um DataFrame com `timestamp` e colunas
`heart_rate, spo2, systolic_bp, diastolic_bp, temperature, resp_rate`
(qualquer subconjunto). Use `column_map` para renomear colunas do dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

VITAL_COLUMNS = ["heart_rate", "spo2", "systolic_bp", "diastolic_bp", "temperature", "resp_rate"]


def load_vitals_csv(path: str | Path, timestamp_col: str, column_map: dict[str, str] | None = None,
                    resample: str | None = None) -> pd.DataFrame:
    """Ex.: load_vitals_csv("bidmc_01_Numerics.csv", "Time [s]",
    {"HR": "heart_rate", "SpO2": "spo2", "RESP": "resp_rate"}).

    Se `timestamp_col` for numérico (segundos), é convertido para datas a partir de uma origem fictícia.
    `resample` (ex.: "1min") reamostra pela média, útil para séries de alta frequência.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={**(column_map or {}), timestamp_col: "timestamp"})
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df["timestamp"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    keep = ["timestamp"] + [c for c in VITAL_COLUMNS if c in df.columns]
    df = df[keep].sort_values("timestamp")
    if resample:
        df = df.set_index("timestamp").resample(resample).mean().dropna(how="all").reset_index()
    return df.reset_index(drop=True)
