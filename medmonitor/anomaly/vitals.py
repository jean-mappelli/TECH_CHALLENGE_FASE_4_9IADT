"""Detecção de anomalias em séries temporais de sinais vitais.

Três camadas complementares:
1. Regras clínicas (limites de atenção/crítico) - determinísticas e explicáveis.
2. Z-score robusto em janela deslizante - captura desvios em relação à linha
   de base *do próprio paciente* (ex.: FC de 70 que sobe para 105 é anômalo
   mesmo dentro da faixa "normal").
3. Distância de Mahalanobis robusta (MinCovDet) - captura combinações incomuns
   de sinais, com taxa de falsos alarmes controlada pelo quantil do qui-quadrado.

Os limites abaixo são ILUSTRATIVOS e configuráveis. Não são orientação clínica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from ..alerts import Alert, Severity
from .common import group_events, rolling_robust_z

# (atenção_baixo, atenção_alto, crítico_baixo, crítico_alto)
DEFAULT_RULES: dict[str, tuple[float | None, float | None, float | None, float | None]] = {
    "heart_rate": (50, 110, 40, 130),
    "spo2": (92, None, 88, None),
    "systolic_bp": (90, 160, 80, 180),
    "diastolic_bp": (55, 100, 45, 110),
    "temperature": (35.5, 37.8, 35.0, 39.5),
    "resp_rate": (10, 22, 8, 30),
}

LABELS_PT = {
    "heart_rate": "frequência cardíaca",
    "spo2": "saturação de O2",
    "systolic_bp": "pressão sistólica",
    "diastolic_bp": "pressão diastólica",
    "temperature": "temperatura",
    "resp_rate": "frequência respiratória",
}


def _clinical_level(vital: str, values: pd.Series, rules) -> pd.Series:
    """0 = normal, 1 = atenção, 2 = crítico."""
    lw, hw, lc, hc = rules[vital]
    level = pd.Series(0, index=values.index)
    if lw is not None:
        level[values < lw] = 1
    if hw is not None:
        level[values > hw] = 1
    if lc is not None:
        level[values < lc] = 2
    if hc is not None:
        level[values > hc] = 2
    return level


def detect_vital_anomalies(
    df: pd.DataFrame,
    patient_id: str,
    rules: dict | None = None,
    z_threshold: float = 4.0,
    window: int = 30,
    use_multivariate: bool = True,
    multivariate_p: float = 0.9995,
    min_stat_samples: int = 2,
) -> tuple[pd.DataFrame, list[Alert]]:
    """Retorna (df com colunas de flags por sinal, lista de alertas).

    `df` deve ter a coluna `timestamp` e ao menos um sinal de `rules`.
    """
    rules = rules or DEFAULT_RULES
    vitals = [v for v in rules if v in df.columns]
    if not vitals:
        raise ValueError(f"Nenhum sinal vital conhecido em {list(df.columns)}")

    out = df.copy().reset_index(drop=True)
    alerts: list[Alert] = []
    any_univariate = np.zeros(len(out), dtype=bool)

    for vital in vitals:
        series = out[vital].astype(float)
        z = rolling_robust_z(series, window=window)
        level = _clinical_level(vital, series, rules)
        stat_flag = (z.abs() > z_threshold).fillna(False).to_numpy()
        clin_flag = (level > 0).to_numpy()
        flag = stat_flag | clin_flag
        out[f"{vital}_z"] = z
        out[f"{vital}_anomaly"] = flag
        any_univariate |= flag

        for start, end in group_events(flag, max_gap=1):
            seg = series.iloc[start : end + 1]
            seg_level = int(level.iloc[start : end + 1].max())
            # desvio puramente estatístico precisa persistir (evita alarme por ruído/artefato de 1 amostra);
            # violações de faixa clínica alertam mesmo com 1 amostra.
            if seg_level == 0 and (end - start + 1) < min_stat_samples:
                continue
            peak_idx = int((seg - series.median()).abs().idxmax()) if len(seg) else start
            peak = float(series.loc[peak_idx])
            if seg_level == 2:
                sev, reason = Severity.CRITICAL, "faixa crítica"
            elif seg_level == 1:
                sev, reason = Severity.WARNING, "fora da faixa de atenção"
            else:
                sev, reason = Severity.WARNING, "desvio da linha de base do paciente"
            ts = out["timestamp"].iloc[start]
            alerts.append(
                Alert(
                    patient_id=patient_id,
                    modality="vitals",
                    kind=f"{vital}_anomaly",
                    severity=sev,
                    timestamp=pd.Timestamp(ts).isoformat(),
                    message=(
                        f"{LABELS_PT.get(vital, vital)} {reason}: pico {peak:.1f} "
                        f"({end - start + 1} amostras a partir de {pd.Timestamp(ts):%d/%m %H:%M})"
                    ),
                    details={"vital": vital, "start_index": start, "end_index": end, "peak": peak},
                )
            )

    if use_multivariate and len(out) >= 50 and len(vitals) >= 2:
        X = out[vitals].astype(float).ffill().bfill().to_numpy()
        # Distância de Mahalanobis com covariância robusta (MinCovDet). Sob normalidade, D² ~ chi²(k),
        # então o limiar `multivariate_p` controla a taxa de falsos alarmes (~1 - p por amostra).
        d2 = MinCovDet(random_state=42).fit(X).mahalanobis(X)
        pred = d2 > chi2.ppf(multivariate_p, df=len(vitals))
        out["multivariate_d2"] = d2
        out["multivariate_anomaly"] = pred
        # só alerta padrões multivariados que os detectores univariados não pegaram
        for start, end in group_events(pred & ~any_univariate, max_gap=1):
            if (end - start + 1) < min_stat_samples:
                continue
            ts = out["timestamp"].iloc[start]
            alerts.append(
                Alert(
                    patient_id=patient_id,
                    modality="vitals",
                    kind="multivariate_pattern",
                    severity=Severity.WARNING,
                    timestamp=pd.Timestamp(ts).isoformat(),
                    message=f"combinação incomum de sinais vitais em {pd.Timestamp(ts):%d/%m %H:%M}",
                    details={"start_index": start, "end_index": end},
                )
            )
    else:
        out["multivariate_anomaly"] = False

    return out, alerts
