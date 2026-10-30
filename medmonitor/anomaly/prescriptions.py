"""Detecção de alterações inesperadas na evolução de prescrições.

Entrada: DataFrame com colunas `timestamp`, `drug`, `dose_mg`, `frequency_per_day`.
Cada linha é uma ordem de prescrição (início/alteração).

Regras (todas configuráveis):
- salto de dose em relação à ordem anterior da mesma droga;
- mudança abrupta de frequência;
- dose diária total acima de um teto de referência (valores ILUSTRATIVOS);
- "rajada" de novos medicamentos em curto intervalo (fora do regime inicial da admissão,
  isto é, após as primeiras `admission_hours` horas do primeiro registro).
"""
from __future__ import annotations

import pandas as pd

from ..alerts import Alert, Severity

# Tetos diários ilustrativos para adultos (mg/dia). NÃO usar como referência clínica.
DEFAULT_DAILY_MAX_MG = {
    "paracetamol": 4000,
    "dipirona": 4000,
    "ibuprofeno": 2400,
}


def detect_prescription_anomalies(
    df: pd.DataFrame,
    patient_id: str,
    warn_ratio: float = 1.5,
    crit_ratio: float = 2.5,
    new_drug_burst: int = 2,
    burst_window_hours: float = 6,
    admission_hours: float = 24,
    daily_max_mg: dict[str, float] | None = None,
) -> list[Alert]:
    daily_max_mg = daily_max_mg if daily_max_mg is not None else DEFAULT_DAILY_MAX_MG
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["drug"] = data["drug"].str.strip().str.lower()
    data = data.sort_values("timestamp").reset_index(drop=True)
    alerts: list[Alert] = []

    for drug, group in data.groupby("drug"):
        group = group.sort_values("timestamp")
        prev = None
        for _, row in group.iterrows():
            ts = row["timestamp"].isoformat()
            daily = float(row["dose_mg"]) * float(row["frequency_per_day"])
            cap = daily_max_mg.get(drug)
            if cap is not None and daily > cap:
                alerts.append(
                    Alert(patient_id, "prescription", "daily_dose_above_reference", Severity.CRITICAL,
                          f"{drug}: dose diária {daily:.0f} mg acima do teto de referência ({cap:.0f} mg)",
                          timestamp=ts, details={"drug": drug, "daily_mg": daily, "cap_mg": cap})
                )
            if prev is not None:
                prev_daily = float(prev["dose_mg"]) * float(prev["frequency_per_day"])
                ratio = daily / prev_daily if prev_daily > 0 else float("inf")
                dose_ratio = float(row["dose_mg"]) / float(prev["dose_mg"]) if prev["dose_mg"] else float("inf")
                for label, r in (("dose diária", ratio), ("dose unitária", dose_ratio)):
                    big = max(r, 1 / r) if r not in (0, float("inf")) else float("inf")
                    if big >= crit_ratio:
                        sev = Severity.CRITICAL
                    elif big >= warn_ratio:
                        sev = Severity.WARNING
                    else:
                        continue
                    direction = "aumento" if r > 1 else "redução"
                    alerts.append(
                        Alert(patient_id, "prescription", "unexpected_dose_change", sev,
                              f"{drug}: {direction} de {label} de {prev_daily:.0f} para {daily:.0f} mg/dia "
                              f"(x{big:.1f})" if label == "dose diária" else
                              f"{drug}: {direction} de {label} de {prev['dose_mg']:.0f} para {row['dose_mg']:.0f} mg (x{big:.1f})",
                              timestamp=ts,
                              details={"drug": drug, "ratio": float(big), "kind": label})
                    )
                    break  # um alerta por mudança
            prev = row

    # rajada de novos medicamentos (ignora o regime inicial da admissão)
    firsts = data.groupby("drug")["timestamp"].min().sort_values()
    admission_end = data["timestamp"].min() + pd.Timedelta(hours=admission_hours)
    times = [(d, t) for d, t in firsts.items() if t > admission_end]
    window = pd.Timedelta(hours=burst_window_hours)
    i = 0
    reported_until = -1
    while i < len(times):
        j = i
        while j + 1 < len(times) and times[j + 1][1] - times[i][1] <= window:
            j += 1
        count = j - i + 1
        if count >= new_drug_burst and i > reported_until:
            drugs = [t[0] for t in times[i : j + 1]]
            alerts.append(
                Alert(patient_id, "prescription", "new_drug_burst", Severity.WARNING,
                      f"{count} medicamentos novos em {burst_window_hours:.0f} h: {', '.join(drugs)}",
                      timestamp=times[j][1].isoformat(), details={"drugs": drugs})
            )
            reported_until = j
            i = j + 1
        else:
            i += 1
    return alerts
