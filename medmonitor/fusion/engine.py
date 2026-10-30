"""Fusão multimodal: combina alertas de vitais, prescrição, movimento, áudio e vídeo
em um escore de risco por paciente e emite um alerta consolidado.

Estratégia (fusão tardia, "late fusion"):
- cada alerta contribui com peso = severidade x peso da modalidade;
- alertas de modalidades DIFERENTES dentro da mesma janela temporal ganham bônus
  de concordância (evidência convergente é mais confiável);
- o escore bruto é comprimido em [0, 100) por 100*(1-exp(-x/50)) e mapeado em
  BAIXO (<20) / MÉDIO (<45) / ALTO (<75) / CRÍTICO (>=75).
Pesos e limiares são configuráveis e devem ser calibrados com dados reais.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..alerts import Alert, Severity

DEFAULT_MODALITY_WEIGHTS = {
    "vitals": 1.0, "movement": 0.8, "audio": 0.7, "video": 0.7, "prescription": 0.6, "fusion": 0.0,
}
SEVERITY_POINTS = {Severity.INFO: 2, Severity.WARNING: 8, Severity.CRITICAL: 25}


@dataclass
class RiskAssessment:
    patient_id: str
    score: float
    level: str
    modalities: list[str]
    contributions: list[dict] = field(default_factory=list)
    window_start: str | None = None
    window_end: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _level(score: float) -> str:
    if score >= 75:
        return "CRÍTICO"
    if score >= 45:
        return "ALTO"
    if score >= 20:
        return "MÉDIO"
    return "BAIXO"


class FusionEngine:
    def __init__(self, window_minutes: float = 60, weights: dict[str, float] | None = None,
                 agreement_bonus: float = 0.35):
        self.window = pd.Timedelta(minutes=window_minutes)
        self.weights = weights or DEFAULT_MODALITY_WEIGHTS
        self.agreement_bonus = agreement_bonus

    def assess(self, patient_id: str, alerts: list[Alert]) -> RiskAssessment:
        mine = [a for a in alerts if a.patient_id == patient_id and a.modality != "fusion"]
        if not mine:
            return RiskAssessment(patient_id, 0.0, "BAIXO", [])
        rows = []
        for a in mine:
            ts = pd.Timestamp(a.timestamp)
            ts = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
            rows.append((ts, a))
        rows.sort(key=lambda r: r[0])

        # janela deslizante: escolhe a janela com maior escore
        best = (0.0, [], None, None)
        for i, (t0, _) in enumerate(rows):
            in_win = [(t, a) for t, a in rows[i:] if t - t0 <= self.window]
            raw = sum(SEVERITY_POINTS[a.severity] * self.weights.get(a.modality, 0.5) for _, a in in_win)
            n_mod = len({a.modality for _, a in in_win})
            score = raw * (1 + self.agreement_bonus * (n_mod - 1))
            if score > best[0]:
                best = (score, in_win, t0, in_win[-1][0])
        score, win, t0, t1 = best
        score = 100.0 * (1.0 - math.exp(-score / 50.0))  # saturação suave: mantém a ordem entre casos graves
        mods = sorted({a.modality for _, a in win})
        contributions = [
            {"modality": a.modality, "kind": a.kind, "severity": a.severity.name,
             "points": round(SEVERITY_POINTS[a.severity] * self.weights.get(a.modality, 0.5), 1)}
            for _, a in win
        ]
        return RiskAssessment(patient_id, round(score, 1), _level(score), mods, contributions,
                              t0.isoformat() if t0 is not None else None,
                              t1.isoformat() if t1 is not None else None)

    def consolidated_alert(self, assessment: RiskAssessment, min_level: str = "ALTO") -> Alert | None:
        order = ["BAIXO", "MÉDIO", "ALTO", "CRÍTICO"]
        if order.index(assessment.level) < order.index(min_level):
            return None
        sev = Severity.CRITICAL if assessment.level == "CRÍTICO" else Severity.WARNING
        mods = ", ".join(assessment.modalities)
        return Alert(
            assessment.patient_id, "fusion", "multimodal_risk", sev,
            f"risco {assessment.level} (escore {assessment.score:.0f}/100) com evidências em: {mods}",
            details=assessment.to_dict(),
        )
