"""Estrutura comum de achados da análise de vídeo."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Finding:
    kind: str
    severity: str  # "INFO" | "WARNING" | "CRITICAL"
    t_start: float
    t_end: float
    message: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
