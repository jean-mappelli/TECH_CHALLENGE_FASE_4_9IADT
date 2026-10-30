"""Análise postural: ângulos articulares, séries temporais e desvios de movimento.

O núcleo (`angle_series`, `analyze_pose_sequence`) é puro numpy/pandas e funciona
com keypoints de qualquer estimador. Estimadores incluídos:

- `OpenPoseDnn`  : OpenPose (modelo COCO/Caffe) via `cv2.dnn` - requer baixar os pesos.
- `MediaPipePose`: alternativa leve, sem download manual de pesos (quando disponível).

Formato de keypoints por frame: {nome: (x, y, confiança)} ou None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..anomaly.common import group_events
from .findings import Finding

JOINT_TRIPLETS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}
SYMMETRIC_PAIRS = [("left_elbow", "right_elbow"), ("left_knee", "right_knee")]


def joint_angle(a, b, c) -> float:
    """Ângulo em graus no vértice b formado por a-b-c."""
    a, b, c = (np.asarray(p[:2], dtype=float) for p in (a, b, c))
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


def angle_series(frames: list[dict | None], fps: float, min_conf: float = 0.3) -> pd.DataFrame:
    rows = []
    for kp in frames:
        row = {}
        for joint, (a, b, c) in JOINT_TRIPLETS.items():
            val = np.nan
            if kp and all(k in kp and (len(kp[k]) < 3 or kp[k][2] >= min_conf) for k in (a, b, c)):
                val = joint_angle(kp[a], kp[b], kp[c])
            row[joint] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    df.index = np.arange(len(df)) / fps
    df.index.name = "t"
    return df.dropna(axis=1, how="all")


def analyze_pose_sequence(
    frames: list[dict | None],
    fps: float,
    window_s: float = 3.0,
    freeze_ratio: float = 0.3,
    jerk_ratio: float = 3.0,
    jerk_min_speed: float = 200.0,
    asymmetry_threshold: float = 0.65,
) -> tuple[list[Finding], pd.DataFrame]:
    """Detecta (1) congelamento/redução de amplitude, (2) movimentos bruscos
    e (3) assimetria entre lados. Retorna (achados, série de ângulos)."""
    angles = angle_series(frames, fps)
    findings: list[Finding] = []
    if angles.empty or len(angles) < int(2 * window_s * fps):
        return findings, angles

    raw = angles.interpolate(limit=3, limit_direction="both")
    smooth = raw.interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).mean()
    win = max(2, int(window_s * fps))
    step = max(1, win // 2)

    for joint in smooth.columns:
        x = smooth[joint].to_numpy()

        # 1) redução de amplitude / congelamento
        starts = list(range(0, len(x) - win + 1, step))
        rom = np.array([np.ptp(x[s : s + win]) for s in starts])
        ref = np.percentile(rom, 75)
        if ref >= 15:  # só faz sentido se o paciente se movimenta de fato
            low = rom < freeze_ratio * ref
            for a, b in group_events(low, max_gap=0):
                t0, t1 = starts[a] / fps, (starts[b] + win) / fps
                findings.append(Finding(
                    "movement_freeze", "WARNING", t0, t1,
                    f"{joint}: amplitude de movimento muito reduzida ({rom[a:b+1].mean():.0f}° vs ~{ref:.0f}° esperado)",
                    {"joint": joint, "rom_deg": float(rom[a:b + 1].mean()), "reference_rom_deg": float(ref)}))

        # 2) movimentos bruscos: velocidade angular muito acima da faixa típica DESTE paciente/sessão.
        #    Usa os ângulos sem suavização (a média móvel diluiria saltos de 1 quadro).
        xr = raw[joint].to_numpy()
        v = np.abs(np.diff(xr)) * fps  # graus/s
        ref_v = np.nanpercentile(v, 95)
        mask = (v > jerk_ratio * ref_v) & (v > jerk_min_speed)
        for a, b in group_events(np.nan_to_num(mask, nan=False), max_gap=1):
            findings.append(Finding(
                "abrupt_movement", "WARNING", a / fps, (b + 2) / fps,
                f"{joint}: movimento brusco (velocidade angular de pico {v[a:b+1].max():.0f}°/s vs ~{ref_v:.0f}°/s típico)",
                {"joint": joint, "peak_speed_deg_s": float(v[a:b + 1].max()), "typical_speed_deg_s": float(ref_v)}))

    # 3) assimetria
    for left, right in SYMMETRIC_PAIRS:
        if left in smooth and right in smooth:
            rl = np.percentile(smooth[left], 95) - np.percentile(smooth[left], 5)
            rr = np.percentile(smooth[right], 95) - np.percentile(smooth[right], 5)
            hi, lo = max(rl, rr), min(rl, rr)
            if hi >= 15 and lo / hi < asymmetry_threshold:
                weaker = left if rl < rr else right
                findings.append(Finding(
                    "asymmetry", "WARNING", 0.0, len(x) / fps,
                    f"assimetria de amplitude entre {left} e {right}: {weaker} com {lo / hi:.0%} da amplitude do lado oposto",
                    {"left_rom_deg": float(rl), "right_rom_deg": float(rr)}))
    return findings, angles


# ------------------------------------------------------------ estimadores ---


class PoseEstimator(Protocol):
    def estimate(self, frame_bgr: np.ndarray) -> dict | None: ...


class MediaPipePose:
    """Estimador MediaPipe (API 'solutions'). Levanta RuntimeError se indisponível."""

    _IDX = {
        "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
    }

    def __init__(self, model_path: str | Path | None = None, model_complexity: int = 1):
        """Versões antigas do MediaPipe têm a API `solutions` (sem arquivo de modelo). Versões
        recentes removeram-na: use a API Tasks com `model_path` apontando para um `pose_landmarker_*.task`
        (baixado do site do MediaPipe). Ambos os caminhos NÃO foram testados com vídeo real neste projeto."""
        self._legacy = None
        self._landmarker = None
        try:
            import mediapipe as mp

            self._mp = mp
            if hasattr(mp, "solutions"):
                self._legacy = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=model_complexity)
            elif model_path:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                opts = vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.IMAGE)
                self._landmarker = vision.PoseLandmarker.create_from_options(opts)
            else:
                raise RuntimeError("esta versão do MediaPipe exige `model_path` (arquivo .task do PoseLandmarker)")
        except Exception as exc:  # ImportError, API removida, modelo ausente
            raise RuntimeError(f"MediaPipe Pose indisponível: {exc}") from exc

    def estimate(self, frame_bgr: np.ndarray) -> dict | None:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            res = self._legacy.process(rgb)
            lm = res.pose_landmarks.landmark if res.pose_landmarks else None
            vis = (lambda p: p.visibility)
        else:
            res = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
            lm = res.pose_landmarks[0] if res.pose_landmarks else None
            vis = (lambda p: p.visibility if p.visibility is not None else 1.0)
        if not lm:
            return None
        return {name: (lm[i].x * w, lm[i].y * h, vis(lm[i])) for name, i in self._IDX.items()}


class OpenPoseDnn:
    """OpenPose COCO (18 keypoints) via cv2.dnn, pessoa única (argmax dos heatmaps).

    Pesos: `pose_deploy_linevec.prototxt` e `pose_iter_440000.caffemodel`
    (distribuídos pelo projeto OpenPose; ver docs/PLANO_IMPLEMENTACAO.md).
    """

    _COCO = {"right_shoulder": 2, "right_elbow": 3, "right_wrist": 4, "left_shoulder": 5, "left_elbow": 6,
             "left_wrist": 7, "right_hip": 8, "right_knee": 9, "right_ankle": 10, "left_hip": 11,
             "left_knee": 12, "left_ankle": 13}

    def __init__(self, prototxt: str | Path, caffemodel: str | Path, input_size: int = 368, threshold: float = 0.1):
        import cv2

        for p in (prototxt, caffemodel):
            if not Path(p).exists():
                raise RuntimeError(f"arquivo do OpenPose não encontrado: {p}")
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        self._size, self._thr = input_size, threshold

    def estimate(self, frame_bgr: np.ndarray) -> dict | None:
        import cv2

        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0 / 255, (self._size, self._size), (0, 0, 0), swapRB=False, crop=False)
        self._net.setInput(blob)
        out = self._net.forward()  # (1, 57, H', W')
        oh, ow = out.shape[2:]
        kp = {}
        for name, idx in self._COCO.items():
            heat = out[0, idx]
            _, conf, _, point = cv2.minMaxLoc(heat)
            kp[name] = (point[0] * w / ow, point[1] * h / oh, float(conf)) if conf >= self._thr else (0.0, 0.0, 0.0)
        return kp
