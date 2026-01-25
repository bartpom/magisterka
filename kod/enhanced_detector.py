# enhanced_detector.py
# Zaawansowane algorytmy detekcji deepfake i AI generacji
# Wersja 2.0 - Ulepszona dla lepszej precyzji i mniejszej ilości false positives

from __future__ import annotations

import os
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


# =============================================================================
# Enums i struktury danych
# =============================================================================

class DetectionMode(Enum):
    """Tryby detekcji"""
    AI = "ai"  # Detekcja AI/Generacji (Sora, DALL-E, itp.)
    DEEPFAKE = "deepfake"  # Detekcja deepfake twarzy
    COMBINED = "combined"  # Połączona detekcja
    WATERMARK = "watermark"  # Detekcja watermarków


class ArtifactType(Enum):
    """Typy artefaktów"""
    TEXTURE = "texture"  # Artefakty tekstur
    EDGE = "edge"  # Artefakty krawędzi
    COLOR = "color"  # Anomalie kolorów
    TEMPORAL = "temporal"  # Nieprawidłowości temporalne
    COMPRESSION = "compression"  # Artefakty kompresji


@dataclass
class ArtifactScore:
    """Wynik artefaktu"""
    type: ArtifactType
    score: float  # 0-1, gdzie 1 = bardziej podejrzane
    confidence: float  # Pewność detekcji 0-1
    description: str  # Opis artefaktu
    location: Optional[Tuple[int, int, int, int]] = None  # Lokalizacja (x,y,w,h)


@dataclass
class FrameAnalysis:
    """Analiza pojedynczej klatki"""
    frame_idx: int
    timestamp: float  # Czas w sekundach
    has_face: bool
    face_bbox: Optional[Tuple[int, int, int, int]]
    artifacts: List[ArtifactScore]
    texture_metrics: Dict[str, float]
    color_metrics: Dict[str, float]
    edge_metrics: Dict[str, float]


@dataclass
class VideoAnalysis:
    """Analiza całego wideo"""
    video_path: str
    fps: float
    total_frames: int
    duration: float
    frame_analyses: List[FrameAnalysis]
    temporal_metrics: Dict[str, float]
    aggregated_scores: Dict[str, float]
    detection_flags: List[str]
    ai_probability: float
    deepfake_probability: float
    watermark_detected: bool


# =============================================================================
# Główna klasa detektora
# =============================================================================

class EnhancedDeepfakeDetector:
    """
    Zaawansowany detektor deepfake i AI generacji.

    Wykorzystuje multiple techniki:
    1. Analiza tekstur (Laplacian variance, GLCM)
    2. Analiza krawędzi (Canny, Sobel)
    3. Analiza kolorów (histogram, HSV anomalie)
    4. Analiza temporalna (spójność między klatkami)
    5. Detekcja artefaktów kompresji
    6. Pattern matching dla AI generacji
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicjalizacja detektora.

        Args:
            config_path: Ścieżka do pliku konfiguracyjnego JSON
        """
        self.config = self._load_config(config_path)
        self.face_detector = self._init_face_detector()
        self.edge_detector = self._init_edge_detector()
        self.texture_analyzer = TextureAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.ai_pattern_detector = AIPatternDetector()

        # Cache dla przetworzonych klatek
        self._frame_cache = {}

        print(f"[EnhancedDetector] Zainicjalizowano z trybem: {self.config.get('mode', 'combined')}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Ładuje konfigurację z pliku JSON lub używa domyślnej."""
        default_config = {
            "mode": "combined",
            "min_face_size": 80,
            "min_face_ratio": 0.02,  # 2% obszaru klatki
            "max_jitter": 200.0,
            "texture_threshold": 0.7,
            "edge_anomaly_threshold": 0.15,
            "color_consistency_threshold": 0.8,
            "temporal_threshold": 0.7,
            "compression_artifact_threshold": 0.6,
            "ai_pattern_threshold": 0.65,
            "frame_sampling_strategy": "uniform",  # "uniform", "keyframes", "adaptive"
            "max_frames_to_analyze": 60,
            "debug_mode": False,
            "save_artifacts": False,
            "artifact_save_path": "detection_artifacts"
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge z domyślnym
                default_config.update(user_config)
                print(f"[EnhancedDetector] Załadowano konfigurację z: {config_path}")
            except Exception as e:
                print(f"[EnhancedDetector] Błąd ładowania konfiguracji: {e}")

        return default_config

    def _init_face_detector(self):
        """Inicjalizuje detektor twarzy (Haar Cascade)."""
        if cv2 is None:
            print("[EnhancedDetector] OpenCV nie jest dostępny - pomijam detekcję twarzy")
            return None

        try:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(haar_path)
            if cascade.empty():
                print("[EnhancedDetector] Nie udało się załadować Haar Cascade")
                return None
            return cascade
        except Exception as e:
            print(f"[EnhancedDetector] Błąd inicjalizacji detektora twarzy: {e}")
            return None

    def _init_edge_detector(self):
        """Inicjalizuje detektor krawędzi."""
        # Może być rozszerzone o bardziej zaawansowane metody
        return {
            "canny_low": 50,
            "canny_high": 150,
            "sobel_kernel": 3
        }

    # =========================================================================
    # Główne metody analizy
    # =========================================================================

    def analyze_video(self, video_path: str, mode: str = "combined") -> VideoAnalysis:
        """
        Główna metoda analizy wideo.

        Args:
            video_path: Ścieżka do pliku wideo
            mode: Tryb detekcji ("ai", "deepfake", "combined", "watermark")

        Returns:
            VideoAnalysis: Kompletna analiza wideo
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Plik wideo nie istnieje: {video_path}")

        if cv2 is None:
            raise RuntimeError("OpenCV jest wymagane do analizy wideo")

        print(f"[EnhancedDetector] Rozpoczynam analizę: {video_path} (tryb: {mode})")

        # Otwórz wideo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć wideo: {video_path}")

        # Pobierz informacje o wideo
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Wybierz klatki do analizy
        frame_indices = self._select_frames_for_analysis(total_frames)

        # Analizuj klatki
        frame_analyses = []
        temporal_buffer = []  # Do analizy temporalnej

        for idx, frame_idx in enumerate(frame_indices):
            # Progress
            if idx % 10 == 0:
                print(f"[EnhancedDetector] Analizuję klatkę {idx + 1}/{len(frame_indices)}")

            # Ustaw pozycję i odczytaj klatkę
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Analizuj klatkę
            timestamp = frame_idx / fps if fps > 0 else 0
            frame_analysis = self._analyze_frame(frame, frame_idx, timestamp, mode)
            frame_analyses.append(frame_analysis)

            # Zapisz do bufora temporalnego
            temporal_buffer.append(frame)

            # Ogranicz bufor do ostatnich 5 klatek dla analizy temporalnej
            if len(temporal_buffer) > 5:
                temporal_buffer.pop(0)

        # Zamknij wideo
        cap.release()

        # Analiza temporalna
        temporal_metrics = self._analyze_temporal_consistency(temporal_buffer)

        # Agreguj wyniki
        aggregated_scores = self._aggregate_scores(frame_analyses)
        detection_flags = self._generate_detection_flags(frame_analyses, aggregated_scores)

        # Oblicz prawdopodobieństwa
        ai_probability = self._calculate_ai_probability(frame_analyses, aggregated_scores, mode)
        deepfake_probability = self._calculate_deepfake_probability(frame_analyses, aggregated_scores, mode)

        # Sprawdź watermark
        watermark_detected = self._check_for_watermarks(frame_analyses)

        # Utwórz obiekt analizy
        analysis = VideoAnalysis(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            frame_analyses=frame_analyses,
            temporal_metrics=temporal_metrics,
            aggregated_scores=aggregated_scores,
            detection_flags=detection_flags,
            ai_probability=ai_probability,
            deepfake_probability=deepfake_probability,
            watermark_detected=watermark_detected
        )

        # Zapisz artefakty jeśli wymagane
        if self.config.get("save_artifacts", False):
            self._save_artifacts(analysis)

        print(f"[EnhancedDetector] Analiza zakończona. AI: {ai_probability:.2%}, Deepfake: {deepfake_probability:.2%}")

        return analysis

    def _select_frames_for_analysis(self, total_frames: int) -> List[int]:
        """Wybierz klatki do analizy na podstawie strategii próbkowania."""
        max_frames = self.config.get("max_frames_to_analyze", 60)
        strategy = self.config.get("frame_sampling_strategy", "uniform")

        if total_frames <= max_frames:
            return list(range(total_frames))

        if strategy == "uniform":
            step = total_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            return [min(idx, total_frames - 1) for idx in indices]

        elif strategy == "keyframes":
            # Prosta heurystyka: co 10% długości wideo
            keyframe_indices = []
            for i in range(max_frames):
                idx = int((i / max_frames) * total_frames)
                keyframe_indices.append(min(idx, total_frames - 1))
            return keyframe_indices

        else:  # "adaptive" - domyślnie uniform
            step = total_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            return [min(idx, total_frames - 1) for idx in indices]

    def _analyze_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float, mode: str) -> FrameAnalysis:
        """
        Analizuje pojedynczą klatkę.

        Args:
            frame: Klatka wideo (BGR)
            frame_idx: Indeks klatki
            timestamp: Czas w sekundach
            mode: Tryb detekcji

        Returns:
            FrameAnalysis: Analiza klatki
        """
        # Konwertuj do RGB dla niektórych analiz
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame

        # Detekcja twarzy
        has_face, face_bbox = self._detect_face(frame)

        # Zbierz artefakty
        artifacts = []

        # Analiza tekstur
        texture_metrics = self.texture_analyzer.analyze(frame_rgb)
        texture_artifact = self._detect_texture_artifacts(texture_metrics)
        if texture_artifact:
            artifacts.append(texture_artifact)

        # Analiza kolorów
        color_metrics = self.color_analyzer.analyze(frame_rgb)
        color_artifact = self._detect_color_anomalies(color_metrics)
        if color_artifact:
            artifacts.append(color_artifact)

        # Analiza krawędzi
        edge_metrics = self._analyze_edges(frame)
        edge_artifact = self._detect_edge_artifacts(edge_metrics)
        if edge_artifact:
            artifacts.append(edge_artifact)

        # Analiza kompresji (tylko jeśli twarz)
        if has_face and face_bbox:
            compression_artifact = self._detect_compression_artifacts(frame, face_bbox)
            if compression_artifact:
                artifacts.append(compression_artifact)

        # Pattern matching dla AI (tylko w trybach AI i COMBINED)
        if mode in ["ai", "combined"]:
            ai_patterns = self.ai_pattern_detector.detect_patterns(frame_rgb)
            for pattern in ai_patterns:
                artifacts.append(pattern)

        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp=timestamp,
            has_face=has_face,
            face_bbox=face_bbox,
            artifacts=artifacts,
            texture_metrics=texture_metrics,
            color_metrics=color_metrics,
            edge_metrics=edge_metrics
        )

    # =========================================================================
    # Metody detekcji artefaktów
    # =========================================================================

    def _detect_face(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Wykrywa twarz w klatce."""
        if self.face_detector is None or frame.ndim != 3:
            return False, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return False, None

        # Weź największą twarz
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Sprawdź minimalny rozmiar
        min_size = self.config.get("min_face_size", 80)
        if w < min_size or h < min_size:
            return False, None

        # Sprawdź minimalny ratio obszaru
        height, width = frame.shape[:2]
        face_ratio = (w * h) / (width * height)
        min_ratio = self.config.get("min_face_ratio", 0.02)
        if face_ratio < min_ratio:
            return False, None

        return True, (int(x), int(y), int(w), int(h))

    def _analyze_edges(self, frame: np.ndarray) -> Dict[str, float]:
        """Analizuje krawędzie w klatce."""
        if frame.ndim != 3:
            return {"edge_density": 0.0, "edge_uniformity": 0.0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detekcja Canny
        edges = cv2.Canny(
            gray,
            self.edge_detector["canny_low"],
            self.edge_detector["canny_high"]
        )

        # Gęstość krawędzi
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # Jednorodność krawędzi (wariancja bloków)
        h, w = edges.shape
        block_size = 32
        blocks_h = h // block_size
        blocks_w = w // block_size

        if blocks_h > 0 and blocks_w > 0:
            block_densities = []
            for i in range(blocks_h):
                for j in range(blocks_w):
                    block = edges[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    block_density = np.sum(block > 0) / (block_size * block_size)
                    block_densities.append(block_density)

            if block_densities:
                edge_uniformity = 1.0 - np.std(block_densities)
            else:
                edge_uniformity = 0.0
        else:
            edge_uniformity = 0.0

        return {
            "edge_density": float(edge_density),
            "edge_uniformity": float(edge_uniformity)
        }

    def _detect_texture_artifacts(self, texture_metrics: Dict[str, float]) -> Optional[ArtifactScore]:
        """Wykrywa artefakty tekstur."""
        # Sprawdź wariancję Laplaciana (niska = zbyt gładka, podejrzana dla AI)
        laplacian_var = texture_metrics.get("laplacian_variance", 0)

        # Próg z konfiguracji
        texture_threshold = self.config.get("texture_threshold", 0.7)

        if laplacian_var < 100:  # Zbyt gładkie tekstury
            score = min(1.0, (100 - laplacian_var) / 100)
            confidence = 0.8 if score > texture_threshold else 0.3

            return ArtifactScore(
                type=ArtifactType.TEXTURE,
                score=score,
                confidence=confidence,
                description=f"Zbyt gładkie tekstury (Laplacian var: {laplacian_var:.1f})"
            )

        return None

    def _detect_color_anomalies(self, color_metrics: Dict[str, float]) -> Optional[ArtifactScore]:
        """Wykrywa anomalie kolorów."""
        # Sprawdź spójność kolorów między kanałami
        color_consistency = color_metrics.get("channel_correlation", 1.0)
        color_threshold = self.config.get("color_consistency_threshold", 0.8)

        if color_consistency < color_threshold:
            score = 1.0 - color_consistency
            confidence = 0.7

            return ArtifactScore(
                type=ArtifactType.COLOR,
                score=score,
                confidence=confidence,
                description=f"Niska spójność kolorów między kanałami: {color_consistency:.2f}"
            )

        return None

    def _detect_edge_artifacts(self, edge_metrics: Dict[str, float]) -> Optional[ArtifactScore]:
        """Wykrywa artefakty krawędzi."""
        # Nienaturalna gęstość krawędzi
        edge_density = edge_metrics.get("edge_density", 0)
        edge_threshold = self.config.get("edge_anomaly_threshold", 0.15)

        if edge_density > edge_threshold:
            score = min(1.0, edge_density / edge_threshold)
            confidence = 0.6

            return ArtifactScore(
                type=ArtifactType.EDGE,
                score=score,
                confidence=confidence,
                description=f"Nienaturalnie wysoka gęstość krawędzi: {edge_density:.3f}"
            )

        return None

    def _detect_compression_artifacts(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[
        ArtifactScore]:
        """Wykrywa artefakty kompresji wokół twarzy."""
        x, y, w, h = face_bbox

        # Wyciągnij obszar wokół twarzy z marginesem
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Analiza bloków DCT (uproszczona)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Rozmiar bloku DCT
        block_size = 8
        h_roi, w_roi = gray_roi.shape

        # Oblicz energię wysokich częstotliwości
        high_freq_energy = 0
        total_blocks = 0

        for i in range(0, h_roi - block_size, block_size):
            for j in range(0, w_roi - block_size, block_size):
                block = gray_roi[i:i + block_size, j:j + block_size].astype(np.float32)

                # DCT 2D
                dct_block = cv2.dct(block)

                # Energia wysokich częstotliwości (prawy dolny róg)
                high_freq = dct_block[block_size // 2:, block_size // 2:]
                high_freq_energy += np.sum(high_freq ** 2)
                total_blocks += 1

        if total_blocks > 0:
            avg_high_freq_energy = high_freq_energy / total_blocks

            # Wysoka energia HF może wskazywać na artefakty kompresji
            compression_threshold = self.config.get("compression_artifact_threshold", 0.6)
            if avg_high_freq_energy > 1000:  # Arbitralny próg
                score = min(1.0, avg_high_freq_energy / 5000)
                confidence = 0.5

                return ArtifactScore(
                    type=ArtifactType.COMPRESSION,
                    score=score,
                    confidence=confidence,
                    description=f"Wysoka energia wysokich częstotliwości: {avg_high_freq_energy:.1f}",
                    location=(x1, y1, x2 - x1, y2 - y1)
                )

        return None

    # =========================================================================
    # Analiza temporalna
    # =========================================================================

    def _analyze_temporal_consistency(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analizuje spójność temporalną między klatkami."""
        if len(frames) < 2:
            return {
                "temporal_consistency": 1.0,
                "motion_consistency": 1.0,
                "color_consistency": 1.0
            }

        temporal_scores = []
        motion_scores = []
        color_scores = []

        for i in range(1, len(frames)):
            frame1 = frames[i - 1]
            frame2 = frames[i]

            # Różnica między klatkami
            diff = cv2.absdiff(frame1, frame2)
            diff_mean = np.mean(diff)

            # Spójność temporalna (im mniejsza różnica, tym wyższa spójność)
            temporal_score = 1.0 - (diff_mean / 255.0)
            temporal_scores.append(temporal_score)

            # Spójność ruchu (analiza optycznego przepływu - uproszczona)
            if frame1.ndim == 3 and frame2.ndim == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                # Oblicz różnicę gradientów
                grad_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
                grad_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
                grad_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
                grad_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)

                grad_diff_x = np.mean(np.abs(grad_x1 - grad_x2))
                grad_diff_y = np.mean(np.abs(grad_y1 - grad_y2))

                motion_score = 1.0 - ((grad_diff_x + grad_diff_y) / (255.0 * 2))
                motion_scores.append(motion_score)

            # Spójność kolorów
            if frame1.ndim == 3 and frame2.ndim == 3:
                hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

                cv2.normalize(hist1, hist1)
                cv2.normalize(hist2, hist2)

                color_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                color_scores.append(max(0, color_correlation))  # Korelacja może być ujemna

        return {
            "temporal_consistency": float(np.mean(temporal_scores)) if temporal_scores else 1.0,
            "motion_consistency": float(np.mean(motion_scores)) if motion_scores else 1.0,
            "color_consistency": float(np.mean(color_scores)) if color_scores else 1.0
        }

    # =========================================================================
    # Agregacja i ocena
    # =========================================================================

    def _aggregate_scores(self, frame_analyses: List[FrameAnalysis]) -> Dict[str, float]:
        """Agreguje wyniki z wszystkich klatek."""
        if not frame_analyses:
            return {}

        # Zbierz wszystkie artefakty
        all_artifacts = []
        for analysis in frame_analyses:
            all_artifacts.extend(analysis.artifacts)

        # Oblicz średnie wyniki dla każdego typu artefaktu
        artifact_scores = {}
        artifact_counts = {}

        for artifact in all_artifacts:
            artifact_type = artifact.type.value
            if artifact_type not in artifact_scores:
                artifact_scores[artifact_type] = 0.0
                artifact_counts[artifact_type] = 0

            artifact_scores[artifact_type] += artifact.score * artifact.confidence
            artifact_counts[artifact_type] += 1

        # Oblicz średnie ważone
        aggregated = {}
        for artifact_type in artifact_scores:
            if artifact_counts[artifact_type] > 0:
                aggregated[f"artifact_{artifact_type}"] = (
                        artifact_scores[artifact_type] / artifact_counts[artifact_type]
                )

        # Dodaj statystyki twarzy
        face_frames = sum(1 for a in frame_analyses if a.has_face)
        face_ratio = face_frames / len(frame_analyses) if frame_analyses else 0
        aggregated["face_ratio"] = face_ratio
        aggregated["face_frames"] = face_frames

        # Dodaj średnie metryki tekstur
        texture_vars = [a.texture_metrics.get("laplacian_variance", 0) for a in frame_analyses]
        aggregated["avg_texture_variance"] = float(np.mean(texture_vars)) if texture_vars else 0

        # Dodaj średnie metryki krawędzi
        edge_densities = [a.edge_metrics.get("edge_density", 0) for a in frame_analyses]
        aggregated["avg_edge_density"] = float(np.mean(edge_densities)) if edge_densities else 0

        return aggregated

    def _generate_detection_flags(self, frame_analyses: List[FrameAnalysis],
                                  aggregated_scores: Dict[str, float]) -> List[str]:
        """Generuje flagi detekcji na podstawie wyników."""
        flags = []

        # Flagi dotyczące twarzy
        face_ratio = aggregated_scores.get("face_ratio", 0)
        if face_ratio < 0.2:  # 20%
            flags.append("LOW_FACE_RATIO")
        elif face_ratio > 0.8:  # 80%
            flags.append("HIGH_FACE_RATIO")

        # Flagi artefaktów
        texture_artifact = aggregated_scores.get("artifact_texture", 0)
        if texture_artifact > 0.7:
            flags.append("HIGH_TEXTURE_ARTIFACTS")

        edge_artifact = aggregated_scores.get("artifact_edge", 0)
        if edge_artifact > 0.6:
            flags.append("HIGH_EDGE_ARTIFACTS")

        color_artifact = aggregated_scores.get("artifact_color", 0)
        if color_artifact > 0.5:
            flags.append("COLOR_ANOMALIES")

        # Flagi tekstur
        avg_texture_var = aggregated_scores.get("avg_texture_variance", 0)
        if avg_texture_var < 50:
            flags.append("LOW_TEXTURE_VARIANCE")
        elif avg_texture_var > 300:
            flags.append("HIGH_TEXTURE_VARIANCE")

        # Flagi krawędzi
        avg_edge_density = aggregated_scores.get("avg_edge_density", 0)
        if avg_edge_density > 0.15:
            flags.append("HIGH_EDGE_DENSITY")

        return flags

    def _calculate_ai_probability(self, frame_analyses: List[FrameAnalysis],
                                  aggregated_scores: Dict[str, float], mode: str) -> float:
        """Oblicza prawdopodobieństwo, że wideo jest AI-generowane."""
        if mode not in ["ai", "combined"]:
            return 0.0

        indicators = []
        weights = []

        # 1. Niska wariancja tekstur (AI często ma zbyt gładkie tekstury)
        texture_var = aggregated_scores.get("avg_texture_variance", 0)
        if texture_var < 100:
            indicators.append(1.0 - min(1.0, texture_var / 100))
            weights.append(0.3)

        # 2. Wysoka gęstość krawędzi
        edge_density = aggregated_scores.get("avg_edge_density", 0)
        if edge_density > 0.1:
            indicators.append(min(1.0, edge_density / 0.3))
            weights.append(0.2)

        # 3. Artefakty tekstur
        texture_artifact = aggregated_scores.get("artifact_texture", 0)
        if texture_artifact > 0:
            indicators.append(texture_artifact)
            weights.append(0.25)

        # 4. Niski ratio twarzy (AI często nie ma twarzy lub ma nienaturalne)
        face_ratio = aggregated_scores.get("face_ratio", 0)
        if face_ratio < 0.3:
            indicators.append(1.0 - face_ratio / 0.3)
            weights.append(0.15)

        # 5. Artefakty krawędzi
        edge_artifact = aggregated_scores.get("artifact_edge", 0)
        if edge_artifact > 0:
            indicators.append(edge_artifact)
            weights.append(0.1)

        # Oblicz ważoną średnią
        if indicators and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(i * w for i, w in zip(indicators, weights))
                probability = weighted_sum / total_weight
                return min(1.0, probability)

        return 0.0

    def _calculate_deepfake_probability(self, frame_analyses: List[FrameAnalysis],
                                        aggregated_scores: Dict[str, float], mode: str) -> float:
        """Oblicza prawdopodobieństwo, że wideo jest deepfake."""
        if mode not in ["deepfake", "combined"]:
            return 0.0

        indicators = []
        weights = []

        # 1. Wysoki ratio twarzy (deepfake często skupia się na twarzach)
        face_ratio = aggregated_scores.get("face_ratio", 0)
        if face_ratio > 0.3:
            indicators.append(min(1.0, face_ratio))
            weights.append(0.25)

        # 2. Artefakty kompresji wokół twarzy
        compression_artifact = aggregated_scores.get("artifact_compression", 0)
        if compression_artifact > 0:
            indicators.append(compression_artifact)
            weights.append(0.3)

        # 3. Artefakty krawędzi (szczególnie wokół twarzy)
        edge_artifact = aggregated_scores.get("artifact_edge", 0)
        if edge_artifact > 0:
            indicators.append(edge_artifact)
            weights.append(0.2)

        # 4. Artefakty tekstur (nienaturalne tekstury twarzy)
        texture_artifact = aggregated_scores.get("artifact_texture", 0)
        if texture_artifact > 0:
            indicators.append(texture_artifact)
            weights.append(0.15)

        # 5. Anomalie kolorów (nierównomierne kolory twarzy)
        color_artifact = aggregated_scores.get("artifact_color", 0)
        if color_artifact > 0:
            indicators.append(color_artifact)
            weights.append(0.1)

        # Oblicz ważoną średnią
        if indicators and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(i * w for i, w in zip(indicators, weights))
                probability = weighted_sum / total_weight
                return min(1.0, probability)

        return 0.0

    def _check_for_watermarks(self, frame_analyses: List[FrameAnalysis]) -> bool:
        """Sprawdza obecność watermarków w klatkach."""
        # Prosta heurystyka: szukaj tekstu w dolnej lub górnej części klatek
        # W rzeczywistej implementacji można dodać OCR

        # Na razie zwracamy False - implementacja OCR wymaga dodatkowych bibliotek
        return False

    def _save_artifacts(self, analysis: VideoAnalysis):
        """Zapisuje artefakty detekcji do plików."""
        save_path = self.config.get("artifact_save_path", "detection_artifacts")
        os.makedirs(save_path, exist_ok=True)

        # Nazwa pliku na podstawie nazwy wideo i timestampu
        video_name = os.path.splitext(os.path.basename(analysis.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{video_name}_{timestamp}"

        # Zapisz analizę jako JSON
        analysis_dict = {
            "video_path": analysis.video_path,
            "fps": analysis.fps,
            "total_frames": analysis.total_frames,
            "duration": analysis.duration,
            "aggregated_scores": analysis.aggregated_scores,
            "detection_flags": analysis.detection_flags,
            "ai_probability": analysis.ai_probability,
            "deepfake_probability": analysis.deepfake_probability,
            "watermark_detected": analysis.watermark_detected,
            "temporal_metrics": analysis.temporal_metrics
        }

        json_path = os.path.join(save_path, f"{base_name}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)

        print(f"[EnhancedDetector] Zapisano analizę do: {json_path}")

        # Jeśli włączony tryb debug, zapisz przykładowe klatki z artefaktami
        if self.config.get("debug_mode", False) and cv2 is not None:
            # Znajdź klatki z najwyższymi wynikami artefaktów
            frame_scores = []
            for i, frame_analysis in enumerate(analysis.frame_analyses):
                if frame_analysis.artifacts:
                    max_score = max((a.score * a.confidence for a in frame_analysis.artifacts), default=0)
                    frame_scores.append((i, max_score))

            # Posortuj i weź top 3 klatki
            frame_scores.sort(key=lambda x: x[1], reverse=True)
            top_frames = frame_scores[:3]

            # Otwórz wideo i zapisz klatki
            cap = cv2.VideoCapture(analysis.video_path)
            for idx, score in top_frames:
                frame_idx = analysis.frame_analyses[idx].frame_idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Dodaj adnotacje
                    frame_with_annotations = self._annotate_frame(frame, analysis.frame_analyses[idx])

                    # Zapisz klatkę
                    frame_path = os.path.join(save_path, f"{base_name}_frame_{frame_idx}.jpg")
                    cv2.imwrite(frame_path, frame_with_annotations)

            cap.release()

    def _annotate_frame(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        """Dodaje adnotacje do klatki."""
        frame_copy = frame.copy()

        # Narysuj bounding box twarzy
        if analysis.has_face and analysis.face_bbox:
            x, y, w, h = analysis.face_bbox
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_copy, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dodaj informacje o artefaktach
        y_offset = 30
        for artifact in analysis.artifacts[:3]:  # Pokazuj tylko top 3
            text = f"{artifact.type.value}: {artifact.score:.2f}"
            cv2.putText(frame_copy, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += 20

        return frame_copy

    # =========================================================================
    # Metody pomocnicze
    # =========================================================================

    def get_summary(self, analysis: VideoAnalysis) -> Dict[str, Any]:
        """Zwraca podsumowanie analizy."""
        return {
            "video": os.path.basename(analysis.video_path),
            "duration": f"{analysis.duration:.2f}s",
            "frames_analyzed": len(analysis.frame_analyses),
            "ai_probability": f"{analysis.ai_probability:.2%}",
            "deepfake_probability": f"{analysis.deepfake_probability:.2%}",
            "watermark_detected": analysis.watermark_detected,
            "detection_flags": analysis.detection_flags,
            "face_ratio": f"{analysis.aggregated_scores.get('face_ratio', 0):.2%}",
            "key_findings": self._generate_key_findings(analysis)
        }

    def _generate_key_findings(self, analysis: VideoAnalysis) -> List[str]:
        """Generuje kluczowe wnioski z analizy."""
        findings = []

        if analysis.ai_probability > 0.7:
            findings.append("Wysokie prawdopodobieństwo AI generacji")
        elif analysis.ai_probability > 0.4:
            findings.append("Umiarkowane prawdopodobieństwo AI generacji")

        if analysis.deepfake_probability > 0.7:
            findings.append("Wysokie prawdopodobieństwo deepfake")
        elif analysis.deepfake_probability > 0.4:
            findings.append("Umiarkowane prawdopodobieństwo deepfake")

        if "LOW_FACE_RATIO" in analysis.detection_flags:
            findings.append("Niska ilość klatek z twarzą")

        if "HIGH_TEXTURE_ARTIFACTS" in analysis.detection_flags:
            findings.append("Wykryto artefakty tekstur")

        if "HIGH_EDGE_ARTIFACTS" in analysis.detection_flags:
            findings.append("Wykryto artefakty krawędzi")

        if analysis.watermark_detected:
            findings.append("Wykryto potencjalne watermarky")

        if not findings:
            findings.append("Brak wyraźnych wskazań manipulacji")

        return findings


# =============================================================================
# Klasy pomocnicze dla analizy
# =============================================================================

class TextureAnalyzer:
    """Analizator tekstur obrazu."""

    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """Analizuje tekstury obrazu."""
        if image.ndim != 3:
            return {"laplacian_variance": 0.0}

        # Konwertuj do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if cv2 is not None else image

        # Wariancja Laplaciana (miara szczegółowości)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()

        # Kontrast (RMS)
        contrast = np.std(gray)

        # Entropia (miara złożoności tekstury)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return {
            "laplacian_variance": float(laplacian_var),
            "contrast": float(contrast),
            "entropy": float(entropy)
        }


class ColorAnalyzer:
    """Analizator kolorów obrazu."""

    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """Analizuje kolory obrazu."""
        if image.ndim != 3:
            return {"channel_correlation": 1.0}

        # Korelacja między kanałami (R-G, R-B, G-B)
        r = image[:, :, 0].flatten()
        g = image[:, :, 1].flatten()
        b = image[:, :, 2].flatten()

        # Ogranicz do próbki dla wydajności
        sample_size = min(10000, len(r))
        indices = np.random.choice(len(r), sample_size, replace=False)

        r_sample = r[indices]
        g_sample = g[indices]
        b_sample = b[indices]

        # Oblicz korelacje
        rg_corr = np.corrcoef(r_sample, g_sample)[0, 1]
        rb_corr = np.corrcoef(r_sample, b_sample)[0, 1]
        gb_corr = np.corrcoef(g_sample, b_sample)[0, 1]

        # Średnia korelacja
        avg_correlation = (rg_corr + rb_corr + gb_corr) / 3

        # Nasycenie kolorów (w przestrzeni HSV)
        if cv2 is not None:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])
        else:
            saturation = 0.0

        return {
            "channel_correlation": float(avg_correlation),
            "saturation": float(saturation),
            "rg_correlation": float(rg_corr),
            "rb_correlation": float(rb_corr),
            "gb_correlation": float(gb_corr)
        }


class TemporalAnalyzer:
    """Analizator spójności temporalnej."""

    def analyze_sequence(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analizuje sekwencję klatek."""
        if len(frames) < 2:
            return {"temporal_stability": 1.0}

        # Oblicz różnice między kolejnymi klatkami
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i - 1], frames[i])
            diffs.append(np.mean(diff))

        # Stabilność temporalna (im mniejsza średnia różnica, tym wyższa stabilność)
        avg_diff = np.mean(diffs) if diffs else 0
        temporal_stability = 1.0 - min(1.0, avg_diff / 50.0)  # Normalizacja

        # Wariancja różnic (nieregularne zmiany)
        diff_variance = np.var(diffs) if len(diffs) > 1 else 0

        return {
            "temporal_stability": float(temporal_stability),
            "diff_variance": float(diff_variance),
            "avg_frame_diff": float(avg_diff)
        }


class AIPatternDetector:
    """Detektor patternów charakterystycznych dla AI generacji."""

    def __init__(self):
        # Wzorce charakterystyczne dla różnych generatorów AI
        self.patterns = {
            "sora": {
                "description": "Sora OpenAI - charakterystyczne płynne przejścia",
                "indicators": ["smooth_transitions", "consistent_lighting", "unnatural_physics"]
            },
            "dalle": {
                "description": "DALL-E - artefakty tekstur, nienaturalne szczegóły",
                "indicators": ["texture_repetition", "detail_inconsistency", "perspective_errors"]
            },
            "midjourney": {
                "description": "Midjourney - styl artystyczny, nienaturalne kolory",
                "indicators": ["artistic_style", "color_bleeding", "soft_edges"]
            }
        }

    def detect_patterns(self, image: np.ndarray) -> List[ArtifactScore]:
        """Wykrywa patterny charakterystyczne dla AI generacji."""
        artifacts = []

        # 1. Sprawdź powtarzalność tekstur (częsta w AI)
        texture_repetition = self._check_texture_repetition(image)
        if texture_repetition > 0.5:
            artifacts.append(ArtifactScore(
                type=ArtifactType.TEXTURE,
                score=texture_repetition,
                confidence=0.6,
                description=f"Powtarzalność tekstur (AI pattern): {texture_repetition:.2f}"
            ))

        # 2. Sprawdź nienaturalne krawędzie
        edge_anomalies = self._check_edge_anomalies(image)
        if edge_anomalies > 0.4:
            artifacts.append(ArtifactScore(
                type=ArtifactType.EDGE,
                score=edge_anomalies,
                confidence=0.5,
                description=f"Nienaturalne krawędzie (AI pattern): {edge_anomalies:.2f}"
            ))

        # 3. Sprawdź niespójności w szczegółach
        detail_inconsistency = self._check_detail_inconsistency(image)
        if detail_inconsistency > 0.6:
            artifacts.append(ArtifactScore(
                type=ArtifactType.TEXTURE,
                score=detail_inconsistency,
                confidence=0.7,
                description=f"Niespójność szczegółów (AI pattern): {detail_inconsistency:.2f}"
            ))

        return artifacts

    def _check_texture_repetition(self, image: np.ndarray) -> float:
        """Sprawdza powtarzalność tekstur (częsta w AI)."""
        if image.ndim != 3:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Oblicz autokorelację w małych blokach
        block_size = 32
        h, w = gray.shape

        if h < block_size * 2 or w < block_size * 2:
            return 0.0

        # Porównaj sąsiednie bloki
        similarities = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block1 = gray[i:i + block_size, j:j + block_size]

                # Porównaj z blokiem obok
                if j + block_size * 2 < w:
                    block2 = gray[i:i + block_size, j + block_size:j + block_size * 2]
                    similarity = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]
                    similarities.append(abs(similarity))

        if similarities:
            # Wysoka średnia korelacja wskazuje na powtarzalność
            avg_similarity = np.mean(similarities)
            return min(1.0, avg_similarity * 1.5)  # Skalowanie
        return 0.0

    def _check_edge_anomalies(self, image: np.ndarray) -> float:
        """Sprawdza nienaturalne krawędzie."""
        if image.ndim != 3:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detekcja krawędzi Canny
        edges = cv2.Canny(gray, 50, 150)

        # Analizuj rozkład kierunków krawędzi
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Oblicz kierunki gradientów
        magnitudes = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        angles = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

        # Analizuj histogram kierunków (AI często ma zbyt uporządkowane kierunki)
        edge_mask = edges > 0
        if np.sum(edge_mask) > 100:
            edge_angles = angles[edge_mask]

            # Podziel na przedziały 45 stopni
            bins = np.arange(-180, 181, 45)
            hist, _ = np.histogram(edge_angles, bins=bins)

            # Oblicz entropię rozkładu (niska entropia = zbyt uporządkowane)
            hist_norm = hist / hist.sum()
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            max_entropy = np.log2(len(bins) - 1)

            # Niska entropia = podejrzana
            edge_anomaly = 1.0 - (entropy / max_entropy)
            return max(0.0, min(1.0, edge_anomaly))

        return 0.0

    def _check_detail_inconsistency(self, image: np.ndarray) -> float:
        """Sprawdza niespójność szczegółów w różnych obszarach."""
        if image.ndim != 3:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Podziel obraz na regiony
        h, w = gray.shape
        regions = 4
        region_h = h // regions
        region_w = w // regions

        region_details = []

        for i in range(regions):
            for j in range(regions):
                y1 = i * region_h
                y2 = min((i + 1) * region_h, h)
                x1 = j * region_w
                x2 = min((j + 1) * region_w, w)

                region = gray[y1:y2, x1:x2]

                # Oblicz szczegółowość regionu (wariancja Laplaciana)
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                detail_level = laplacian.var()
                region_details.append(detail_level)

        # Oblicz współczynnik zmienności między regionami
        if len(region_details) > 1 and np.mean(region_details) > 0:
            cv = np.std(region_details) / np.mean(region_details)

            # Wysoka zmienność = niespójność szczegółów
            return min(1.0, cv * 2)  # Skalowanie

        return 0.0


# =============================================================================
# Funkcje pomocnicze
# =============================================================================

def analyze_video_enhanced(video_path: str, mode: str = "combined",
                           config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Uproszczona funkcja do analizy wideo.

    Args:
        video_path: Ścieżka do pliku wideo
        mode: Tryb detekcji ("ai", "deepfake", "combined")
        config_path: Ścieżka do pliku konfiguracyjnego

    Returns:
        Słownik z wynikami analizy
    """
    try:
        detector = EnhancedDeepfakeDetector(config_path)
        analysis = detector.analyze_video(video_path, mode)
        summary = detector.get_summary(analysis)

        # Dodaj szczegółowe wyniki
        summary.update({
            "detailed_scores": analysis.aggregated_scores,
            "temporal_analysis": analysis.temporal_metrics,
            "frame_count": len(analysis.frame_analyses),
            "success": True
        })

        return summary

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "video": os.path.basename(video_path) if video_path else "Unknown"
        }


def batch_analyze_videos(video_paths: List[str], mode: str = "combined",
                         config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Analizuje listę wideo.

    Args:
        video_paths: Lista ścieżek do wideo
        mode: Tryb detekcji
        config_path: Ścieżka do konfiguracji

    Returns:
        Lista wyników analizy
    """
    results = []
    detector = EnhancedDeepfakeDetector(config_path)

    for i, video_path in enumerate(video_paths):
        print(f"[Batch Analysis] {i + 1}/{len(video_paths)}: {os.path.basename(video_path)}")

        try:
            analysis = detector.analyze_video(video_path, mode)
            summary = detector.get_summary(analysis)
            summary["video_path"] = video_path
            summary["success"] = True
            results.append(summary)

        except Exception as e:
            print(f"[Batch Analysis] Błąd analizy {video_path}: {e}")
            results.append({
                "video": os.path.basename(video_path),
                "success": False,
                "error": str(e)
            })

    return results


# =============================================================================
# Eksport i serializacja
# =============================================================================

def save_analysis(analysis: VideoAnalysis, filepath: str):
    """Zapisuje analizę do pliku."""
    with open(filepath, 'wb') as f:
        pickle.dump(analysis, f)


def load_analysis(filepath: str) -> VideoAnalysis:
    """Wczytuje analizę z pliku."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Testowanie
# =============================================================================

if __name__ == "__main__":
    # Przykładowe użycie
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        mode = sys.argv[2] if len(sys.argv) > 2 else "combined"

        print(f"Analizuję: {video_path} (tryb: {mode})")

        result = analyze_video_enhanced(video_path, mode)

        print("\n=== WYNIKI ANALIZY ===")
        print(f"Wideo: {result['video']}")
        print(f"Czas trwania: {result['duration']}")
        print(f"Prawdopodobieństwo AI: {result['ai_probability']}")
        print(f"Prawdopodobieństwo Deepfake: {result['deepfake_probability']}")
        print(f"Watermark: {'TAK' if result['watermark_detected'] else 'NIE'}")
        print(f"Flagi detekcji: {', '.join(result['detection_flags'])}")
        print(f"Ratio twarzy: {result['face_ratio']}")
        print(f"Klatki przeanalizowane: {result['frames_analyzed']}")
        print("\nKluczowe wnioski:")
        for finding in result.get('key_findings', []):
            print(f"  - {finding}")

    else:
        print("Użycie: python enhanced_detector.py <ścieżka_do_wideo> [tryb]")
        print("Tryby: ai, deepfake, combined")
        print("\nPrzykład: python enhanced_detector.py video.mp4 ai")