"""Domain entities and value objects for DrawingIA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# 15 Recognizable Classes: Digits 0-9 + 5 Iconic Shapes
DRAWING_CLASSES: tuple[dict[str, str], ...] = (
    {"id": "0", "label": "Digit 0", "icon": "0", "category": "digit"},
    {"id": "1", "label": "Digit 1", "icon": "1", "category": "digit"},
    {"id": "2", "label": "Digit 2", "icon": "2", "category": "digit"},
    {"id": "3", "label": "Digit 3", "icon": "3", "category": "digit"},
    {"id": "4", "label": "Digit 4", "icon": "4", "category": "digit"},
    {"id": "5", "label": "Digit 5", "icon": "5", "category": "digit"},
    {"id": "6", "label": "Digit 6", "icon": "6", "category": "digit"},
    {"id": "7", "label": "Digit 7", "icon": "7", "category": "digit"},
    {"id": "8", "label": "Digit 8", "icon": "8", "category": "digit"},
    {"id": "9", "label": "Digit 9", "icon": "9", "category": "digit"},
    {"id": "triangle", "label": "Triangle", "icon": "△", "category": "shape"},
    {"id": "star", "label": "Star", "icon": "★", "category": "shape"},
    {"id": "heart", "label": "Heart", "icon": "♥", "category": "shape"},
    {"id": "lightning", "label": "Lightning", "icon": "⚡", "category": "shape"},
    {"id": "smile", "label": "Smile", "icon": "☺", "category": "shape"},
)

CLASS_IDS: tuple[str, ...] = tuple(item["id"] for item in DRAWING_CLASSES)
CLASS_ID_TO_INDEX: dict[str, int] = {cid: idx for idx, cid in enumerate(CLASS_IDS)}
INDEX_TO_CLASS_META: dict[int, dict[str, str]] = {idx: item for idx, item in enumerate(DRAWING_CLASSES)}


@dataclass(frozen=True, slots=True)
class ClassProbability:
    """Single class prediction probability item."""

    class_id: str
    label: str
    icon: str
    category: str
    probability: float


@dataclass(slots=True)
class PredictionPayload:
    """Complete inference result including XAI spatial heatmap and telemetry."""

    predicted_id: str
    predicted_label: str
    predicted_icon: str
    confidence: float
    entropy: float
    latency_ms: float
    engine_used: str
    top_predictions: list[ClassProbability] = field(default_factory=list)
    normalized_grid_28x28: list[list[float]] = field(default_factory=list)
    activation_heatmap_28x28: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize payload to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass(slots=True)
class TrainingReport:
    """Summary metrics produced after training or fine-tuning."""

    epochs: int
    samples_seen: int
    final_loss: float
    validation_accuracy: float
    duration_seconds: float
    engine: str
