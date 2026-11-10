"""Abstract domain interfaces for DrawingIA preprocessing and inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class IImagePreprocessor(ABC):
    """Contract for normalizing raw user canvas drawings into 28x28 tensors."""

    @abstractmethod
    def process(self, source: str | np.ndarray) -> np.ndarray:
        """Convert a Base64 PNG string or 2D array into a normalized (28, 28) float32 array in [0, 1]."""


class IDrawingClassifier(ABC):
    """Contract for neural drawing classifiers supporting inference, XAI heatmaps, and online teaching."""

    @abstractmethod
    def predict_with_heatmap(self, grid_28x28: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (probabilities_vector, spatial_heatmap_28x28) for a normalized (28, 28) input."""

    @abstractmethod
    def teach_sample(self, grid_28x28: np.ndarray, class_idx: int, steps: int = 4) -> float:
        """Perform online active-learning gradient updates on a user-labeled sample and return the loss."""
