"""Residual CNN + PyForgeAI Hybrid Neural Classifier with Spatial XAI Heatmaps."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyforgeai import AIFactory, load_model as pyforge_load_model, save_model as pyforge_save_model

from ...domain.entities import CLASS_IDS
from ...domain.interfaces import IDrawingClassifier


class ResidualDrawingCNN(nn.Module):
    """Compact Residual Convolutional Network for 28x28 stroke classification with Grad-CAM-like spatial maps."""

    def __init__(self, num_classes: int = len(CLASS_IDS)) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

        # Stem: 28x28x1 -> 28x28x32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual block at 14x14x32
        self.res_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.res_bn1 = nn.BatchNorm2d(32)
        self.res_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(32)

        # Deep feature extractor: 14x14x32 -> 7x7x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc_out = nn.Linear(128, self.num_classes)

        self.last_feature_map: torch.Tensor | None = None

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, 28, 28)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # (B, 32, 14, 14)

        identity = x
        out = F.relu(self.res_bn1(self.res_conv1(x)))
        out = self.res_bn2(self.res_conv2(out))
        x = F.relu(out + identity)

        feat_14x14 = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 14, 14)
        pooled_7x7 = F.max_pool2d(feat_14x14, 2)  # (B, 64, 7, 7)
        self.last_feature_map = feat_14x14.detach()

        flat = pooled_7x7.view(pooled_7x7.size(0), -1)
        hidden = F.relu(self.fc1(self.dropout(flat)))
        logits = self.fc_out(hidden)
        return logits, feat_14x14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits


class HybridDrawingClassifier(IDrawingClassifier):
    """Combines a PyTorch Residual CNN and a PyForgeAI MLP with Explainable AI spatial heatmaps."""

    def __init__(self, num_classes: int = len(CLASS_IDS)) -> None:
        self.num_classes = int(num_classes)
        self.device = torch.device("cpu")
        self.cnn = ResidualDrawingCNN(num_classes=self.num_classes).to(self.device)
        # PyForgeAI MLP (784 -> 256 -> 128 -> 15) built via PyForgeAI's AIFactory
        self.pyforge_mlp = AIFactory().create_Model(
            type_model="Pytorch",
            input_size=28 * 28,
            output_size=self.num_classes,
            hidden_sizes=[256, 128],
            activation="leaky_relu",
        ).to(self.device)
        self.cnn.eval()
        self.pyforge_mlp.eval()

    def predict_with_heatmap(
        self,
        grid_28x28: np.ndarray,
        engine: str = "ensemble",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute class probabilities and a 28x28 spatial attention heatmap."""
        self.cnn.eval()
        self.pyforge_mlp.eval()

        arr = np.asarray(grid_28x28, dtype=np.float32).reshape(1, 1, 28, 28)
        tensor_2d = torch.from_numpy(arr).to(self.device)
        tensor_flat = tensor_2d.view(1, 28 * 28)

        with torch.no_grad():
            cnn_logits, feat_14x14 = self.cnn.forward_features(tensor_2d)
            cnn_probs = F.softmax(cnn_logits, dim=-1).cpu().numpy()[0]

            mlp_logits = self.pyforge_mlp(tensor_flat)
            mlp_probs = F.softmax(mlp_logits, dim=-1).cpu().numpy()[0]

            mode = engine.strip().lower()
            if mode == "cnn":
                probs = cnn_probs
            elif mode in ("pyforgeai", "mlp"):
                probs = mlp_probs
            else:
                probs = 0.78 * cnn_probs + 0.22 * mlp_probs
                probs = probs / max(float(probs.sum()), 1e-8)

            # Build 28x28 spatial activation map from the 14x14 convolutional feature map
            spatial_map = feat_14x14.mean(dim=1, keepdim=True)  # (1, 1, 14, 14)
            upsampled = F.interpolate(
                spatial_map,
                size=(28, 28),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()

            if upsampled.max() > upsampled.min():
                upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min())
            else:
                upsampled = np.zeros((28, 28), dtype=np.float32)

        return probs.astype(np.float32), upsampled.astype(np.float32)

    def teach_sample(self, grid_28x28: np.ndarray, class_idx: int, steps: int = 5) -> float:
        """Online active-learning update on a user-labeled stroke sample."""
        self.cnn.train()
        self.pyforge_mlp.train()

        arr = np.asarray(grid_28x28, dtype=np.float32).reshape(1, 1, 28, 28)
        x_2d = torch.from_numpy(arr).to(self.device)
        x_flat = x_2d.view(1, 28 * 28)
        target = torch.tensor([int(class_idx)], dtype=torch.long, device=self.device)

        opt_cnn = torch.optim.AdamW(self.cnn.parameters(), lr=1.5e-3, weight_decay=1e-4)
        opt_mlp = torch.optim.AdamW(self.pyforge_mlp.parameters(), lr=1.5e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        last_loss = 0.0
        for _ in range(max(1, int(steps))):
            opt_cnn.zero_grad()
            opt_mlp.zero_grad()
            loss_cnn = criterion(self.cnn(x_2d), target)
            loss_mlp = criterion(self.pyforge_mlp(x_flat), target)
            total_loss = loss_cnn + loss_mlp
            total_loss.backward()
            opt_cnn.step()
            opt_mlp.step()
            last_loss = float(loss_cnn.item())

        self.cnn.eval()
        self.pyforge_mlp.eval()
        return last_loss

    def save_checkpoints(self, cnn_path: Path, mlp_path: Path) -> None:
        cnn_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.cnn.state_dict(),
                "num_classes": self.num_classes,
            },
            cnn_path,
        )
        pyforge_save_model(self.pyforge_mlp, str(mlp_path))

    def load_checkpoints(self, cnn_path: Path, mlp_path: Path) -> bool:
        loaded = False
        if cnn_path.exists():
            ckpt = torch.load(cnn_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.cnn.load_state_dict(state_dict, strict=False)
            self.cnn.eval()
            loaded = True
        if mlp_path.exists():
            restored_mlp = pyforge_load_model(
                str(mlp_path),
                "Pytorch",
                input_size=28 * 28,
                output_size=self.num_classes,
                hidden_sizes=[256, 128],
            )
            if restored_mlp is not None:
                self.pyforge_mlp = restored_mlp.to(self.device)
                self.pyforge_mlp.eval()
                loaded = True
        return loaded
