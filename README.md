# ✦ DrawingIA Studio 2.0 — Full-Stack Neural Sketch Recognition & Explainable AI

[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-4B32C3?logo=read-the-docs&logoColor=white)](https://deepwiki.com/Drakonosinc/DrawingIA)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.2-092E20?logo=django&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Residual_CNN-EE4C2C?logo=pytorch&logoColor=white)
![PyForgeAI](https://img.shields.io/badge/AI_Engine-pyforgeai--engine-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

**DrawingIA Studio 2.0** is a modular Full-Stack Web + Deep Learning platform built with **Django 5.2**, **PyTorch**, and **[`pyforgeai-engine`](https://pypi.org/project/pyforgeai-engine/)** (`PyForgeAI`). It recognizes hand-drawn digits (`0–9`) and geometric/symbolic doodles (`★ Star`, `♥ Heart`, `☺ Smiley`, `⚡ Lightning`, `△ Triangle`) in real time directly from an interactive HTML5 Canvas, while exposing **Explainable AI (XAI)** spatial feature attribution maps and **Online Active Learning** ("Teach the AI").

---

## 🚀 Key Capabilities

| Module | Description |
| :--- | :--- |
| **Hybrid Neural Ensemble** | Combines a custom **PyTorch Residual Convolutional Neural Network (`ResidualDrawingCNN`)** for spatial invariance with a **`PyForgeAI` Multi-Layer Perceptron (`AIFactory`)** into a calibrated probability ensemble (`ensemble`, `cnn`, or `pyforgeai` selectable live). |
| **Explainable AI (XAI) Spatial Inspector** | Generates real-time `28×28` **Convolutional Activation Heatmaps** alongside the **Center-of-Mass Normalized Input Tensor**, revealing which stroke regions drove the neural decision. |
| **MNIST-Style Center-of-Mass Pipeline** | Automatically crops the stroke bounding box, scales to a `20×20` inner frame preserving aspect ratio, and translates the mass centroid to `(14, 14)` on a `28×28` tensor. |
| **Online Active Learning ("Teach the AI")** | 1-click weight adaptation (`POST /api/teach/`) that fine-tunes both the Residual CNN and the `PyForgeAI` MLP on user strokes with synthetic experience replay to prevent catastrophic forgetting. |
| **⚡ QuickDraw Challenge Mode** | Built-in timed 10-second drawing mini-game with live streak & score tracking, plus a 1-click **✨ Auto-Demo** generator (`GET /api/sample/`). |

---

## 🏗️ Clean Architecture

```text
DrawingIA/
├── manage.py
├── requirements.txt
├── artifacts/
│   ├── drawing_cnn_v1.pth              # Pre-trained PyTorch Residual CNN weights
│   └── drawing_mlp_pyforgeai.pth       # Pre-trained PyForgeAI MLP weights
├── src/
│   ├── config/                         # Django 5.2 configuration (ASGI/WSGI/Settings/URLs)
│   └── apps/
│       └── recognizer/
│           ├── domain/                 # Pure domain entities & repository/service interfaces
│           │   ├── entities.py         # ClassProbability, PredictionPayload, TrainingReport
│           │   └── interfaces.py       # IImagePreprocessor, IDrawingClassifier
│           ├── infrastructure/         # Concrete ML, computer vision, and synthetic data adapters
│           │   ├── ai/
│           │   │   └── cnn_classifier.py      # ResidualDrawingCNN + PyForgeAI HybridClassifier
│           │   ├── dataset/
│           │   │   └── synthetic_strokes.py   # Procedural 15-class stroke generator with affine jitter
│           │   └── preprocessing/
│           │       └── image_pipeline.py      # Bounding-box + Center-of-Mass 28x28 normalizer
│           ├── services/
│           │   └── recognizer_service.py      # Thread-safe singleton application orchestrator
│           ├── templates/recognizer/
│           │   └── index.html                 # Dark Glassmorphism Studio UI
│           ├── static/recognizer/
│           │   ├── css/styles.css
│           │   └── js/app.js                  # HTML5 Canvas, XAI renderer & Challenge loop
│           ├── views.py                       # REST API endpoints (/api/predict, /api/teach, ...)
│           └── urls.py
└── tests/
    └── test_drawing_ia.py              # End-to-end unit & HTTP integration test suite
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Studio

```bash
python manage.py runserver
```

Then open **`http://127.0.0.1:8000/`** in your browser.
> On first launch, if model checkpoints are not present in `artifacts/`, `DrawingRecognizerService` automatically synthesizes a 15-class procedural stroke dataset and trains both the Residual CNN and `PyForgeAI` MLP in a few seconds.

### 3. Run the Automated Test Suite

```bash
python manage.py test tests -v 2
```

---

## 🔌 REST API Reference

| Method | Endpoint | Payload / Params | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | — | Renders the interactive DrawingIA Studio web interface. |
| `POST` | `/api/predict/` | `{"image": "data:image/png;base64,...", "engine": "ensemble"}` | Returns top-5 class probabilities, Shannon entropy, latency, normalized `28×28` tensor, and `28×28` XAI activation heatmap. |
| `POST` | `/api/teach/` | `{"image": "data:image/png;base64,...", "class_id": "star"}` | Performs online active-learning gradient steps with replay memory and persists updated checkpoints. |
| `POST` | `/api/retrain/` | `{"samples_per_class": 90, "epochs": 6}` | Re-trains and calibrates the hybrid ensemble on procedural + user-taught strokes. |
| `GET` | `/api/sample/` | `?class_id=heart` (optional) | Generates a synthetic stroke sample and returns its `28×28` grid + live prediction. |
| `GET` | `/api/stats/` | — | Returns model telemetry, validation accuracy, and recent prediction history. |
