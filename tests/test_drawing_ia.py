"""Unit and integration test suite for DrawingIA Studio."""

from __future__ import annotations

import base64
import io
import json
from django.test import Client, SimpleTestCase
import numpy as np
from PIL import Image, ImageDraw

from apps.recognizer.domain.entities import CLASS_IDS
from apps.recognizer.infrastructure.dataset.synthetic_strokes import ProceduralStrokeGenerator
from apps.recognizer.infrastructure.preprocessing.image_pipeline import CenterOfMassPreprocessor
from apps.recognizer.services.recognizer_service import DrawingRecognizerService


class DrawingIATestSuite(SimpleTestCase):
    """End-to-end verification of preprocessing, hybrid AI inference, XAI heatmaps, and HTTP APIs."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.service = DrawingRecognizerService.get_instance()

    def _create_base64_canvas(self, draw_stroke: bool = True) -> str:
        img = Image.new("RGB", (200, 200), color=(0, 0, 0))
        if draw_stroke:
            draw = ImageDraw.Draw(img)
            draw.ellipse((45, 40, 155, 160), outline=(255, 255, 255), width=16)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def test_01_preprocessor_center_of_mass_and_blank(self) -> None:
        preprocessor = CenterOfMassPreprocessor()
        b64_img = self._create_base64_canvas(draw_stroke=True)
        grid = preprocessor.process(b64_img)
        self.assertEqual(grid.shape, (28, 28))
        self.assertGreater(float(grid.max()), 0.8)

        blank_b64 = self._create_base64_canvas(draw_stroke=False)
        blank_grid = preprocessor.process(blank_b64)
        self.assertEqual(float(blank_grid.max()), 0.0)

    def test_02_procedural_generator_all_15_classes(self) -> None:
        gen = ProceduralStrokeGenerator(seed=123)
        for cid in CLASS_IDS:
            sample = gen.render_sample(cid, jitter=True)
            self.assertEqual(sample.shape, (28, 28))
            self.assertGreater(float(sample.max()), 0.5, f"Class {cid} rendered empty sample")

    def test_03_hybrid_inference_and_xai_heatmap(self) -> None:
        sample_star = self.service.generator.render_sample("star", jitter=False)
        for engine in ("ensemble", "cnn", "pyforgeai"):
            payload = self.service.predict(sample_star, engine=engine, top_k=5)
            self.assertEqual(len(payload.top_predictions), 5)
            self.assertEqual(len(payload.normalized_grid_28x28), 28)
            self.assertEqual(len(payload.activation_heatmap_28x28), 28)
            self.assertGreater(payload.confidence, 0.25)

    def test_04_online_active_learning_teach(self) -> None:
        sample_heart = self.service.generator.render_sample("heart", jitter=True)
        res = self.service.teach(sample_heart, "heart")
        self.assertEqual(res["taught_class"], "heart")
        self.assertGreaterEqual(res["new_confidence"], 50.0)

    def test_05_http_endpoints(self) -> None:
        client = Client()

        # 1. GET /
        resp_index = client.get("/")
        self.assertEqual(resp_index.status_code, 200)
        self.assertIn(b"DrawingIA", resp_index.content)

        # 2. POST /api/predict/
        b64_img = self._create_base64_canvas(draw_stroke=True)
        resp_pred = client.post(
            "/api/predict/",
            data=json.dumps({"image": b64_img, "engine": "ensemble"}),
            content_type="application/json",
        )
        self.assertEqual(resp_pred.status_code, 200)
        pred_json = resp_pred.json()
        self.assertEqual(pred_json["status"], "ok")
        self.assertIn("activation_heatmap_28x28", pred_json["prediction"])

        # 3. POST /api/teach/
        resp_teach = client.post(
            "/api/teach/",
            data=json.dumps({"image": b64_img, "class_id": "0"}),
            content_type="application/json",
        )
        self.assertEqual(resp_teach.status_code, 200)
        self.assertEqual(resp_teach.json()["status"], "ok")

        # 4. GET /api/sample/
        resp_sample = client.get("/api/sample/?class_id=star")
        self.assertEqual(resp_sample.status_code, 200)
        self.assertEqual(resp_sample.json()["class_id"], "star")

        # 5. GET /api/stats/
        resp_stats = client.get("/api/stats/")
        self.assertEqual(resp_stats.status_code, 200)
        self.assertEqual(resp_stats.json()["stats"]["total_classes"], 15)
