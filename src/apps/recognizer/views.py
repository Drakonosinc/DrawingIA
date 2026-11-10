"""HTTP Views and REST API endpoints for DrawingIA Studio."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .domain.entities import CLASS_IDS, DRAWING_CLASSES
from .services.recognizer_service import DrawingRecognizerService


def _parse_json_body(request: HttpRequest) -> dict[str, Any]:
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    """Render the main DrawingIA Interactive Studio."""
    service = DrawingRecognizerService.get_instance()
    stats = service.get_stats()
    return render(
        request,
        "recognizer/index.html",
        {
            "drawing_classes": DRAWING_CLASSES,
            "drawing_classes_json": json.dumps(DRAWING_CLASSES),
            "initial_stats": stats,
        },
    )


@csrf_exempt
@require_POST
def api_predict(request: HttpRequest) -> JsonResponse:
    """Run real-time neural inference and return top-K probabilities + 28x28 XAI heatmap."""
    try:
        payload = _parse_json_body(request)
        image_data = payload.get("image")
        if image_data is None:
            return JsonResponse({"status": "error", "message": "Missing 'image' field."}, status=400)

        engine = str(payload.get("engine", "ensemble")).lower().strip()
        if engine not in {"ensemble", "cnn", "pyforgeai"}:
            engine = "ensemble"

        service = DrawingRecognizerService.get_instance()
        result = service.predict(image_data, engine=engine, top_k=5)
        return JsonResponse({"status": "ok", "prediction": result.to_dict()})
    except ValueError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:  # pragma: no cover - defensive API guard
        return JsonResponse({"status": "error", "message": f"Inference error: {exc}"}, status=500)


@csrf_exempt
@require_POST
def api_teach(request: HttpRequest) -> JsonResponse:
    """Online Active Learning endpoint: fine-tune the model on a user-labeled canvas drawing."""
    try:
        payload = _parse_json_body(request)
        image_data = payload.get("image")
        class_id = payload.get("class_id")
        if image_data is None or not class_id:
            return JsonResponse(
                {"status": "error", "message": "Both 'image' and 'class_id' are required."},
                status=400,
            )

        service = DrawingRecognizerService.get_instance()
        teach_result = service.teach(image_data, str(class_id))
        updated_prediction = service.predict(image_data, engine="ensemble", top_k=5)
        return JsonResponse(
            {
                "status": "ok",
                "teach_result": teach_result,
                "prediction": updated_prediction.to_dict(),
            }
        )
    except ValueError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:  # pragma: no cover
        return JsonResponse({"status": "error", "message": f"Active learning error: {exc}"}, status=500)


@csrf_exempt
@require_POST
def api_retrain(request: HttpRequest) -> JsonResponse:
    """Re-train and calibrate the hybrid CNN + PyForgeAI ensemble."""
    try:
        payload = _parse_json_body(request)
        samples_per_class = int(payload.get("samples_per_class", 80))
        epochs = int(payload.get("epochs", 5))
        samples_per_class = max(20, min(samples_per_class, 200))
        epochs = max(1, min(epochs, 15))

        service = DrawingRecognizerService.get_instance()
        report = service.train_on_procedural_dataset(
            samples_per_class=samples_per_class,
            epochs=epochs,
        )
        return JsonResponse({"status": "ok", "training_report": asdict(report)})
    except Exception as exc:  # pragma: no cover
        return JsonResponse({"status": "error", "message": f"Retraining failed: {exc}"}, status=500)


@require_GET
def api_stats(request: HttpRequest) -> JsonResponse:
    """Return telemetry, supported classes, recent predictions, and training accuracy."""
    service = DrawingRecognizerService.get_instance()
    return JsonResponse({"status": "ok", "stats": service.get_stats()})


@require_GET
def api_sample(request: HttpRequest) -> JsonResponse:
    """Generate a synthetic 28x28 stroke grid for a requested class (or random class)."""
    service = DrawingRecognizerService.get_instance()
    class_id = request.GET.get("class_id", "").strip().lower()
    if class_id not in CLASS_IDS:
        class_id = str(service.generator.rng.choice(CLASS_IDS))
    grid = service.generator.render_sample(class_id, jitter=True)
    prediction = service.predict(grid, engine="ensemble", top_k=5)
    return JsonResponse(
        {
            "status": "ok",
            "class_id": class_id,
            "grid_28x28": grid.round(3).tolist(),
            "prediction": prediction.to_dict(),
        }
    )
