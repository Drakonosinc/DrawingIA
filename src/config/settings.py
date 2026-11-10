"""Django settings for DrawingIA."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-drawingia-dev-key-change-in-production-9f8e7d6c5b4a",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "apps.recognizer.apps.RecognizerConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [SRC_DIR / "apps" / "recognizer" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [
    SRC_DIR / "apps" / "recognizer" / "static",
]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Model checkpoint path
MODEL_ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_CHECKPOINT_PATH = MODEL_ARTIFACTS_DIR / "drawing_cnn_v1.pth"
PYFORGEAI_CHECKPOINT_PATH = MODEL_ARTIFACTS_DIR / "drawing_mlp_pyforgeai.pth"
