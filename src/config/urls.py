"""Root URL configuration for DrawingIA."""

from __future__ import annotations

from django.urls import include, path

urlpatterns = [
    path("", include("apps.recognizer.urls")),
]
