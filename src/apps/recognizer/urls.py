"""URL routing for the recognizer application."""

from django.urls import path
from . import views

app_name = "recognizer"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/teach/", views.api_teach, name="api_teach"),
    path("api/retrain/", views.api_retrain, name="api_retrain"),
    path("api/stats/", views.api_stats, name="api_stats"),
    path("api/sample/", views.api_sample, name="api_sample"),
]
