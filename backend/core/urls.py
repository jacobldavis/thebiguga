from django.urls import path
from . import views

urlpatterns = [
    path("", views.api_info, name="api_info"),
    path("api/process-audio/", views.process_audio, name="process_audio"),
]
