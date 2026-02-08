from django.urls import path
from . import views

urlpatterns = [
    path("", views.api_info, name="api_info"),
    path("api/about/", views.about_content, name="about_content"),
    path("api/process-audio/", views.process_audio, name="process_audio"),
    path("api/register-voice/", views.register_voice, name="register_voice"),
    path("api/authenticate-voice/", views.authenticate_voice, name="authenticate_voice"),
]
