"""Django settings for auditory_auth project."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-dev-only-change-in-production"

DEBUG = True

ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "corsheaders",
    "core",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

# CORS settings to allow frontend to communicate with backend
# Allow all origins in development
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

ROOT_URLCONF = "auditory_auth.urls"

# No templates needed - this is an API-only backend
TEMPLATES = []

WSGI_APPLICATION = "auditory_auth.wsgi.application"

# No database needed yet
DATABASES = {}

STATIC_URL = "/static/"
