"""Entrypoint untuk hosting yang mencari `app` di index.py (mis. Vercel)."""
from app.api.main import app  # noqa: F401
