"""Shared pytest setup for workspace-local temporary files."""

from pathlib import Path

Path(".runtime").mkdir(exist_ok=True)
