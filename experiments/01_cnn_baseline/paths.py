"""Paths for the CNN baseline experiment. Scripts can run from any working directory."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO_ROOT = HERE.parents[1]
MODELS = REPO_ROOT / "models"
BREAKHIS_ROOT = REPO_ROOT / "BreaKHis_v1" / "BreaKHis_v1"
