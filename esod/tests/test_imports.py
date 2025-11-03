"""Smoke tests for core project imports."""

import subprocess
import sys


_MODULES = [
    "torch",
    "numpy",
    "models.yolo",
    "utils.general",
]


def test_imports():
    """Ensure key modules import without raising in a clean process."""
    for name in _MODULES:
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                f"import importlib; importlib.import_module('{name}')",
            ]
        )


def test_cv2_import():
    """Validate OpenCV import in a clean interpreter (workaround for pytest rewrite bug)."""
    subprocess.check_call([sys.executable, "-c", "import cv2; print(cv2.__version__)"])
