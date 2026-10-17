import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

VINI_WEIGHTS = Path.home() / "unisagrado" / "epi-vini-referencia" / "best.pt"


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session", autouse=True)
def isolated_ultralytics(tmp_path_factory):
    """Point Ultralytics and matplotlib at a temporary config dir before any test imports them.

    ~/.config/Ultralytics/settings.json sends runs into this repository; tests always pass an
    explicit project dir, and nothing may install packages. YOLO_OFFLINE does not stop
    Ultralytics' `check_font` from downloading Arial.ttf into the run's config dir during the
    end-to-end validation test.
    """
    config = tmp_path_factory.mktemp("ultralytics-config")
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["MPLCONFIGDIR"] = str(config / "matplotlib")
    yield
