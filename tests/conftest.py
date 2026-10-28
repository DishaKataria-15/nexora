import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexora.validation.profiles import load_builtin_profiles  # noqa: E402


@pytest.fixture(scope="session")
def builtin_profiles():
    return load_builtin_profiles(ROOT / "config" / "domains")
