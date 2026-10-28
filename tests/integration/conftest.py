import json
from pathlib import Path

import pytest

from nexora.workflow import Workspace

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "sample_data" / "synthetic"
AS_OF = "2026-10-26T00:00:00+00:00"


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "data"))
    return Workspace()


@pytest.fixture(scope="session")
def truth():
    return json.loads((SAMPLES / "expected_results.json").read_text())


def decide(plan, **by_op):
    """Pick a choice for each decision action by operation name; default is the conservative one."""
    return {a.action_id: by_op.get(a.op, a.default_choice) for a in plan.actions if a.kind == "decision"}
