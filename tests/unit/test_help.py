"""Built-in help assistant: every answer is reachable, nonsense is not answered, page context helps."""
import pytest

from nexora.help import assistant as A
from nexora.help.kb import KB, SUGGESTED
from nexora.ui.state import PAGES


def test_knowledge_base_is_consistent():
    ids = [e["id"] for e in KB]
    assert len(ids) == len(set(ids)) and len(KB) >= 50
    for e in KB:
        assert e["q"] and e["a"].strip() and (e["page"] == "any" or e["page"] in PAGES), e["id"]
        assert e.get("goto") is None or e["goto"] in PAGES, e["id"]
    assert set(SUGGESTED) <= set(PAGES) and all(i in ids for v in SUGGESTED.values() for i in v)


@pytest.mark.parametrize("e", KB, ids=lambda e: e["id"])
def test_every_phrasing_finds_its_own_answer(e):
    for q in e["q"]:
        a = A.ask(q, e["page"] if e["page"] != "any" else None)
        assert a.found and a.entry["id"] == e["id"], (q, a.entry and a.entry["id"])


@pytest.mark.parametrize("q,expected", [
    ("how do i merge two csv files", "combine"), ("change colour of graph", "colours"), ("my data is imbalanced", "imbalance"),
    ("forcast sales next month", "forecast"), ("how to remove outlers", "outliers_prep"), ("predict new customers with my model", "saved_model"),
    ("explain recall", "metric_defs"), ("can i save my chart as image", "chart_image"), ("where do i click to train", "ml_start"),
])
def test_free_wording_and_typos(q, expected):
    a = A.ask(q)
    assert a.found and a.entry["id"] == expected


@pytest.mark.parametrize("q", ["what is the weather today", "what is the weather", "tell me a joke", "pizza recipe", "hello", "", "  ",
                               "who won the match", "write me a poem", "best movie of 2020"])
def test_unrelated_questions_are_not_answered(q):
    a = A.ask(q)
    assert not a.found and a.entry is None


def test_page_context_and_suggestions():
    assert [e["id"] for e in A.suggestions("Prepare Data")][:2] == ["prep_kinds", "domain"]
    assert A.suggestions("Unknown page")
    assert "General" in A.topics() and sum(len(v) for v in A.topics().values()) == len(KB)
