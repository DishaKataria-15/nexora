"""Combining datasets: joins, per-key summaries, stacking, and the match reporting around them."""
import pandas as pd
import pytest

from nexora.combine import CombineError, Step, combine, suggest_keys


@pytest.fixture()
def movies():
    return pd.DataFrame({"movie_id": ["1", "2", "3", "4"], "title": ["Alpha", "Beta", "Gamma", "Delta"], "year": ["2001", "2002", "2003", "2004"]})


@pytest.fixture()
def ratings():
    return pd.DataFrame({"user": ["u1", "u2", "u1", "u3", "u2", "u4"], "movie_id": ["1", "1", "2", "2", "2", "9"],
                         "rating": ["4", "5", "3", "2", "4", "5"]})


def test_left_join_adds_movie_details_to_every_rating(ratings, movies):
    out, [rep] = combine({"ratings": ratings, "movies": movies}, "ratings", [Step("join", "movies", ["movie_id"], ["movie_id"])], 10_000)
    assert len(out) == len(ratings) and list(out.columns) == ["user", "movie_id", "rating", "title", "year"]
    assert out.loc[out["movie_id"] == "2", "title"].unique().tolist() == ["Beta"]
    assert rep.matched_rows == 5 and rep.unmatched_main == 1 and rep.examples_unmatched == ["9"]
    assert rep.unmatched_other_keys == 2                         # movies 3 and 4 have no ratings
    assert pd.isna(out.loc[out["movie_id"] == "9", "title"]).all()
    assert any("found no match" in w for w in rep.warnings)


def test_inner_and_outer_joins(ratings, movies):
    inner, _ = combine({"r": ratings, "m": movies}, "r", [Step("join", "m", ["movie_id"], ["movie_id"], how="inner")], 10_000)
    assert len(inner) == 5 and "9" not in set(inner["movie_id"])
    outer, _ = combine({"r": ratings, "m": movies}, "r", [Step("join", "m", ["movie_id"], ["movie_id"], how="outer")], 10_000)
    assert len(outer) == 8 and {"3", "4"} <= set(outer["movie_id"])   # unrated movies appear with their id filled in


def test_summary_adds_per_movie_statistics(ratings, movies):
    out, [rep] = combine({"m": movies, "r": ratings}, "m",
                         [Step("summary", "r", ["movie_id"], ["movie_id"], aggs={"rating": ["mean", "count"], "user": ["nunique"]})], 10_000)
    assert len(out) == 4                                          # one row per movie, never multiplied
    row = out.set_index("movie_id")
    assert row.loc["1", "rating_mean"] == 4.5 and row.loc["2", "rating_count"] == 3 and row.loc["2", "user_nunique"] == 3
    assert pd.isna(row.loc["3", "rating_mean"]) and rep.columns_added == ["rating_mean", "rating_count", "user_nunique"]


def test_duplicate_keys_in_a_join_are_reported_and_explosions_refused(movies, ratings):
    out, [rep] = combine({"m": movies, "r": ratings}, "m", [Step("join", "r", ["movie_id"], ["movie_id"])], 10_000)
    assert rep.duplicate_other_keys == 3 and any("Summarise" in w for w in rep.warnings) and len(out) == 7
    big = pd.DataFrame({"movie_id": ["1"] * 200, "x": range(200)})
    with pytest.raises(CombineError, match="probably not the right ones"):
        combine({"m": movies, "b": big}, "m", [Step("join", "b", ["movie_id"], ["movie_id"])], 10_000)
    with pytest.raises(CombineError, match="above the limit"):
        combine({"m": movies, "b": big}, "m", [Step("join", "b", ["movie_id"], ["movie_id"])], 50)


def test_key_normalisation_spaces_case_and_spreadsheet_decimals():
    a = pd.DataFrame({"id": [" 12", "x7", "007"], "v": ["a", "b", "c"]})
    b = pd.DataFrame({"code": ["12.0", "X7", "7"], "w": ["A", "B", "C"]})
    out, [rep] = combine({"a": a, "b": b}, "a", [Step("join", "b", ["id"], ["code"], ignore_case=True)], 100)
    assert out["w"].tolist()[:2] == ["A", "B"]
    assert pd.isna(out["w"].iloc[2])                              # "007" is not silently turned into "7"
    out2, _ = combine({"a": a, "b": b}, "a", [Step("join", "b", ["id"], ["code"])], 100)
    assert pd.isna(out2["w"].iloc[1])                             # case matters unless asked otherwise


def test_clashing_column_names_get_a_prefix_and_multi_key_joins():
    a = pd.DataFrame({"user": ["u1", "u1"], "day": ["d1", "d2"], "score": ["1", "2"]})
    b = pd.DataFrame({"user": ["u1", "u1"], "day": ["d2", "d1"], "score": ["9", "8"]})
    out, [rep] = combine({"a": a, "b (2)": b}, "a", [Step("join", "b (2)", ["user", "day"], ["user", "day"])], 100)
    assert out["b_score"].tolist() == ["8", "9"] and rep.columns_added == ["b_score"]


def test_stack_rows_and_three_way_chain(movies):
    jan = pd.DataFrame({"movie_id": ["1"], "rating": ["5"]})
    feb = pd.DataFrame({"movie_id": ["2", "3"], "rating": ["4", "3"], "device": ["tv", "phone"]})
    out, reps = combine({"jan": jan, "feb": feb, "movies": movies}, "jan",
                        [Step("stack", "feb"), Step("join", "movies", ["movie_id"], ["movie_id"], columns=["title"])], 100)
    assert len(out) == 3 and out["title"].tolist() == ["Alpha", "Beta", "Gamma"] and "year" not in out
    assert any("do not have the same columns" in w for w in reps[0].warnings)


@pytest.mark.parametrize("step,msg", [(Step("join", "m", [], ["movie_id"]), "key column"), (Step("join", "m", ["nope"], ["movie_id"]), "no column"),
                                      (Step("join", "m", ["movie_id"], ["movie_id", "title"]), "same number"),
                                      (Step("summary", "m", ["movie_id"], ["movie_id"]), "at least one statistic"),
                                      (Step("summary", "m", ["movie_id"], ["movie_id"], aggs={"title": ["mean"]}), "not numeric"),
                                      (Step("join", "m", ["movie_id"], ["movie_id"], how="cross"), "Unknown join"),
                                      (Step("magic", "m"), "Unknown combine mode")])
def test_bad_settings_are_readable_errors(ratings, movies, step, msg):
    with pytest.raises(CombineError, match=msg):
        combine({"r": ratings, "m": movies}, "r", [step], 1000)


def test_sources_are_never_modified(ratings, movies):
    r0, m0 = ratings.copy(), movies.copy()
    combine({"r": ratings, "m": movies}, "r", [Step("join", "m", ["movie_id"], ["movie_id"], how="outer")], 1000)
    pd.testing.assert_frame_equal(ratings, r0)
    pd.testing.assert_frame_equal(movies, m0)


def test_key_suggestions(ratings, movies):
    assert suggest_keys(ratings, movies)[0][:2] == ("movie_id", "movie_id")


def test_workspace_saves_the_result_as_a_new_dataset(tmp_path, monkeypatch, ratings, movies):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "d"))
    from nexora.workflow import Workspace
    w = Workspace()
    hr = w.ingest_frame(ratings, "ratings", "upload")
    hm = w.ingest_frame(movies, "movies", "upload")
    h, reps, recipe = w.combine_datasets({"ratings": hr.dataset_id, "movies": hm.dataset_id}, "ratings",
                                         [Step("join", "movies", ["movie_id"], ["movie_id"])], "ratings with titles")
    assert h.source == "combined" and len(h.raw) == 6 and "title" in h.raw.columns and recipe["rows"] == 6
    assert len(w.open_dataset(hr.dataset_id).raw) == 6 and list(w.open_dataset(hr.dataset_id).raw.columns) == ["user", "movie_id", "rating"]


def test_blank_and_missing_keys_never_match_and_never_crash():
    a = pd.DataFrame({"id": ["1", "", None, "N/A", "2"], "v": list("abcde")})
    b = pd.DataFrame({"id": ["", "1", float("nan"), "n/a"], "w": ["blank", "one", "nan", "na"]})
    out, [rep] = combine({"a": a, "b": b}, "a", [Step("join", "b", ["id"], ["id"])], 100)
    assert out["w"].tolist()[0] == "one" and out["w"].iloc[1:].isna().all()
    assert rep.matched_rows == 1 and any("empty key" in w for w in rep.warnings)


def test_suggestions_ignore_blanks_measurements_and_unrelated_columns():
    rng = __import__("numpy").random.RandomState(0)
    sensors = pd.DataFrame({"temperature": [f"{v:.1f}" for v in rng.normal(50, 5, 300)], "note": [""] * 300, "qty": [str(i % 7) for i in range(300)]})
    churn = pd.DataFrame({"cancellation_reason": [""] * 290 + ["price"] * 10, "tenure_months": [str(i % 7) for i in range(300)],
                          "region": ["North", "South"] * 150})
    assert suggest_keys(sensors, churn) == []              # blank = blank, small number = small number and decimals are not keys
    orders = pd.DataFrame({"customer_id": [f"C{i % 40:03d}" for i in range(300)], "amount": ["1"] * 300})
    customers = pd.DataFrame({"cust_id": [f"C{i:03d}" for i in range(40)], "name": [f"n{i}" for i in range(40)]})
    assert suggest_keys(orders, customers)[0][:2] == ("customer_id", "cust_id")   # ID-like names with full overlap
