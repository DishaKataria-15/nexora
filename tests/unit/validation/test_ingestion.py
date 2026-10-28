import io

import pandas as pd
import pytest

from nexora.validation.ingestion import IngestError, load_upload, safe_filename


def load(data, name="t.csv", **kw):
    return load_upload(data if isinstance(data, bytes) else data.encode(), name, **kw)


def test_valid_csv_keeps_exact_text():
    r = load("id,code,amount\n007,A-1,1.50\n008,,2\n")
    assert r.n_rows == 2 and list(r.dataframe.columns) == ["id", "code", "amount"]
    assert r.dataframe["id"].tolist() == ["007", "008"]          # leading zeros survive
    assert r.dataframe["code"].tolist() == ["A-1", ""]            # blanks stay blank, not NaN
    assert r.dataframe["amount"].tolist() == ["1.50", "2"]
    assert len(r.sha256) == 64 and r.encoding in ("utf-8-sig", "utf-8")


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_delimiters_are_detected(sep):
    r = load(f"a{sep}b{sep}c\n1{sep}2{sep}3\n4{sep}5{sep}6\n")
    assert list(r.dataframe.columns) == ["a", "b", "c"] and r.delimiter == sep


def test_bom_and_quoted_fields_with_commas_and_newlines():
    r = load(b"\xef\xbb\xbfname,note\n\"Smith, J\",\"line1\nline2\"\n")
    assert list(r.dataframe.columns) == ["name", "note"]
    assert r.dataframe["name"][0] == "Smith, J" and "\n" in r.dataframe["note"][0]


def test_cp1252_is_accepted_with_a_warning():
    r = load("name\ncaf\xe9\n".encode("cp1252"))
    assert r.dataframe["name"][0] == "café" and any("not valid UTF-8" in w for w in r.warnings)


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"), (b"   \n \n", "empty"), (b"a,b\n", "no data rows"), (b"a,b\n\n\n", "no data rows"),
    (b"\x00\x01\x02binary", "binary"),
])
def test_empty_or_unusable_files_are_rejected_with_a_message(data, msg):
    with pytest.raises(IngestError, match=msg):
        load(data)


@pytest.mark.parametrize("name", ["x.exe", "x.pdf", "x", "x.csv.exe", "x.json"])
def test_file_type_is_validated(name):
    with pytest.raises(IngestError, match="not accepted"):
        load("a,b\n1,2\n", name)


def test_size_row_and_column_limits():
    with pytest.raises(IngestError, match="limit is 0 MB|MB"):
        load("a,b\n1,2\n", max_bytes=3)
    with pytest.raises(IngestError, match="rows"):
        load("a\n" + "\n".join(str(i) for i in range(20)), max_rows=10)
    with pytest.raises(IngestError, match="columns"):
        load(",".join(f"c{i}" for i in range(6)) + "\n" + ",".join("1" * 6) + "\n", max_cols=5)


def test_ragged_rows_too_long_is_error_too_short_is_padded():
    with pytest.raises(IngestError, match="more fields"):
        load("a,b\n1,2,3\n")
    r = load("a,b,c\n1,2\n3,4,5\n")
    assert r.dataframe.iloc[0].tolist() == ["1", "2", ""] and any("padded" in w for w in r.warnings)


def test_duplicate_and_blank_headers_are_renamed_not_lost():
    r = load("a,a, ,b\n1,2,3,4\n")
    assert list(r.dataframe.columns) == ["a", "a__2", "unnamed_3", "b"] and len(r.warnings) >= 2


def test_blank_lines_are_ignored_with_a_note():
    r = load("a,b\n1,2\n\n3,4\n")
    assert r.n_rows == 2 and any("blank line" in w for w in r.warnings)


def test_xlsx_upload_and_fake_xlsx():
    buf = io.BytesIO()
    pd.DataFrame({"id": ["007", "008"], "v": [1.5, 2]}).to_excel(buf, index=False)
    r = load_upload(buf.getvalue(), "t.xlsx")
    assert r.n_rows == 2 and r.dataframe["id"].tolist() == ["007", "008"] and any("Excel" in w for w in r.warnings)
    with pytest.raises(IngestError, match="xlsx"):
        load_upload(b"not a zip", "t.xlsx")
    with pytest.raises(IngestError):
        load_upload(b"PK\x03\x04garbage", "t.xlsx")


def test_filename_is_sanitised():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\dir\\my file<>.csv") == "my file__.csv"
    assert "/" not in safe_filename("a/b/c.csv") and safe_filename("") == "upload"
    r = load("a\n1\n", "../../evil.csv")
    assert r.filename == "evil.csv"
