"""Database and web API connectors: read-only access, query validation, secret handling, SSRF protection, limits."""
import json
import logging
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from nexora.connectors import database as D
from nexora.connectors import web_api as W
from nexora.workflow import Workspace


# ---- database ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def sqlite_db(tmp_path):
    p = tmp_path / "shop.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL, \"comment\" TEXT)")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", [(i, f"c{i % 7}", i * 1.5, "ok") for i in range(1, 501)])
    con.execute("CREATE TABLE secrets (k TEXT)")
    con.execute("CREATE VIEW big_orders AS SELECT * FROM orders WHERE amount > 600")
    con.commit()
    con.close()
    return str(p)


def test_sqlite_catalogue_preview_and_limits(sqlite_db):
    with D.DatabaseConnection(D.DBConfig("sqlite", database=sqlite_db)) as c:
        names = {(t["name"], t["kind"]) for t in c.list_tables()}
        assert {("orders", "table"), ("big_orders", "view")} <= names
        assert [x["name"] for x in c.list_columns("orders")] == ["id", "customer", "amount", "comment"]
        r = c.read_table("orders", limit=50)
        assert len(r.frame) == 50 and r.truncated and "more rows" in r.note()
        r = c.read_table("orders", columns=["id", "amount"], limit=1000)
        assert list(r.frame.columns) == ["id", "amount"] and len(r.frame) == 500 and not r.truncated
        assert c.count_rows("orders") == 500
        with pytest.raises(D.ConnectorError, match="Confirm the full read"):
            c.read_table("orders", full=True)
        assert len(c.read_table("orders", full=True, confirm_full=True).frame) == 500


def test_identifiers_are_checked_against_the_catalogue(sqlite_db):
    with D.DatabaseConnection(D.DBConfig("sqlite", database=sqlite_db)) as c:
        with pytest.raises(D.ConnectorError, match="not a table"):
            c.read_table("orders; DROP TABLE orders")
        with pytest.raises(D.ConnectorError, match="Unknown column"):
            c.read_table("orders", columns=["id", "amount) FROM secrets --"])
        assert c.count_rows("orders") == 500


def test_parameterised_queries_and_value_injection_is_inert(sqlite_db):
    with D.DatabaseConnection(D.DBConfig("sqlite", database=sqlite_db)) as c:
        r = c.run_query("SELECT customer, SUM(amount) AS total FROM orders WHERE customer = :who GROUP BY customer", {"who": "c3"})
        assert r.frame.iloc[0]["customer"] == "c3"
        r = c.run_query("SELECT * FROM orders WHERE customer = :who", {"who": "x' OR '1'='1"})
        assert len(r.frame) == 0                                  # the value is bound, never pasted into the SQL
        with pytest.raises(D.ConnectorError, match="parameter"):
            c.run_query("SELECT * FROM orders WHERE id = :id")
        r = c.run_query('SELECT "comment", REPLACE(customer, \'c\', \'C\') AS cu FROM orders WHERE id < 3 -- note: delete nothing')
        assert list(r.frame["cu"]) == ["C1", "C2"]


@pytest.mark.parametrize("sql", ["DELETE FROM orders", "UPDATE orders SET amount = 0", "DROP TABLE orders", "SELECT 1; DROP TABLE orders",
                                 "WITH x AS (SELECT 1) DELETE FROM orders", "SELECT * INTO copy FROM orders", "PRAGMA table_info(orders)",
                                 "ATTACH DATABASE 'x.db' AS x", "select * from orders /* unbalanced", "", "SELECT pg_sleep(100)",
                                 "SELECT load_extension('evil')", "INSERT INTO orders VALUES (1,'a',1,'b')"])
def test_anything_but_one_select_is_refused(sql):
    with pytest.raises(D.ConnectorError):
        D.validate_select(sql)


def test_keywords_inside_strings_and_quoted_names_are_allowed():
    D.validate_select("SELECT 'please delete me' AS msg, \"update\" FROM t WHERE note <> 'drop'")


def test_sqlite_is_opened_read_only_even_if_validation_were_bypassed(sqlite_db):
    with D.DatabaseConnection(D.DBConfig("sqlite", database=sqlite_db)) as c:
        from sqlalchemy import text
        with pytest.raises(Exception, match="readonly|read-only|read only"):
            with c.engine.connect() as con:
                con.execute(text("DELETE FROM orders"))
                con.commit()
        assert c.count_rows("orders") == 500


def test_slow_queries_are_stopped_by_the_timeout(sqlite_db):
    with D.DatabaseConnection(D.DBConfig("sqlite", database=sqlite_db, timeout_s=1)) as c:
        slow = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r"
        with pytest.raises(D.ConnectorError, match="longer than 1 seconds"):
            c.run_query(slow, limit=1)


def test_passwords_never_appear_in_repr_descriptions_or_errors(monkeypatch, caplog):
    cfg = D.DBConfig("postgresql", host="db.invalid", database="sales", username="ana", password="S3cr3t!pw", timeout_s=1)
    assert "S3cr3t" not in repr(cfg) and "S3cr3t" not in cfg.describe()
    assert D.redact("could not connect to postgresql://ana:S3cr3t!pw@db.invalid/sales", "S3cr3t!pw").count("S3cr3t") == 0
    assert "pwd=***" in D.redact("DRIVER=x;PWD=hunter2;UID=a").lower()
    ok, why = D.driver_status("postgresql")
    caplog.set_level(logging.DEBUG)
    with pytest.raises(D.ConnectorError) as ei:
        D.DatabaseConnection(cfg)
    assert "S3cr3t" not in str(ei.value) and "S3cr3t" not in caplog.text
    if not ok:
        assert "psycopg2" in str(ei.value)


def test_password_from_environment_variable(monkeypatch):
    monkeypatch.setenv("SHOP_DB_PW", "fromenv")
    assert D.DBConfig("mysql", password_env="SHOP_DB_PW").secret() == "fromenv"
    with pytest.raises(D.ConnectorError, match="not set"):
        D.DBConfig("mysql", password_env="NOPE_NOT_SET_X").secret()


def test_workspace_ingests_a_database_table_as_an_immutable_dataset(sqlite_db, tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "data"))
    ws = Workspace()
    with ws.connect_database(D.DBConfig("sqlite", database=sqlite_db)) as c:
        h, res = ws.ingest_database(c, table="orders", limit=100)
    assert h.source == "database" and len(h.raw) == 100 and res.truncated
    assert ws.repo.get_dataset(h.dataset_id)["source"] == "database"


# ---- web API -----------------------------------------------------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    seen: list = []
    fail_first = {"n": 0}

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        _Handler.seen.append({"path": u.path, "q": q, "auth": self.headers.get("Authorization"), "key": self.headers.get("X-API-Key")})
        if u.path == "/items":
            page, per = int(q.get("page", 1)), int(q.get("per_page", 10))
            items = [{"id": i, "name": f"n{i}", "meta": {"score": i / 10}} for i in range(1, 26)]
            self._send(200, {"data": {"items": items[(page - 1) * per: page * per]}})
        elif u.path == "/cursor":
            cur = int(q.get("cursor", 0))
            self._send(200, {"results": [{"v": cur * 10 + j} for j in range(10)], "next_cursor": cur + 1 if cur < 2 else None})
        elif u.path == "/link":
            n = int(q.get("n", 1))
            hdr = {"Link": f'<http://{self.headers["Host"]}/link?n={n + 1}>; rel="next"'} if n < 3 else {}
            self._send(200, [{"n": n}], headers=hdr)
        elif u.path == "/elsewhere":
            self._send(200, [{"a": 1}], headers={"Link": '<http://example.com/steal>; rel="next"'})
        elif u.path == "/csv":
            self._send(200, b"a,b\n1,2\n3,4\n", "text/csv")
        elif u.path == "/flaky":
            _Handler.fail_first["n"] += 1
            if _Handler.fail_first["n"] <= 2:
                self._send(503, {"error": "busy"})
            else:
                self._send(200, [{"ok": True}])
        elif u.path == "/redirect-out":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
        elif u.path == "/huge":
            self._send(200, b"[" + b'{"x": 1},' * 10 + b'{"x": 1}]', headers={})
        elif u.path == "/denied":
            self._send(401, {"error": "no"})
        else:
            self._send(404, {"error": "missing"})


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Handler.seen.clear()
    _Handler.fail_first["n"] = 0
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("url,msg", [("ftp://example.com/x", "Only http"), ("file:///etc/passwd", "Only http"), ("http://127.0.0.1:8080/", "internal"),
                                     ("http://localhost/", "internal"), ("http://169.254.169.254/latest/meta-data/", "internal"),
                                     ("http://10.1.2.3/", "internal"), ("http://[::1]/", "internal"), ("http://user:pw@example.com/", "credentials"),
                                     ("http:///nohost", "no host")])
def test_unsafe_urls_are_refused(url, msg):
    with pytest.raises(W.ConnectorError, match=msg):
        W.check_url(url)


def test_dns_names_pointing_inside_are_refused():
    with pytest.raises(W.ConnectorError, match="internal"):
        W.check_url("http://innocent.example/", resolver=lambda h, p: ["192.168.0.5"])
    assert W.check_url("https://api.example/", resolver=lambda h, p: ["93.184.216.34"]) == "api.example"


def test_credentials_require_https_for_public_hosts():
    with pytest.raises(W.ConnectorError, match="https"):
        W.APIClient(W.APIConfig("http://api.example/x", auth="bearer", secret="tok"), resolver=lambda h, p: ["93.184.216.34"])
    with pytest.raises(W.ConnectorError, match="authentication fields"):
        W.APIClient(W.APIConfig("https://api.example/x", headers={"Authorization": "Bearer x"}), resolver=lambda h, p: ["93.184.216.34"])


def test_page_pagination_nested_records_and_secret_header(server):
    cfg = W.APIConfig(f"{server}/items", auth="bearer", secret="tok-123", records_path="data.items", pagination="page", page_size=10, max_pages=10)
    r = W.fetch_api(cfg, allow_private=True)
    assert len(r.frame) == 25 and r.pages == 3 and "meta.score" in r.frame.columns
    assert all(s["auth"] == "Bearer tok-123" for s in _Handler.seen)
    assert "tok-123" not in repr(cfg) and "?" not in r.source


def test_cursor_link_csv_and_page_limit(server):
    r = W.fetch_api(W.APIConfig(f"{server}/cursor", pagination="cursor", records_path="results"), allow_private=True)
    assert list(r.frame["v"]) == list(range(30))
    r = W.fetch_api(W.APIConfig(f"{server}/link", pagination="link"), allow_private=True)
    assert list(r.frame["n"]) == [1, 2, 3]
    r = W.fetch_api(W.APIConfig(f"{server}/link", pagination="link", max_pages=2), allow_private=True)
    assert r.truncated and len(r.frame) == 2 and any("page limit" in n for n in r.notes)
    r = W.fetch_api(W.APIConfig(f"{server}/csv"), allow_private=True)
    assert r.frame.to_dict("list") == {"a": [1, 3], "b": [2, 4]}


def test_pagination_never_follows_credentials_to_another_host(server):
    r = W.fetch_api(W.APIConfig(f"{server}/elsewhere", auth="api_key_header", secret="k-1", pagination="link"), allow_private=True)
    assert len(r.frame) == 1 and any("different host" in n for n in r.notes)


def test_redirect_to_metadata_address_is_blocked_even_from_an_allowed_start(server, monkeypatch):
    orig = W.check_url

    def strict_after_first(url, **kw):           # the start host is local (allowed in tests); anything else gets the real checks
        if url.startswith(server):
            return orig(url, allow_private=True)
        return orig(url, allow_private=False)
    monkeypatch.setattr(W, "check_url", strict_after_first)
    with pytest.raises(W.ConnectorError, match="internal"):
        W.fetch_api(W.APIConfig(f"{server}/redirect-out"), allow_private=True)


def test_retries_with_backoff_then_success_and_bounded_failure(server):
    waits = []
    r = W.APIClient(W.APIConfig(f"{server}/flaky", retries=2), allow_private=True, sleep=waits.append).fetch()
    assert r.frame.iloc[0]["ok"] and waits == [1, 2]
    _Handler.fail_first["n"] = 0
    with pytest.raises(W.ConnectorError, match="kept failing"):
        W.APIClient(W.APIConfig(f"{server}/flaky", retries=1), allow_private=True, sleep=lambda s: None).fetch()


def test_size_limit_and_errors(server, monkeypatch):
    monkeypatch.setattr(W, "MAX_RESPONSE_BYTES", 50)
    with pytest.raises(W.ConnectorError, match="larger than"):
        W.fetch_api(W.APIConfig(f"{server}/huge"), allow_private=True)
    with pytest.raises(W.ConnectorError, match="refused access"):
        W.fetch_api(W.APIConfig(f"{server}/denied"), allow_private=True)


def test_secrets_are_not_logged(server, caplog):
    caplog.set_level(logging.DEBUG)
    W.fetch_api(W.APIConfig(f"{server}/items", auth="api_key_query", key_name="api_key", secret="QSECRET9", records_path="data.items"), allow_private=True)
    assert _Handler.seen[0]["q"]["api_key"] == "QSECRET9"              # it was sent ...
    assert "QSECRET9" not in caplog.text                                 # ... but never logged


def test_workspace_api_ingestion_respects_the_private_host_setting(server, tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "d"))
    with pytest.raises(W.ConnectorError, match="internal"):
        Workspace().ingest_api(W.APIConfig(f"{server}/csv"))
    monkeypatch.setenv("NEXORA_ALLOW_PRIVATE_API", "1")
    h, res = Workspace().ingest_api(W.APIConfig(f"{server}/csv"), name="csv feed")
    assert h.source == "web_api" and h.raw.shape == (2, 2)
