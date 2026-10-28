"""Read-only database connectors (SQLite, PostgreSQL, MySQL / MariaDB, SQL Server) through SQLAlchemy.

Safety model
* Read-only by default and by construction: SQLite is opened with ``mode=ro``; PostgreSQL sessions are set to
  ``default_transaction_read_only``; MySQL sessions to ``TRANSACTION READ ONLY``; SQL Server connections declare
  ``ApplicationIntent=ReadOnly``. There is no write mode in this module. For full protection, connect with a
  database user that only has SELECT permission: that is the guarantee the database itself enforces.
* Table and column names are never pasted into SQL: they are checked against the database's own catalogue and the
  query is built with SQLAlchemy constructs, which quote identifiers.
* Free-form SQL is limited to one SELECT (or WITH ... SELECT) statement, checked before it is sent; values are
  passed as bound parameters (``:name``), never concatenated.
* Passwords live only in memory for the session. They can come from an environment variable so they never appear
  in a file. Connection descriptions, logs and error messages are redacted.
* Every read has a row limit; reading a whole table needs an explicit confirmation flag; connections and queries
  have timeouts.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

DIALECTS = {
    "sqlite": {"label": "SQLite file", "driver": "sqlite", "package": None, "port": None},
    "postgresql": {"label": "PostgreSQL", "driver": "postgresql+psycopg2", "package": "psycopg2", "alt": ("psycopg", "postgresql+psycopg"), "port": 5432},
    "mysql": {"label": "MySQL / MariaDB", "driver": "mysql+pymysql", "package": "pymysql", "port": 3306},
    "mssql": {"label": "Microsoft SQL Server", "driver": "mssql+pyodbc", "package": "pyodbc", "port": 1433},
}
INSTALL = {"psycopg2": "pip install psycopg2-binary", "pymysql": "pip install pymysql", "pyodbc": "pip install pyodbc (plus the Microsoft ODBC driver)",
           "sqlalchemy": "pip install sqlalchemy"}
SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None
MAX_ROW_LIMIT = 1_000_000
PREVIEW_ROWS = 200

FORBIDDEN = {"insert", "update", "delete", "merge", "upsert", "drop", "alter", "create", "truncate", "rename", "grant", "revoke",
             "exec", "execute", "call", "copy", "into", "attach", "detach", "pragma", "vacuum", "reindex", "lock",
             "unlock", "set", "reset", "begin", "commit", "rollback", "savepoint", "handler", "outfile", "dumpfile", "shutdown",
             "kill", "listen", "notify", "prepare", "deallocate", "declare", "bulk", "openrowset", "opendatasource", "xp_cmdshell", "dbcc", "backup", "restore", "sp_executesql",
             "load_extension", "pg_terminate_backend", "pg_cancel_backend", "pg_read_file", "pg_read_binary_file", "lo_import", "lo_export",
             "dblink", "dblink_exec", "nextval", "setval", "sleep", "pg_sleep", "benchmark", "waitfor"}


class ConnectorError(ValueError):
    """A connector problem. Messages are redacted and safe to show to the user."""


@dataclass
class DBConfig:
    dialect: str
    database: str = ""                 # database name, or the file path for SQLite
    host: str = ""
    port: int | None = None
    username: str = ""
    password: str = field(default="", repr=False)
    password_env: str = ""             # read the password from this environment variable instead
    schema: str | None = None
    timeout_s: int = 30
    odbc_driver: str = "ODBC Driver 18 for SQL Server"

    def __repr__(self) -> str:         # never print the password
        return f"DBConfig(dialect={self.dialect!r}, host={self.host!r}, database={self.database!r}, username={self.username!r})"

    def secret(self) -> str:
        if self.password_env:
            val = os.environ.get(self.password_env)
            if val is None:
                raise ConnectorError(f"The environment variable '{self.password_env}' is not set.")
            return val
        return self.password

    def describe(self) -> str:
        """A safe, password-free description for display and dataset metadata."""
        if self.dialect == "sqlite":
            return f"SQLite file {os.path.basename(self.database)}"
        who = f"{self.username}@" if self.username else ""
        port = f":{self.port}" if self.port else ""
        return f"{DIALECTS[self.dialect]['label']} {who}{self.host}{port}/{self.database}"


def redact(text: str, *secrets: str) -> str:
    out = str(text)
    for s in secrets:
        if s and len(s) >= 1:
            out = out.replace(s, "***")
    out = re.sub(r"(://[^:/@\s]+:)[^@\s]+@", r"\1***@", out)                          # user:password@ in URLs
    out = re.sub(r"(?i)(password|pwd)\s*=\s*[^;\s]+", r"\1=***", out)                  # ODBC-style key=value
    return out


def driver_status(dialect: str) -> tuple[bool, str]:
    if not SQLALCHEMY_AVAILABLE:
        return False, f"Database connections need SQLAlchemy ({INSTALL['sqlalchemy']})."
    d = DIALECTS.get(dialect)
    if d is None:
        return False, f"Unknown database type '{dialect}'."
    pkg = d["package"]
    if pkg is None or importlib.util.find_spec(pkg) is not None:
        return True, ""
    alt = d.get("alt")
    if alt and importlib.util.find_spec(alt[0]) is not None:
        return True, ""
    return False, f"{d['label']} needs the driver package '{pkg}' ({INSTALL.get(pkg, 'pip install ' + pkg)})."


def _driver_name(dialect: str) -> str:
    d = DIALECTS[dialect]
    if d["package"] and importlib.util.find_spec(d["package"]) is None and d.get("alt") and importlib.util.find_spec(d["alt"][0]) is not None:
        return d["alt"][1]
    return d["driver"]


def create_readonly_engine(cfg: DBConfig):
    """Build a SQLAlchemy engine whose connections are read-only."""
    ok, why = driver_status(cfg.dialect)
    if not ok:
        raise ConnectorError(why)
    from sqlalchemy import create_engine, event
    from sqlalchemy.engine import URL
    t = max(1, min(int(cfg.timeout_s), 600))
    secret = cfg.secret() if cfg.dialect != "sqlite" else ""
    if cfg.dialect == "sqlite":
        path = os.path.abspath(cfg.database)
        if not cfg.database or not os.path.isfile(path):
            raise ConnectorError("The SQLite file was not found.")
        import sqlite3
        from urllib.parse import quote

        def _connect():
            return sqlite3.connect(f"file:{quote(path)}?mode=ro", uri=True, timeout=t, check_same_thread=False)
        eng = create_engine("sqlite://", creator=_connect)
        return eng
    if not cfg.host or not cfg.database:
        raise ConnectorError("Enter the host and the database name.")
    port = cfg.port or DIALECTS[cfg.dialect]["port"]
    query: dict[str, Any] = {}
    args: dict[str, Any] = {}
    if cfg.dialect == "postgresql":
        args = {"connect_timeout": t, "options": f"-c default_transaction_read_only=on -c statement_timeout={t * 1000}"}
    elif cfg.dialect == "mysql":
        args = {"connect_timeout": t, "read_timeout": t}
    elif cfg.dialect == "mssql":
        query = {"driver": cfg.odbc_driver, "ApplicationIntent": "ReadOnly"}
        args = {"timeout": t}
    url = URL.create(_driver_name(cfg.dialect), username=cfg.username or None, password=secret or None, host=cfg.host, port=port,
                     database=cfg.database, query=query)
    eng = create_engine(url, connect_args=args, pool_pre_ping=True, hide_parameters=True)
    if cfg.dialect == "mysql":
        @event.listens_for(eng, "connect")
        def _ro(dbapi_conn, _rec):
            cur = dbapi_conn.cursor()
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute(f"SET SESSION MAX_EXECUTION_TIME={t * 1000}")
            cur.close()
    return eng


def _wrap(e: Exception, cfg: DBConfig) -> ConnectorError:
    try:
        secret = cfg.secret()
    except ConnectorError:
        secret = ""
    msg = getattr(e, "orig", None) or e
    text = redact(str(msg).strip().splitlines()[0] if str(msg).strip() else type(e).__name__, secret, cfg.password)
    log.info("Database error (%s): %s", cfg.describe(), text)
    return ConnectorError(f"The database reported an error: {text}")


class DatabaseConnection:
    """A read-only session against one database. Use as a context manager or call close()."""

    def __init__(self, cfg: DBConfig):
        self.cfg = cfg
        try:
            self.engine = create_readonly_engine(cfg)
            with self.engine.connect() as c:
                from sqlalchemy import text
                c.execute(text("SELECT 1"))
        except ConnectorError:
            raise
        except Exception as e:
            raise _wrap(e, cfg) from None

    def __enter__(self) -> "DatabaseConnection":
        return self

    def __exit__(self, *a) -> None:
        self.close()

    def close(self) -> None:
        self.engine.dispose()

    # ---- catalogue ----------------------------------------------------------------------------------
    def list_tables(self) -> list[dict[str, str]]:
        from sqlalchemy import inspect
        try:
            ins = inspect(self.engine)
            out = [{"schema": self.cfg.schema or "", "name": t, "kind": "table"} for t in ins.get_table_names(schema=self.cfg.schema)]
            out += [{"schema": self.cfg.schema or "", "name": v, "kind": "view"} for v in ins.get_view_names(schema=self.cfg.schema)]
            return sorted(out, key=lambda r: (r["kind"], r["name"]))
        except Exception as e:
            raise _wrap(e, self.cfg) from None

    def list_columns(self, table: str) -> list[dict[str, str]]:
        self._known(table)
        from sqlalchemy import inspect
        try:
            return [{"name": c["name"], "type": str(c["type"])} for c in inspect(self.engine).get_columns(table, schema=self.cfg.schema)]
        except Exception as e:
            raise _wrap(e, self.cfg) from None

    def _known(self, table: str) -> None:
        names = {t["name"] for t in self.list_tables()}
        if table not in names:
            raise ConnectorError(f"'{table}' is not a table or view in this database.")

    def _table(self, table: str, columns: list[str] | None):
        from sqlalchemy import MetaData, Table, select
        self._known(table)
        try:
            tbl = Table(table, MetaData(), schema=self.cfg.schema, autoload_with=self.engine)
        except Exception as e:
            raise _wrap(e, self.cfg) from None
        if columns:
            unknown = [c for c in columns if c not in tbl.c]
            if unknown:
                raise ConnectorError(f"Unknown column(s) in '{table}': {', '.join(unknown)}.")
            return tbl, select(*[tbl.c[c] for c in columns])
        return tbl, select(tbl)

    def count_rows(self, table: str) -> int:
        from sqlalchemy import func, select
        tbl, _ = self._table(table, None)
        try:
            with self.engine.connect() as c:
                return int(c.execute(select(func.count()).select_from(tbl)).scalar_one())
        except Exception as e:
            raise _wrap(e, self.cfg) from None

    # ---- reads --------------------------------------------------------------------------------------
    def read_table(self, table: str, *, columns: list[str] | None = None, limit: int = PREVIEW_ROWS, full: bool = False,
                   confirm_full: bool = False) -> "ReadResult":
        """Read rows from a table. ``full=True`` reads every row and requires ``confirm_full=True``."""
        _, stmt = self._table(table, columns)
        if full:
            if not confirm_full:
                raise ConnectorError("Reading a whole table can be slow and expensive. Confirm the full read first.")
            cap = MAX_ROW_LIMIT
        else:
            cap = _check_limit(limit)
        return self._run(stmt, {}, cap, f"table {table}")

    def run_query(self, sql: str, params: dict[str, Any] | None = None, limit: int = PREVIEW_ROWS) -> "ReadResult":
        """Run one validated SELECT statement with bound parameters."""
        validate_select(sql)
        from sqlalchemy import text
        stmt = text(sql.strip().rstrip(";"))
        needed = set(re.findall(r"(?<![:\w]):([A-Za-z_]\w*)", _strip_literals(sql)))
        params = dict(params or {})
        missing = needed - set(params)
        if missing:
            raise ConnectorError(f"Give a value for the parameter(s): {', '.join(sorted(missing))}.")
        return self._run(stmt, {k: v for k, v in params.items() if k in needed}, _check_limit(limit), "query")

    def _run(self, stmt, params: dict[str, Any], cap: int, what: str) -> "ReadResult":
        t0 = time.time()
        try:
            with self.engine.connect() as c:
                if self.cfg.dialect == "sqlite":
                    raw = c.connection.dbapi_connection
                    deadline = t0 + max(1, int(self.cfg.timeout_s))
                    raw.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
                res = c.execution_options(stream_results=True).execute(stmt, params)
                cols = list(res.keys())
                rows = res.fetchmany(cap + 1)
                if self.cfg.dialect == "sqlite":
                    raw.set_progress_handler(None, 0)
        except Exception as e:
            if "interrupted" in str(e).lower():
                raise ConnectorError(f"The {what} took longer than {self.cfg.timeout_s} seconds and was stopped.") from None
            raise _wrap(e, self.cfg) from None
        truncated = len(rows) > cap
        df = pd.DataFrame([tuple(r) for r in rows[:cap]], columns=cols)
        return ReadResult(df, truncated, cap, round(time.time() - t0, 3), what)


@dataclass
class ReadResult:
    frame: pd.DataFrame
    truncated: bool
    limit: int
    seconds: float
    what: str

    def note(self) -> str:
        n = len(self.frame)
        return (f"{n:,} rows read in {self.seconds:.2f}s" + (f"; stopped at the limit of {self.limit:,} rows, so there are more rows than shown." if self.truncated else "."))


def _check_limit(limit: int) -> int:
    try:
        n = int(limit)
    except (TypeError, ValueError):
        raise ConnectorError("The row limit must be a whole number.") from None
    if not 1 <= n <= MAX_ROW_LIMIT:
        raise ConnectorError(f"The row limit must be between 1 and {MAX_ROW_LIMIT:,}.")
    return n


def _strip_literals(sql: str) -> str:
    """Remove comments and quoted strings / identifiers so keyword checks only see SQL structure."""
    s = re.sub(r"--[^\n]*", " ", sql)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"'(?:[^']|'')*'", "''", s)
    s = re.sub(r'"(?:[^"]|"")*"', '""', s)
    s = re.sub(r"`[^`]*`", "``", s)
    s = re.sub(r"\[[^\]]*\]", "[]", s)
    return s


def validate_select(sql: str) -> None:
    """Allow exactly one read-only SELECT / WITH statement. Raises ConnectorError otherwise."""
    if not sql or not sql.strip():
        raise ConnectorError("Enter a SELECT query.")
    if len(sql) > 20_000:
        raise ConnectorError("The query is too long (20,000 characters at most).")
    s = _strip_literals(sql)
    if "/*" in s or "*/" in s:
        raise ConnectorError("Unbalanced comment in the query.")
    body = s.strip().rstrip(";").strip()
    if ";" in body:
        raise ConnectorError("Only one statement can be run at a time.")
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body.lower())
    if not words or words[0] not in ("select", "with"):
        raise ConnectorError("Only SELECT queries can be run (read-only access).")
    bad = sorted(set(words) & FORBIDDEN)
    if bad:
        raise ConnectorError(f"The query contains {', '.join(w.upper() for w in bad)}, which is not allowed: only reading data is permitted. "
                              "If this is a column name, put it in double quotes.")
