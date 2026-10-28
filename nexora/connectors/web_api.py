"""Web API ingestion: fetch JSON or CSV from an HTTP(S) endpoint into a table.

Safety model
* Only http and https URLs. Credentials are only sent over https.
* Server-side request forgery protection: the host is resolved and every address is checked; loopback, private,
  link-local (including cloud metadata 169.254.169.254), multicast, reserved and unspecified addresses are refused
  unless the operator explicitly allows private hosts (NEXORA_ALLOW_PRIVATE_API=1). Redirects are followed
  manually (at most 3) and re-checked; credentials are dropped if a redirect or a pagination link leaves the
  original host.
* Secrets (bearer tokens, API keys, basic-auth passwords) are kept in memory, never logged, and redacted from
  error messages. Requests are logged without headers or query strings.
* Timeouts on every request, bounded retries with exponential backoff (429 / 5xx / connection errors only),
  a per-response and a total size limit, and a hard page limit for pagination.
"""
from __future__ import annotations

import io
import ipaddress
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse

import pandas as pd

from .database import ConnectorError, redact

log = logging.getLogger(__name__)
# urllib3 logs full request lines (including query strings that may carry API keys) at DEBUG level; keep it quiet.
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

AUTH = {"none": "No authentication", "bearer": "Bearer token", "api_key_header": "API key in a header", "api_key_query": "API key in the URL query",
        "basic": "Username and password (basic)"}
PAGINATION = {"none": "Single request", "page": "Page number parameter", "offset": "Offset / limit parameters", "cursor": "Cursor from the response",
              "link": "Next link (Link header or a field in the response)"}
MAX_PAGES = 50
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_RETRIES = 3
MAX_REDIRECTS = 3


@dataclass
class APIConfig:
    url: str
    auth: str = "none"
    secret: str = field(default="", repr=False)           # token, API key or password
    username: str = ""
    key_name: str = "X-API-Key"                            # header or query parameter name for API keys
    headers: dict[str, str] = field(default_factory=dict)  # extra non-secret headers
    params: dict[str, str] = field(default_factory=dict)
    records_path: str = ""                                 # dot path to the list of records in the JSON, e.g. "data.items"
    pagination: str = "none"
    page_param: str = "page"
    size_param: str = "per_page"
    page_size: int = 100
    start_page: int = 1
    offset_param: str = "offset"
    limit_param: str = "limit"
    cursor_path: str = "next_cursor"                      # where the next cursor is in the JSON
    cursor_param: str = "cursor"
    next_path: str = ""                                    # for "link": a JSON field holding the next URL (empty = Link header)
    max_pages: int = 10
    max_rows: int = 100_000
    timeout_s: float = 20.0
    retries: int = 2
    response_format: str = "auto"                          # auto | json | csv

    def __repr__(self) -> str:
        return f"APIConfig(url={_safe_url(self.url)!r}, auth={self.auth!r}, pagination={self.pagination!r})"


@dataclass
class FetchResult:
    frame: pd.DataFrame
    pages: int
    requests: int
    bytes: int
    truncated: bool
    notes: list[str]
    source: str

    def note(self) -> str:
        return f"{len(self.frame):,} rows from {self.pages} page(s), {self.bytes / 1024:,.0f} KB" + (" (stopped at a limit)" if self.truncated else "")


def _safe_url(url: str) -> str:
    """The URL without query string or credentials, for display and logs."""
    try:
        p = urlparse(url)
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        return urlunparse((p.scheme, host + port, p.path, "", "", ""))
    except Exception:
        return "(invalid URL)"


def _blocked(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified or \
        (isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.ip_network("100.64.0.0/10"))


def check_url(url: str, *, allow_private: bool = False, needs_https: bool = False,
              resolver: Callable[[str, int], list[str]] | None = None) -> str:
    """Validate scheme and target address. Returns the hostname. Raises ConnectorError if unsafe."""
    try:
        p = urlparse(url.strip())
    except Exception:
        raise ConnectorError("That is not a valid URL.") from None
    if p.scheme not in ("http", "https"):
        raise ConnectorError("Only http:// and https:// URLs can be used.")
    if not p.hostname:
        raise ConnectorError("The URL has no host name.")
    if p.username or p.password:
        raise ConnectorError("Do not put credentials in the URL; use the authentication fields instead.")
    if needs_https and p.scheme != "https":
        raise ConnectorError("Credentials are only sent over https. Use an https:// URL.")
    host = p.hostname
    port = p.port or (443 if p.scheme == "https" else 80)
    if allow_private:
        return host
    try:
        addrs = resolver(host, port) if resolver else [ai[4][0] for ai in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)]
    except socket.gaierror:
        raise ConnectorError(f"The host '{host}' could not be found.") from None
    if not addrs:
        raise ConnectorError(f"The host '{host}' could not be found.")
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if _blocked(ip):
            raise ConnectorError(f"'{host}' points to a private or internal address ({ip}). Internal addresses are blocked to protect "
                                 "the server; an administrator can allow them with NEXORA_ALLOW_PRIVATE_API=1.")
    return host


def _dig(obj: Any, path: str) -> Any:
    if not path:
        return obj
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            return None
    return obj


def _records(payload: Any, path: str) -> list[dict[str, Any]]:
    data = _dig(payload, path)
    if data is None:
        raise ConnectorError(f"'{path}' was not found in the response." if path else "The response is empty.")
    if isinstance(data, dict):
        lists = [k for k, v in data.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if not path and len(lists) == 1:
            data = data[lists[0]]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ConnectorError("The response does not contain a list of records. Set the path to the records (for example data.items).")
    return [r if isinstance(r, dict) else {"value": r} for r in data]


class APIClient:
    def __init__(self, cfg: APIConfig, *, allow_private: bool = False, session=None, resolver=None, sleep: Callable[[float], None] = time.sleep):
        import requests
        if cfg.auth not in AUTH:
            raise ConnectorError(f"Unknown authentication type '{cfg.auth}'.")
        if cfg.pagination not in PAGINATION:
            raise ConnectorError(f"Unknown pagination type '{cfg.pagination}'.")
        if not 1 <= int(cfg.max_pages) <= MAX_PAGES:
            raise ConnectorError(f"The page limit must be between 1 and {MAX_PAGES}.")
        if not 0 <= int(cfg.retries) <= MAX_RETRIES:
            raise ConnectorError(f"Retries must be between 0 and {MAX_RETRIES}.")
        if not 1 <= float(cfg.timeout_s) <= 120:
            raise ConnectorError("The timeout must be between 1 and 120 seconds.")
        if cfg.auth != "none" and not cfg.secret:
            raise ConnectorError("Enter the token, key or password for the chosen authentication.")
        for h in cfg.headers:
            if h.lower() in ("authorization", "cookie", "proxy-authorization") or h.lower() == cfg.key_name.lower():
                raise ConnectorError(f"Put credentials in the authentication fields, not in the '{h}' header, so they are protected.")
        self.cfg = cfg
        self.allow_private = allow_private
        self.resolver = resolver
        self.sleep = sleep
        self.session = session or requests.Session()
        host = urlparse(cfg.url.strip()).hostname or ""
        self.home = check_url(cfg.url, allow_private=allow_private, needs_https=cfg.auth != "none" and not (allow_private and _is_local(host)),
                              resolver=resolver)
        self.requests = 0
        self.total = 0

    # ---- one request with retries, manual redirects and size limits -------------------------------------
    def _auth_for(self, url: str, host: str) -> tuple[dict[str, str], dict[str, str], Any]:
        secure = urlparse(url).scheme == "https" or (self.allow_private and _is_local(host))
        if host != self.home or self.cfg.auth == "none" or not secure:      # never to another host, never in clear text
            return {}, {}, None
        c = self.cfg
        if c.auth == "bearer":
            return {"Authorization": f"Bearer {c.secret}"}, {}, None
        if c.auth == "api_key_header":
            return {c.key_name: c.secret}, {}, None
        if c.auth == "api_key_query":
            return {}, {c.key_name: c.secret}, None
        return {}, {}, (c.username, c.secret)

    def get(self, url: str, params: dict[str, Any] | None = None) -> tuple[bytes, str, dict[str, str], str]:
        import requests
        attempt = 0
        while True:
            try:
                return self._get_once(url, params)
            except _Retryable as e:
                attempt += 1
                if attempt > self.cfg.retries:
                    raise ConnectorError(f"The API kept failing ({e}). Tried {attempt} time(s).") from None
                wait = min(2 ** (attempt - 1), 8) if e.retry_after is None else min(e.retry_after, 30)
                log.info("Retrying %s in %.1fs (%s)", _safe_url(url), wait, e)
                self.sleep(wait)
            except requests.RequestException as e:
                raise ConnectorError(redact(f"The request failed: {type(e).__name__}", self.cfg.secret)) from None

    def _get_once(self, url: str, params: dict[str, Any] | None) -> tuple[bytes, str, dict[str, str], str]:
        import requests
        current, cur_params = url, dict(params or {})
        for _ in range(MAX_REDIRECTS + 1):
            host = check_url(current, allow_private=self.allow_private, resolver=self.resolver)
            h_auth, q_auth, basic = self._auth_for(current, host)
            headers = {"Accept": "application/json, text/csv;q=0.9, */*;q=0.5", "User-Agent": "Nexora/1.0", **self.cfg.headers, **h_auth}
            self.requests += 1
            log.info("GET %s", _safe_url(current))                      # never the headers or query (they may hold keys)
            try:
                r = self.session.get(current, params={**cur_params, **q_auth}, headers=headers, auth=basic, timeout=float(self.cfg.timeout_s),
                                     allow_redirects=False, stream=True)
            except (requests.ConnectionError, requests.Timeout) as e:
                raise _Retryable(type(e).__name__) from None
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                current, cur_params = urljoin(current, r.headers["Location"]), {}
                r.close()
                continue
            if r.status_code == 429 or 500 <= r.status_code < 600:
                ra = r.headers.get("Retry-After")
                r.close()
                raise _Retryable(f"HTTP {r.status_code}", float(ra) if ra and ra.isdigit() else None)
            if r.status_code in (401, 403):
                r.close()
                raise ConnectorError(f"The API refused access (HTTP {r.status_code}). Check the credentials and permissions.")
            if r.status_code >= 400:
                r.close()
                raise ConnectorError(f"The API returned HTTP {r.status_code} for {_safe_url(current)}.")
            declared = r.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
                r.close()
                raise ConnectorError(f"The response is larger than the {MAX_RESPONSE_BYTES // 2**20} MB limit.")
            buf = io.BytesIO()
            for chunk in r.iter_content(65536):
                buf.write(chunk)
                if buf.tell() > MAX_RESPONSE_BYTES:
                    r.close()
                    raise ConnectorError(f"The response is larger than the {MAX_RESPONSE_BYTES // 2**20} MB limit.")
            self.total += buf.tell()
            if self.total > MAX_TOTAL_BYTES:
                raise ConnectorError(f"More than {MAX_TOTAL_BYTES // 2**20} MB were downloaded in total; stopping.")
            return buf.getvalue(), r.headers.get("Content-Type", ""), dict(r.headers), current
        raise ConnectorError(f"Too many redirects (more than {MAX_REDIRECTS}).")

    # ---- paginated fetch -------------------------------------------------------------------------------
    def fetch(self, progress: Callable[[str], None] | None = None) -> FetchResult:
        c = self.cfg
        rows: list[dict[str, Any]] = []
        notes: list[str] = []
        url, params = c.url, dict(c.params)
        page, offset, pages, truncated = int(c.start_page), 0, 0, False
        seen_cursors: set[str] = set()
        while True:
            p = dict(params)
            if c.pagination == "page":
                p.update({c.page_param: page, c.size_param: c.page_size})
            elif c.pagination == "offset":
                p.update({c.offset_param: offset, c.limit_param: c.page_size})
            if progress:
                progress(f"page {pages + 1}")
            body, ctype, headers, final_url = self.get(url, p)
            pages += 1
            fmt = c.response_format if c.response_format != "auto" else ("csv" if "csv" in ctype.lower() else "json")
            if fmt == "csv":
                try:
                    df = pd.read_csv(io.BytesIO(body))
                except Exception as e:
                    raise ConnectorError(f"The response could not be read as CSV: {type(e).__name__}.") from None
                batch, payload = df.to_dict("records"), None
            else:
                try:
                    payload = json.loads(body.decode("utf-8-sig") or "null")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ConnectorError("The response is not valid JSON. If it is CSV, set the format to CSV.") from None
                batch = _records(payload, c.records_path)
            rows.extend(batch)
            if len(rows) >= c.max_rows:
                rows, truncated = rows[: c.max_rows], True
                notes.append(f"Stopped at the row limit of {c.max_rows:,}.")
                break
            nxt_url = None
            if c.pagination == "none" or not batch:
                break
            if c.pagination == "page":
                page += 1
            elif c.pagination == "offset":
                offset += len(batch)
            elif c.pagination == "cursor":
                cur = _dig(payload, c.cursor_path) if payload is not None else None
                if not cur or str(cur) in seen_cursors:
                    break
                seen_cursors.add(str(cur))
                params = {**params, c.cursor_param: cur}
            elif c.pagination == "link":
                nxt_url = _dig(payload, c.next_path) if c.next_path and payload is not None else _link_next(headers.get("Link", ""))
                if not nxt_url:
                    break
                nxt_url = urljoin(final_url, str(nxt_url))
                if urlparse(nxt_url).hostname != self.home:
                    notes.append("The next-page link pointed to a different host, so pagination stopped there (credentials are never sent elsewhere).")
                    break
                q = dict(parse_qsl(urlparse(nxt_url).query))
                url, params = urlunparse(urlparse(nxt_url)._replace(query="")), q
            if c.pagination in ("page", "offset") and len(batch) < c.page_size:
                break
            if pages >= c.max_pages:
                truncated = True
                notes.append(f"Stopped after the page limit of {c.max_pages}; there may be more data.")
                break
        if not rows:
            raise ConnectorError("The API returned no records.")
        df = pd.json_normalize(rows, sep=".") if any(isinstance(v, dict) for r in rows[:50] for v in r.values()) else pd.DataFrame(rows)
        for col in df.columns:                                       # lists in cells cannot be analysed; keep them as text
            if df[col].map(lambda v: isinstance(v, (list, dict))).any():
                df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
                notes.append(f"'{col}' holds nested lists; it was kept as JSON text.")
        return FetchResult(df, pages, self.requests, self.total, truncated, notes, _safe_url(c.url))


class _Retryable(Exception):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _link_next(link_header: str) -> str | None:
    for part in link_header.split(","):
        seg = part.split(";")
        if len(seg) >= 2 and any(s.strip().replace(" ", "") in ('rel="next"', "rel=next") for s in seg[1:]):
            return seg[0].strip().strip("<>")
    return None


def fetch_api(cfg: APIConfig, *, allow_private: bool = False, **kw) -> FetchResult:
    try:
        return APIClient(cfg, allow_private=allow_private, **kw).fetch()
    except ConnectorError as e:
        raise ConnectorError(redact(str(e), cfg.secret)) from None


def _is_local(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
