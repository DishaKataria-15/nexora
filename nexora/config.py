"""Environment-based configuration. Nothing here is secret."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
APP_NAME = "Nexora"
TAGLINE = "Connect data, models and decisions."


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Ignoring non-integer value for %s", name)
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_upload_mb: int
    max_rows: int
    max_cols: int
    default_seed: int
    profiles_dir: Path
    sample_dir: Path
    log_level: str
    allow_private_api_hosts: bool = False     # web API ingestion may reach loopback / private addresses only when this is on
    db_timeout_s: int = 30                    # connection and query timeout for database connectors
    db_row_limit: int = 100_000               # default cap on rows read from a database in one go
    public_mode: bool = False                 # hosted for many visitors: one private workspace per browser session, local-only tabs off

    @property
    def db_path(self) -> str:
        return str(self.data_dir / "nexora.db")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    """Read settings from environment variables (called fresh each time)."""
    data_dir = Path(os.environ.get("NEXORA_DATA_DIR") or (PROJECT_ROOT / "data"))
    return Settings(
        data_dir=data_dir,
        max_upload_mb=_int_env("NEXORA_MAX_UPLOAD_MB", 50),
        max_rows=_int_env("NEXORA_MAX_ROWS", 300_000),
        max_cols=_int_env("NEXORA_MAX_COLS", 300),
        default_seed=_int_env("NEXORA_SEED", 42),
        profiles_dir=PROJECT_ROOT / "config" / "domains",
        sample_dir=PROJECT_ROOT / "sample_data" / "synthetic",
        log_level=os.environ.get("NEXORA_LOG_LEVEL", "INFO").upper(),
        allow_private_api_hosts=os.environ.get("NEXORA_ALLOW_PRIVATE_API", "0").strip().lower() in ("1", "true", "yes"),
        db_timeout_s=_int_env("NEXORA_DB_TIMEOUT_S", 30),
        db_row_limit=_int_env("NEXORA_DB_ROW_LIMIT", 100_000),
        public_mode=_public_mode(),
    )


def _public_mode() -> bool:
    """On by NEXORA_PUBLIC=1, and automatically on Streamlit Community Cloud (apps run from /mount/src there)."""
    raw = os.environ.get("NEXORA_PUBLIC")
    if raw is not None and raw.strip():
        return raw.strip().lower() in ("1", "true", "yes")
    return str(PROJECT_ROOT).replace("\\", "/").startswith("/mount/src")


def configure_logging(level: str = "INFO") -> None:
    """Basic logging. Messages must never include raw cell values."""
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # HTTP client libraries log full request URLs at DEBUG level; those can contain API keys.
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
