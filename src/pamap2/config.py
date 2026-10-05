"""Single source of truth for project resource names (persisted to config/project_config.json)."""
from __future__ import annotations

import json
from pathlib import Path

REGION = "us-east-1"
PREFIX = "pamap2-har"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "project_config.json"
S3_ZONES = ("raw", "curated", "athena-results", "features", "models", "monitoring")


def build_config(account_id: str, region: str = REGION, prefix: str = PREFIX) -> dict:
    """Derive every resource name from prefix + account id + region."""
    return {
        "region": region,
        "prefix": prefix,
        "account_id": account_id,
        "bucket": f"{prefix}-{account_id}-{region}",
        "glue_database": prefix.replace("-", "_"),
        "athena_workgroup": f"{prefix}-wg",
        "role_arn": f"arn:aws:iam::{account_id}:role/LabRole",
        "s3_prefixes": {z: f"{z}/" for z in S3_ZONES},
        "raw_table": "raw_pamap2",
        "curated_table": "curated_pamap2",
    }


def save_config(cfg: dict, path: Path | str = CONFIG_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run notebooks/1.0_setup.ipynb first.")
    return json.loads(path.read_text())


def s3_uri(cfg: dict, zone: str, key: str = "") -> str:
    return f"s3://{cfg['bucket']}/{cfg['s3_prefixes'][zone]}{key}"
