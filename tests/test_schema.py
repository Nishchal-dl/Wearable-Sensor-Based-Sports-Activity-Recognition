import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pamap2 import schema
from src.pamap2.config import build_config


def _row(t, act):
    vals = ["NaN" if i == 2 else str(round(0.1 * i, 2)) for i in range(54)]
    vals[0], vals[1] = str(t), str(act)
    return " ".join(vals)


def test_column_count():
    assert len(schema.COLUMNS) == 54 and len(set(schema.COLUMNS)) == 54


def test_invalid_removed():
    assert len(schema.INVALID_COLUMNS) == 12
    assert not set(schema.INVALID_COLUMNS) & set(schema.CURATED_COLUMNS)
    assert len(schema.CURATED_COLUMNS) == 42


def test_parser_synthetic():
    buf = io.StringIO("\n".join([_row(8.38, 0), _row(8.39, 1)]))
    df = schema.parse_dat(buf, subject_id=101, session="protocol")
    assert len(df) == 2
    assert df["heart_rate"].isna().all()
    assert not any(c in df.columns for c in schema.INVALID_COLUMNS)
    assert df["subject_id"].iloc[0] == 101 and df["session"].iloc[0] == "protocol"
    assert df["activity_id"].tolist() == [0, 1]
    assert len(df.columns) == 42 + 2


def test_config_names():
    cfg = build_config("123456789012")
    assert cfg["bucket"] == "pamap2-har-123456789012-us-east-1"
    assert cfg["role_arn"].endswith(":role/LabRole")


def test_wrong_column_count():
    import pytest
    with pytest.raises(ValueError):
        schema.parse_dat(io.StringIO("1 2 3 4 5"))
    with pytest.raises(ValueError):
        schema.parse_dat(io.StringIO(_row(1, 1) + " 9"))


def test_nan_activity_rows_dropped(capsys):
    bad = _row(8.5, 1).split()
    bad[1] = "NaN"
    df = schema.parse_dat(io.StringIO("\n".join([_row(8.38, 1), " ".join(bad)])))
    assert len(df) == 1
    assert "dropping 1" in capsys.readouterr().out


def test_config_helpers(tmp_path):
    import pytest
    from src.pamap2 import config
    cfg = config.build_config("123456789012")
    assert cfg["glue_database"] == "pamap2_har"
    assert cfg["s3_prefixes"]["curated"] == "curated/"
    assert config.s3_uri(cfg, "raw", "pamap2/") == f"s3://{cfg['bucket']}/raw/pamap2/"
    p = tmp_path / "sub" / "c.json"
    config.save_config(cfg, p)
    assert config.load_config(p) == cfg
    with pytest.raises(FileNotFoundError):
        config.load_config(tmp_path / "missing.json")
