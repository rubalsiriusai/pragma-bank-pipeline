"""Actionable-error and warning guards on the public API / ingest paths."""

from __future__ import annotations

import datetime as dt
import logging

import pytest

import pragmatiq.core.schema as schema
from pragmatiq import api
from pragmatiq.core.schema import UserRecord
from tests._real_data import real_sample


def test_load_yaml_rejects_non_mapping(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n")  # a top-level sequence, not a mapping
    with pytest.raises(ValueError, match="top-level mapping"):
        api._load_yaml(p)


def test_naive_datetime_warns_once(caplog) -> None:
    schema._warned_naive_ts = False  # reset the module-level one-time guard
    with caplog.at_level(logging.WARNING):
        UserRecord.from_dict({
            "user_id": "u",
            "events": [{"ts": dt.datetime(2024, 1, 1, 9, 0), "source": "transaction", "fields": {}}],
        })
        UserRecord.from_dict({
            "user_id": "v",
            "events": [{"ts": dt.datetime(2024, 1, 2, 9, 0), "source": "transaction", "fields": {}}],
        })
    naive_warnings = [r for r in caplog.records if "naive datetime" in r.getMessage()]
    assert len(naive_warnings) == 1  # warned, and only once


# ---------------------------------------------------------------------------
# pipeline on a slim install — must fail fast BEFORE validate/tokenize, since
# the pretrain stage hard-requires the [train] extra
# ---------------------------------------------------------------------------


def test_pipeline_fails_fast_without_lightning(tmp_path, monkeypatch) -> None:
    import importlib.util as ilu

    from pragmatiq.core.errors import MissingExtraError

    real_sample({"n_users": 10, "seed": 1}, out=tmp_path / "raw")
    real_find_spec = ilu.find_spec
    monkeypatch.setattr(
        ilu, "find_spec",
        lambda name, *a, **k: None if name == "lightning" else real_find_spec(name, *a, **k),
    )
    with pytest.raises(MissingExtraError, match=r"pragmatiq\[train\]"):
        api.pipeline(tmp_path / "raw", out=tmp_path / "pipe")
    # Fail-fast means no compute happened: the output dir was never created.
    assert not (tmp_path / "pipe").exists()


def test_pipeline_refuses_invalid_data(tmp_path) -> None:
    """A directory that fails the data contract stops the pipeline before tokenizing."""
    from pragmatiq.core.errors import DataContractError

    (tmp_path / "empty").mkdir()
    with pytest.raises(DataContractError, match="fails the data contract"):
        api.pipeline(tmp_path / "empty", out=tmp_path / "pipe")
    assert not (tmp_path / "pipe" / "tok").exists()


# ---------------------------------------------------------------------------
# pretrain resume validation — anything other than None/'auto' must raise,
# never silently start a fresh run over the existing run dir
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_resume", ["latest", "true", "Auto", ""])
def test_pretrain_rejects_unknown_resume_value(tmp_path, bad_resume) -> None:
    with pytest.raises(ValueError, match="resume must be None or 'auto'"):
        api.pretrain(tmp_path / "tok", "some_run", resume=bad_resume)
