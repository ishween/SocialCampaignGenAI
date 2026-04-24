"""Unit tests for logger utilities: fmt_duration and PhaseTimer."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.utils.logger import fmt_duration, PhaseTimer, setup_logger


# ── fmt_duration ──────────────────────────────────────────────────────────────

def test_fmt_duration_under_60s():
    assert fmt_duration(45.2) == "45.2s"


def test_fmt_duration_zero():
    assert fmt_duration(0.0) == "0.0s"


def test_fmt_duration_exactly_60s():
    assert fmt_duration(60.0) == "1m 0.0s"


def test_fmt_duration_over_60s():
    result = fmt_duration(154.1)
    assert result == "2m 34.1s"


def test_fmt_duration_large():
    result = fmt_duration(3661.0)
    assert result.startswith("61m")


# ── PhaseTimer ────────────────────────────────────────────────────────────────

def test_phase_timer_logs_entry_and_exit():
    mock_logger = MagicMock(spec=logging.Logger)
    with PhaseTimer(mock_logger, "Test Phase"):
        pass
    calls = [str(c) for c in mock_logger.info.call_args_list]
    assert any("┌─ Test Phase" in c for c in calls)
    assert any("└─ Test Phase" in c for c in calls)


def test_phase_timer_returns_self():
    mock_logger = MagicMock(spec=logging.Logger)
    timer = PhaseTimer(mock_logger, "Phase")
    with timer as t:
        assert t is timer


def test_phase_timer_duration_in_exit_log():
    mock_logger = MagicMock(spec=logging.Logger)
    with PhaseTimer(mock_logger, "MyPhase"):
        pass
    exit_call = str(mock_logger.info.call_args_list[-1])
    assert "MyPhase" in exit_call
    # Duration format ends with 's'
    assert "s]" in exit_call


def test_phase_timer_suppresses_exceptions():
    """PhaseTimer must not swallow exceptions (it doesn't use __exit__ return)."""
    mock_logger = MagicMock(spec=logging.Logger)
    with pytest.raises(RuntimeError):
        with PhaseTimer(mock_logger, "Phase"):
            raise RuntimeError("boom")


# ── setup_logger ──────────────────────────────────────────────────────────────

def test_setup_logger_returns_root_logger():
    root = setup_logger(level="WARNING")
    assert root is not None
    assert isinstance(root, logging.Logger)


def test_setup_logger_creates_log_file(tmp_path):
    log_file = tmp_path / "test.log"
    setup_logger(level="INFO", log_file=log_file)
    assert log_file.exists()
