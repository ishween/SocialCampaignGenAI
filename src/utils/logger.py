"""Structured logging setup + pipeline timing utilities."""

import contextlib
import logging
import sys
import time
from pathlib import Path


# ── Duration formatting ───────────────────────────────────────────────────────

def fmt_duration(seconds: float) -> str:
    """Format raw seconds as a readable string: '45.2s' or '2m 34.1s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


# ── Phase timer context manager ───────────────────────────────────────────────

class PhaseTimer:
    """
    Context manager that wraps a pipeline phase with clean log bookends
    and automatic wall-clock timing.

    Usage:
        with PhaseTimer(logger, "Phase 2 · Resize"):
            ...

    Output:
        ┌─ Phase 2 · Resize
        └─ Phase 2 · Resize  [1.3s]
    """

    def __init__(self, logger: logging.Logger, name: str) -> None:
        self._logger = logger
        self._name = name
        self._t0: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._t0 = time.time()
        self._logger.info(f"  ┌─ {self._name}")
        return self

    def __exit__(self, *_) -> None:
        elapsed = fmt_duration(time.time() - self._t0)
        self._logger.info(f"  └─ {self._name}  [{elapsed}]\n")


# ── Logger setup ──────────────────────────────────────────────────────────────

def setup_logger(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Silence noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return root
