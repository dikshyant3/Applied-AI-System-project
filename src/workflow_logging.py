"""Privacy-conscious JSON events for TuneWise recommendation workflows."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, TextIO

WORKFLOW_LOGGER_NAME = "tunewise.workflow"
_logger = logging.getLogger(WORKFLOW_LOGGER_NAME)


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.name == WORKFLOW_LOGGER_NAME:
            return record.getMessage()
        return json.dumps({
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }, ensure_ascii=False, sort_keys=True)


def configure_workflow_logging(stream: TextIO | None = None,
                               level: int = logging.INFO) -> None:
    """Send machine-readable workflow events to ``stream`` (stderr by default)."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(_JsonLineFormatter())
    logging.basicConfig(level=level, handlers=[handler])
    _logger.setLevel(level)
    _logger.propagate = True


def log_workflow_event(request_id: str, event: str, **fields: Any) -> None:
    """Emit one JSON object. Callers must pass only required, non-secret fields."""
    record = {"request_id": request_id, "event": event, **fields}
    _logger.info(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))


__all__ = ["configure_workflow_logging", "log_workflow_event"]
