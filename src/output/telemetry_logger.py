"""Telemetry logger that writes per-frame data to a CSV file.

Logs frame-level pipeline results for offline analysis and debugging.
Creates the log directory if it does not exist. Flushes periodically
to avoid data loss.

Phase: 1-2 (active).
"""

import csv
import logging
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

from src.config_loader import TelemetryConfig
from src.pipeline.frame_record import FrameRecord

logger = logging.getLogger(__name__)

CSV_HEADERS = [
    "frame_number",
    "timestamp",
    "raw_prob",
    "smoothed_prob",
    "driver_state",
    "is_distracted",
    "active_triggers",
    "confirmed_objects",
    "alert_fired",
    "inference_ms",
    "pipeline_ms",
]

FLUSH_INTERVAL = 30


class TelemetryLogger:
    """CSV telemetry logger for per-frame pipeline data.

    Opens a CSV file on initialization, writes headers, and appends
    one row per logged frame. Flushes every 30 frames to balance
    performance and data safety.

    Args:
        config: TelemetryConfig with log file path and interval.
    """

    def __init__(self, config: TelemetryConfig) -> None:
        self._config = config
        self._file: Optional[TextIOWrapper] = None
        self._writer: Optional[csv.writer] = None
        self._rows_since_flush = 0

        if config.enabled:
            self._open()

    def _open(self) -> None:
        """Open the CSV log file and write headers."""
        try:
            log_path = Path(self._config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            self._file = open(log_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(CSV_HEADERS)
            self._file.flush()

            logger.info("TelemetryLogger: Logging to '%s'", log_path.resolve())
        except Exception:
            logger.exception("TelemetryLogger: Failed to open log file")
            self._file = None
            self._writer = None

    def log(self, record: FrameRecord) -> None:
        """Log a single frame record to the CSV file.

        Args:
            record: The FrameRecord to log.
        """
        if self._writer is None:
            return

        try:
            raw_prob = (
                record.classifier_result.distracted_probability
                if record.classifier_result
                else 0.0
            )
            active_triggers = (
                ";".join(record.assessment.active_triggers)
                if record.assessment
                else ""
            )
            confirmed_objects = (
                ";".join(record.assessment.confirmed_objects)
                if record.assessment
                else ""
            )

            self._writer.writerow([
                record.frame_number,
                f"{record.timestamp:.6f}",
                f"{raw_prob:.4f}",
                f"{record.smoothed_probability:.4f}",
                record.driver_state.name,
                record.assessment.is_distracted if record.assessment else False,
                active_triggers,
                confirmed_objects,
                record.alert is not None,
                f"{record.inference_time_ms:.2f}",
                f"{record.pipeline_time_ms:.2f}",
            ])

            self._rows_since_flush += 1
            if self._rows_since_flush >= FLUSH_INTERVAL:
                self._file.flush()
                self._rows_since_flush = 0

        except Exception:
            logger.exception("TelemetryLogger: Failed to write row")

    def close(self) -> None:
        """Close the CSV log file."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                logger.exception("TelemetryLogger: Failed to close log file")
            self._file = None
            self._writer = None
            logger.info("TelemetryLogger: Closed")
