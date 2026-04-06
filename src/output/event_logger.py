"""Event logger — JSON-lines rotating log for pipeline events.

Writes one JSON object per line to a rotating log file.
50 MB max size, 5 backup files (PRD §9).

Events logged:
  - ALERT: an AlertCommand was fired
  - STATE_TRANSITION: pipeline state changed
  - CALIBRATION_COMPLETE: calibration completed (success or failure)

PRD §9 — Event Logging
"""

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config_prd import LOG_BACKUP_COUNT, LOG_DIR, LOG_MAX_BYTES
from src.contracts import AlertCommand, DistractionScore

_LOG_FILE = 'events.jsonl'


class EventLogger:
    """JSON-lines event logger with rotating file output.

    PRD §9.

    Args:
        log_dir: Directory for log files. Created if absent.
                 Defaults to LOG_DIR from config_prd.
    """

    def __init__(self, log_dir: str = LOG_DIR) -> None:
        log_path = Path(log_dir) / _LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger('attentia.events')
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False  # don't duplicate to root logger

        # Remove any stale handlers (e.g. from a previous instance pointing elsewhere)
        for h in list(self._logger.handlers):
            h.close()
            self._logger.removeHandler(h)

        self._handler = RotatingFileHandler(
            str(log_path),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8',
        )
        self._handler.setFormatter(logging.Formatter('%(message)s'))
        self._logger.addHandler(self._handler)

    def log_alert(self, alert_cmd: AlertCommand, score: DistractionScore) -> None:
        """Log a fired AlertCommand with its associated DistractionScore.

        PRD §9 — Event type: 'ALERT'

        Args:
            alert_cmd: The AlertCommand that was emitted.
            score: The DistractionScore that triggered the alert.
        """
        record = {
            'event_type': 'ALERT',
            'timestamp_ns': alert_cmd.timestamp_ns,
            'alert_id': alert_cmd.alert_id,
            'alert_type': alert_cmd.alert_type.value,
            'alert_level': alert_cmd.level.name,
            'composite_score': round(score.composite_score, 4),
            'active_classes': alert_cmd.active_classes,
        }
        try:
            self._logger.info(json.dumps(record))
            self._handler.flush()
        except Exception:
            pass

    def log_state_transition(
        self,
        previous: str,
        new: str,
        trigger: str,
        frame_id: int,
    ) -> None:
        """Log a pipeline state machine transition.

        PRD §9 — Event type: 'STATE_TRANSITION'

        Args:
            previous: Previous state name.
            new: New state name.
            trigger: What caused the transition.
            frame_id: Frame number at the time of transition.
        """
        record = {
            'event_type': 'STATE_TRANSITION',
            'timestamp_ns': time.time_ns(),
            'frame_id': frame_id,
            'previous_state': previous,
            'new_state': new,
            'trigger': trigger,
        }
        try:
            self._logger.info(json.dumps(record))
            self._handler.flush()
        except Exception:
            pass

    def log_calibration(
        self,
        yaw_offset: float,
        pitch_offset: float,
        baseline_ear: float,
    ) -> None:
        """Log calibration results.

        PRD §9 — Event type: 'CALIBRATION_COMPLETE'

        Args:
            yaw_offset: Neutral yaw offset computed during calibration (degrees).
            pitch_offset: Neutral pitch offset (degrees).
            baseline_ear: Mean open-eye EAR from calibration window.
        """
        record = {
            'event_type': 'CALIBRATION_COMPLETE',
            'timestamp_ns': time.time_ns(),
            'neutral_yaw_offset': round(yaw_offset, 4),
            'neutral_pitch_offset': round(pitch_offset, 4),
            'baseline_ear': round(baseline_ear, 4),
        }
        try:
            self._logger.info(json.dumps(record))
            self._handler.flush()
        except Exception:
            pass
