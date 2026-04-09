"""Attentia Drive — Entry point.

Parses command-line arguments and logging setup.

Usage:
    python src/main.py                               # Webcam, default config
    python src/main.py --source path/to/video.mp4   # Video replay mode
    python src/main.py --no-display                  # Headless mode
    python src/main.py --config custom.yaml          # Custom config file
    python src/main.py --log-level DEBUG             # Verbose logging
"""

import sys
from pathlib import Path

# Allow running as: python src/main.py (adds project root to path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Attentia Drive — Real-time distracted driving detection",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Frame source override: 'webcam' or path to video file",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV debug display window",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["desk", "pi5"],
        default="desk",
        help="Target mode — applies config/<mode>.yaml overlay onto PRD values (default: desk)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save one debug frame per second + session_log.jsonl to debug_frames/<timestamp>/",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure the Python logging system."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Main entry point for Attentia Drive."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Attentia Drive starting in %s mode...", args.mode)

    # CRITICAL: apply PRD overlay BEFORE importing PipelineManagerV2 (and
    # everything it pulls in). Downstream modules do `from src.config_prd
    # import X`, which binds values at import time — overlaying after the
    # import would be a silent no-op.
    from src.config_loader import apply_prd_overlay
    apply_prd_overlay(args.mode)

    from src.pipeline.pipeline_manager_v2 import PipelineManagerV2

    pipeline = PipelineManagerV2(
        source=args.source,
        display=not args.no_display,
        config_path=args.config,
        save_frames=args.save_frames,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
