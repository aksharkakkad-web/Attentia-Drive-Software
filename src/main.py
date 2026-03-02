"""Attentia Drive — Entry point.

Parses command-line arguments, loads configuration, applies CLI overrides,
and runs the processing pipeline.

Usage:
    python src/main.py                           # Webcam mode
    python src/main.py --source path/to/video.mp4  # Video replay mode
    python src/main.py --no-display              # Headless mode
    python src/main.py --config custom.yaml      # Custom config file
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config_loader import load_config
from src.pipeline.pipeline_manager import PipelineManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
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
        help="Override frame source: 'webcam' or path to video file",
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
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure the Python logging system.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING).
    """
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

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path.resolve())
        sys.exit(1)

    try:
        config = load_config(str(config_path))
    except Exception:
        logger.exception("Failed to load configuration")
        sys.exit(1)

    if args.source is not None:
        if args.source.lower() == "webcam":
            config.frame_source.type = "webcam"
        else:
            source_path = Path(args.source)
            if not source_path.exists():
                logger.error("Video file not found: %s", source_path.resolve())
                sys.exit(1)
            config.frame_source.type = "video"
            config.frame_source.video_path = str(source_path)

    if args.no_display:
        config.display.enabled = False

    logger.info("Attentia Drive starting...")
    logger.info("Frame source: %s", config.frame_source.type)
    logger.info("Classifier enabled: %s", config.classifier.enabled)
    logger.info("Object detector enabled: %s", config.object_detector.enabled)
    logger.info("Display enabled: %s", config.display.enabled)

    try:
        pipeline = PipelineManager(config)
        pipeline.run()
    except Exception:
        logger.exception("Pipeline crashed")
        sys.exit(1)

    logger.info("Attentia Drive stopped.")


if __name__ == "__main__":
    main()
