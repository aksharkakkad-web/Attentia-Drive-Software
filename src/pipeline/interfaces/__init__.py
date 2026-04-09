"""Portability boundary interfaces.

Four and only four things differ between MacBook development and the Raspberry
Pi 5 production target: camera, audio, IMU, thermal. Each is isolated behind
one interface in this package. Downstream pipeline and logic code consumes
only the interface, never a hardware library directly.

See docs/INTERFACES.md for the contracts and .claude/rules/portability.md
for the enforcement rules.
"""

from src.pipeline.interfaces.audio_sink import (
    AfplayAudioSink,
    AudioSink,
    NullAudioSink,
    SystemEvent,
)
from src.pipeline.interfaces.frame_source import (
    ColorSpace,
    FrameResult,
    FrameSource,
    NullFrameSource,
    OpenCVFrameSource,
)
from src.pipeline.interfaces.imu_source import ImuSource, StubImuSource
from src.pipeline.interfaces.thermal_monitor import (
    NoopThermalMonitor,
    ThermalMonitor,
)

__all__ = [
    "AfplayAudioSink",
    "AudioSink",
    "ColorSpace",
    "FrameResult",
    "FrameSource",
    "ImuSource",
    "NoopThermalMonitor",
    "NullAudioSink",
    "NullFrameSource",
    "OpenCVFrameSource",
    "StubImuSource",
    "SystemEvent",
    "ThermalMonitor",
]
