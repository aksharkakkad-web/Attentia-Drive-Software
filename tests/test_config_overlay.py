"""Tests for apply_prd_overlay() in src/config_loader.py.

Validates the YAML overlay mechanism that patches src.config_prd module
attributes at startup.
"""

import importlib

import pytest

from src.config_loader import apply_prd_overlay


class TestApplyPrdOverlay:
    """Test the PRD overlay mechanism."""

    def _reload_config_prd(self):
        """Reload config_prd to get fresh canonical values."""
        import src.config_prd as mod
        importlib.reload(mod)
        return mod

    def test_desk_overlay_applies(self):
        """desk.yaml should override ROAD_ZONE_YAW_MAX and friends."""
        config_prd = self._reload_config_prd()
        original_yaw = config_prd.ROAD_ZONE_YAW_MAX

        apply_prd_overlay('desk')

        import src.config_prd as patched
        assert patched.ROAD_ZONE_YAW_MAX == 25.0
        # Restore
        self._reload_config_prd()

    def test_pi5_overlay_is_noop(self):
        """pi5.yaml is empty — no values should change."""
        config_prd = self._reload_config_prd()
        original_yaw = config_prd.ROAD_ZONE_YAW_MAX

        apply_prd_overlay('pi5')

        import src.config_prd as patched
        assert patched.ROAD_ZONE_YAW_MAX == original_yaw
        self._reload_config_prd()

    def test_missing_mode_is_noop(self):
        """A mode with no YAML file should be a silent no-op."""
        config_prd = self._reload_config_prd()
        original_yaw = config_prd.ROAD_ZONE_YAW_MAX

        apply_prd_overlay('nonexistent_mode_xyz')

        import src.config_prd as patched
        assert patched.ROAD_ZONE_YAW_MAX == original_yaw
        self._reload_config_prd()

    def test_desk_overlay_sets_cooldowns(self):
        """desk.yaml should set cooldown values."""
        self._reload_config_prd()
        apply_prd_overlay('desk')

        import src.config_prd as patched
        assert patched.COOLDOWN_VISUAL == 2.0
        assert patched.COOLDOWN_DROWSINESS == 2.0
        assert patched.COOLDOWN_PHONE == 5.0
        self._reload_config_prd()

    def test_desk_overlay_sets_temporal_thresholds(self):
        """desk.yaml should set gaze/head/face absence thresholds."""
        self._reload_config_prd()
        apply_prd_overlay('desk')

        import src.config_prd as patched
        assert patched.T_GAZE_SECONDS == 1.0
        assert patched.T_HEAD_SECONDS == 0.75
        assert patched.T_FACE_ABSENT_SECONDS == 2.5
        self._reload_config_prd()
