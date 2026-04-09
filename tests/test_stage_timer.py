"""Tests for src/pipeline/stage_timer.py."""

import time

import pytest

from src.pipeline.stage_timer import StageTimer


class TestStageTimer:
    def test_measure_records_samples(self):
        t = StageTimer(report_interval_s=100.0)
        with t.measure('test_stage'):
            pass  # near-zero duration
        assert 'test_stage' in t._samples
        assert len(t._samples['test_stage']) == 1
        assert t._samples['test_stage'][0] >= 0.0

    def test_measure_multiple_stages(self):
        t = StageTimer()
        with t.measure('a'):
            pass
        with t.measure('b'):
            pass
        assert 'a' in t._samples
        assert 'b' in t._samples

    def test_maybe_report_clears_window(self):
        t = StageTimer(report_interval_s=0.0)  # report every call
        with t.measure('x'):
            pass
        assert len(t._samples.get('x', [])) == 1
        t.maybe_report()
        # Window should be cleared after report
        assert len(t._samples) == 0

    def test_maybe_report_respects_interval(self):
        t = StageTimer(report_interval_s=100.0)
        with t.measure('x'):
            pass
        t.maybe_report()
        # Should NOT have reported (interval not elapsed), samples still present
        assert len(t._samples.get('x', [])) == 1

    def test_max_samples_cap(self):
        t = StageTimer(report_interval_s=100.0)
        for _ in range(700):
            with t.measure('stage'):
                pass
        assert len(t._samples['stage']) == StageTimer._MAX_SAMPLES

    def test_empty_report_does_not_raise(self):
        t = StageTimer(report_interval_s=0.0)
        t.maybe_report()  # no samples, should not raise
