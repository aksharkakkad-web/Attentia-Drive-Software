# /test — Run tests for the current phase

Determine which phase is currently being worked on by checking which test files exist and which are passing/failing.

Then run the appropriate tests:
- Phase 1: `pytest tests/test_contracts.py -v`
- Phase 2: `pytest tests/test_kalman_filter.py -v`
- Phase 3: `pytest tests/test_signal_processor.py -v`
- Phase 4: `pytest tests/test_temporal_engine.py -v`
- Phase 5: `pytest tests/test_scoring_and_alerts.py -v`
- Phase 6: `pytest tests/test_calibration.py -v`
- Phase 9: `pytest tests/ -v` (all tests)

Report results clearly. If any test fails, show the failure and suggest a fix.
