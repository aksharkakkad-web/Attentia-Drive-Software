---
globs: ["tests/**"]
---

# Testing Rules

- All tests use synthetic data. Never require a camera, model file, or display to run.
- Test edge cases: zero values, None inputs, boundary thresholds, empty buffers.
- After writing or modifying a test file, run it immediately with `pytest <file> -v` and confirm every test passes before doing anything else.
- Each test function tests ONE behavior. Name it `test_<what_it_verifies>`.
- Use `pytest.approx()` for float comparisons, not `==`.
- If a test fails, fix the code (not the test) unless the test itself has a bug.
