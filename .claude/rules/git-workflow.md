---
globs: ["**"]
---

# Git Workflow

- Commit to main after each phase passes all tests.
- Commit message: `git commit -m "phase N: what was done — all tests passing"`
- Always run `pytest tests/ -v` before committing. Never commit with failing tests.
- After committing a phase, tell the user: "Phase N complete. All tests pass. Committed. Ready for Phase N+1."