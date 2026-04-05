---
globs: ["src/logic/**", "src/config_prd.py"]
---

# Threshold and Config Rules

- Every number used in detection logic (thresholds, weights, durations, angles) must be imported from `src/config_prd.py`.
- If you need a new constant, add it to `config_prd.py` with a comment referencing the PRD section (e.g., `# PRD §5.6`).
- Never hardcode values like `0.55`, `2.0`, `30.0`, `0.45` directly in logic code.
- When the PRD and `config_prd.py` disagree, the PRD wins. Update `config_prd.py` to match.
