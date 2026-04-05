# /verify — Verify current phase is complete

Run ALL tests (not just the current phase): `pytest tests/ -v`

Then check:
1. Do all tests pass? (zero failures)
2. Are there any import errors?
3. Do all files required by the current phase exist?

If everything passes:
```
✅ Phase N verified. All tests pass. Safe to commit and move to Phase N+1.
Suggested: git add -A && git commit -m "phase N: [description] — all tests passing"
```

If anything fails:
```
❌ Phase N NOT verified. Issues found:
- [list each failure]
Fix these before moving on.
```
