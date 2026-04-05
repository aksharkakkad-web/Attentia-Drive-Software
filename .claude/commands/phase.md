# /phase — Show current build progress

Read `docs/BUILD_PLAN.md` and check the current state of the project.

For each phase (0 through 9), check:
1. Do the files that phase creates exist?
2. Do the tests for that phase pass? (Run them silently)

Then report:
- Which phases are COMPLETE (files exist + tests pass)
- Which phase is CURRENT (files partially exist or tests failing)
- Which phases are NOT STARTED

Format:
```
✅ Phase 0: Setup — complete
✅ Phase 1: Data Contracts — complete
🔧 Phase 2: Kalman Filter — in progress (tests failing)
⬜ Phase 3: Signal Processor — not started
...
```

End with: "Next step: [brief description of what to do next from the build plan]"
