# Phase 0 baseline

Frozen before the v2 harness was added. The tree is commit `a383b297deb6137a3e5a11fde2604a705c70f180` on `cursor/colab-cuda-teacher-compare-77a2` (PR #9). That branch was not modified.

Command, from `math-answer-diagnosis`:

```
python3 -m pytest -q
```

Result: **132 passed in 8.72s** (exit 0).

The harness tests were added after this measurement. A later run of both suites is recorded in `docs/harness_v2/CHANGES.md`.

Old inputs and outputs are described in `docs/harness_v2/old_pipeline.md`. The same input files can be sent to the old parser and the new harness separately:

```
python3 scripts/compare_old_vs_new.py --inputs <dir> --out <dir>
```

Each case is a JSON object. The old process reads `legacy_completion`. The new process reads `new_completion`. `reference_score` and `prior_ai_score` are removed before either side builds its view of the input. The two processes do not read each other's output directories. Neither process calls a live model.
