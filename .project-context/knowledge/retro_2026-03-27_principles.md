# Principles from Fidelity Analysis Retro (2026-03-27)

## P1: SMOKE TEST EVERY FLAG PATH
**Rule:** When adding a --skip-X flag, run the FULL pipeline with the flag on
a tiny input (2 graphs) and verify the output differs from without the flag.
Check: are the skipped columns actually empty/missing? Is the runtime actually
shorter? Does it still produce valid output?
**Trigger:** Any time a flag is added to skip/reduce computation.
**Prevents:** Incident 5 (skip-metrics didn't skip metric analysis).
**Incident 10 (KeyError on empty metrics).**

## P2: CLEAR BYTECACHE BEFORE EVERY RELAUNCH
**Rule:** `find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null` before
any `python scripts/...` launch after editing code.
**Trigger:** Any time code is edited and a script is relaunched.
**Prevents:** Incident 3 (bytecache served old code after fix).

## P3: NEVER USE ThreadPoolExecutor FOR CPU-BOUND WORK
**Rule:** If the work involves numpy/torch/scipy math (SVD, bootstrap, matrix
ops), use ProcessPoolExecutor or multiprocessing. ThreadPoolExecutor is ONLY
for I/O-bound work (file reads, network calls). Python's GIL makes threads
SLOWER than serial for CPU work due to context switching.
**Trigger:** Any time parallelism is added.
**Prevents:** Incident 4 (GIL blocking Procrustes parallelism).

## P4: ALWAYS SET PYTHONUNBUFFERED=1
**Rule:** Every nohup/dispatch/background Python command must use
`PYTHONUNBUFFERED=1` in the environment. No exceptions.
**Trigger:** Any background Python process.
**Prevents:** Incident 7 (no progress visibility for hours).

## P5: TEST FROZEN DATACLASS MUTATION AT WRITE TIME
**Rule:** If you add a mutable field to a dataclass, immediately test that
assignment works: `obj.field = value`. If the dataclass is frozen, the test
catches it instantly. Don't wait for a full pipeline run.
**Trigger:** Any time a field is added to a dataclass.
**Prevents:** Incident 2 (frozen dataclass silently broke pairing).

## P6: OVERRIDE CRITICS ON DOMAIN GROUNDS
**Rule:** When an adversarial critic recommends a technically-correct-but-
practically-wrong approach, OVERRIDE with domain judgment and document why.
"The critic said X, but in practice Y because Z."
**Trigger:** Adversarial critique of a spec.
**Prevents:** Incident 1 (no scale normalization because critic said scale matters).

## P7: DEFAULT TO 1000 BOOTSTRAP SAMPLES
**Rule:** 1000 bootstrap samples gives ~1% CI precision. 10K gives ~0.3%.
The difference is never decision-relevant. Default to 1000. Only use 10K
if explicitly requested for publication-quality figures.
**Trigger:** Any bootstrap CI computation.
**Prevents:** Incident 6 (10x compute for negligible statistical benefit).

## P8: DRY-RUN ON 2 GRAPHS BEFORE FULL RUN
**Rule:** Before any analysis that processes all 105 graphs, run with
`--max-graphs 2` first. Verify: output files created, correct columns,
runtime proportional, no errors. Only then launch full.
**Trigger:** Any full analysis pipeline run.
**Prevents:** Incidents 1, 2, 3, 5, 10 (all would have been caught in <1 min).

## P9: STOP ESTIMATING RUNTIMES
**Rule:** When asked "how long will this take?" say "I don't know, let's
run it and watch." Every estimate in this session was wrong by 2-5x.
**Trigger:** User asks for time estimate.
**Prevents:** Incident 8 (false expectations, repeated disappointment).

## P10: USE nohup DIRECTLY, NOT dispatch.sh, FOR ANALYSIS
**Rule:** For analysis scripts (not benchmark runs), use `PYTHONUNBUFFERED=1
nohup python script.py > stdout.log 2> stderr.log &` directly. dispatch.sh
adds complexity (wrappers, status files, notification hooks) that causes
confusion for interactive monitoring.
**Trigger:** Running analysis/report scripts.
**Prevents:** Incident 9 (dispatch.sh confusion with kill/restart cycles).
