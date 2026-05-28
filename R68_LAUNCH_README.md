# R68 100-seed bit-exact + TOST launch instructions

After the Anthropic interview, run these in order. Total expected time: ~12-24 hours unattended.

## Step 1: Patch benchmark variants (codex, ~30 min)

The R66b 5-seed report revealed the benchmark variants don't include `fidelity_mode` in their params,
so the benchmark is running dagua's default (fast tensor) implementations, NOT the R36-R65 bit-exact ports.

This codex patches every variant in `dagua/eval/variants.py` to opt into the right fidelity_mode:

```bash
~/.claude/scripts/codex-bg.sh /tmp/r68_variants.log /tmp/PROMPT_68_variant_fidelity_mode.md \
  --cd /home/jtaylor/projects/dagua --sandbox danger-full-access \
  -c model_reasoning_effort=high
```

Wait for `CODEX_DONE` notification. The codex will commit the patched variants.py.

## Step 2: Launch 100-seed benchmark + combined report

```bash
nohup bash scripts/r68_100seed_with_tost.sh </dev/null > /tmp/r68_launcher.log 2>&1 &
echo "R68 launched, PID=$!"
```

The script:
1. Sanity-checks variants.py has `fidelity_mode` (from step 1)
2. Purges affected classic_* + ref rows from results.json
3. Runs `run_benchmark.py --seeds 100` with 5-min timeouts
4. Consolidates positions.h5
5. Runs `fast_fidelity_report.py` for per-seed Procrustes RMSDs
6. For variants with max RMSD >= 1e-3, runs `r68_tost_followup.py` (statistical equivalence)
7. Generates combined report at `eval_output/fidelity_report_r68/report.md`
8. iMessages on completion

## Expected outcome

After R68:
- Deterministic engines (~10): MACHINE_EPSILON tier (per-seed bit-exact)
- Stochastic engines with verified RNG ports (~10): BIT_EXACT tier (per-seed RMSD <1e-3 on most graphs)
- Chaotic engines on real benchmark graphs (gem, drl, lgl, etc.): STRONG_EQUIVALENT or WEAK_EQUIVALENT
  via TOST (per-seed differs by basin, distributions match)

## Monitor

```bash
tail -f /tmp/r68_100seed_run.log
```

Or:
```bash
~/.claude/scripts/bg-watch.sh <PID> /tmp/r68_100seed_run.log --label r68-100seed
```

## If something fails

Each step is committed separately. The 5-seed data in results.json is preserved by the smart purge
(only classic_* + 3 re-paired refs are purged). If R68 fails partway, the fast 5-seed report at
`eval_output/fidelity_report_fast/report.md` is still the fallback.
