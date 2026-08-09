# W2-1 deterministic multi-seed replication

## Changes

- Added `native_seed_replication.py` with the frozen seed-offset bank `(0, 1, 2, 17, 43)`,
  aggregate family-cost admission, forced-`k=1` replication-machinery-off seam, and W1-C
  within-family proxy culling to at most two geometrically distinct finalists.
- Wrapped stochastic SFDP, fCoSE, tsNET, and sparse t-FDP contest arms without modifying their
  shared pipeline implementations. Directed and undirected fCoSE use the same frozen-bank policy.
- Changed sparse-arm admission to price every seed trajectory up front while reserving honest
  referee work only for the two proxy finalists.
- Added focused ledger, frozen-bank, proxy-cull, and `k=1` tests. Updated the legacy cleanup test
  to isolate its original cleanup-ladder assertion through the forced-`k=1` seam.
- FIXUP F1: the within-family cull now spends one of its two seats on the best-proxy raw variant
  whenever a replicated family has raw candidates, preserving the fidelity parity floor without
  increasing the reserved referee cost.
- FIXUP F2: family admission now tries deterministic frozen prefixes `k -> 3 -> 1` and returns the
  admitted prefix to every SFDP, fCoSE, tsNET, and normal-contest t-FDP caller. The replication
  increment can be rejected without discarding an affordable base arm.
- FIXUP F3: corrected the forced-`k=1` attribution language. It disables W2-1 replication but does
  not recreate the pre-packet three-seed fCoSE/tsNET loops or their ledger event stream.

## Assumptions

- The packet's stochastic-family scope is the currently admitted high-variance force/embedding
  arms (SFDP, fCoSE, tsNET, and t-FDP). Deterministic planar, ordering, lattice, and stress
  certificate arms remain single-shot and are never placed in the replica map.
- A fixed parameter grid multiplied by a seed bank (tsNET perplexities) is one arm family for
  ledger admission and proxy culling.

## Test results

- Pre-fix regression run against `296b8adf`: `4 failed, 4 passed`. Failures independently covered
  full-prefix return semantics, unaffordable-package base fallback, raw-floor retention, and the
  inaccurate forced-`k=1` completion claim.
- FIXUP seed-replication regressions: `9 passed, 2 warnings in 0.04s`.
- FIXUP packet/cascade/cleanup set: `17 passed, 3 warnings in 81.85s`.
- FIXUP `ruff check . --fix`: passed.
- FIXUP mypy on `dagua/cli.py` and all four changed pipeline modules: passed.
- Required layout/graph gate was interrupted after the native W5 finisher exceeded the practical
  30-minute ceiling: `16 passed, 3 warnings in 1794.31s`; no failure occurred before interruption.
- Final non-slow/non-benchmark/non-rare suite reached `217 passed, 1 skipped, 107 deselected,
  1 xfailed` with no failures before its 15-minute bound (`890.92s`, exit 124).
- Import guard: passed; `dagua.__file__` resolved inside this worktree.
- `ruff check . --fix`: passed with no unrelated modifications.
- `mypy --follow-imports=silent dagua/cli.py`: `Success: no issues found in 1 source file`.
- Packet module mypy: `Success: no issues found in 1 source file`.
- Packet/cascade/cleanup fast tests: `13 passed, 3 warnings in 12.83s`.
- Seed-replication tests after final API cleanup: `5 passed, 2 warnings in 0.01s`.
- Broader portfolio integration run: `113 passed` before one expected legacy assertion failed
  because it counted every seed replica as honestly scored. The test was corrected to force the
  packet's replication-machinery-off `k=1` path; its rerun passed
  (`1 passed, 3 warnings in 4.79s`).
- Twelve-row winnable tie smoke: `2 strict / 10 tied / 0 behind`.
  - `rome/grafo10046.40`: `+0.491 tied -> +1.434 strictly_best`.
  - `rome/grafo10044.38`: `+0.083 tied -> +1.921 strictly_best`.
  - `north/g.10.0` and `suitesparse/arc130` remained tied, not behind.
- Forced `k=1` removed the replication increment on both conversions and reproduced the observed
  position bytes and tied deltas (`+0.491`, `+0.083`) on those rows. This seam is not the
  pre-packet engine: legacy fCoSE/tsNET ran three seeds and used different ledger events.
- Won-row regression smoke on `north/g.10.11`, `north/g.10.12`, and `north/g.10.14`:
  `3 strict / 0 tied / 0 behind`, with no status regressions.

## Controversial choices

- Replica culling occurs after all cheap proxy scores are available and before the global W1-C
  cascade. This makes the unsafe generic proxy compare only siblings inside one stochastic family;
  the global honest referee and incumbent-first tie rule remain unchanged.
- Raw-floor retention is folded into the existing two-finalist quota. This keeps package pricing
  unchanged while ensuring the proxy cannot spend both family seats on polished variants.
- Prefix fallback uses the frozen requested count followed by `3` and `1`; every attempted prefix
  remains one all-or-nothing ledger package, and tsNET scales its two-perplexity trajectory count
  with the admitted seed prefix.

## Concerns

- The packet ran targeted fast dev smoke and three previously-won tripwire rows, not the complete
  dev63/training-121 tallies; those remain orchestrator-scale verification.
- Existing hard return-reserve checks inside `admit_native_work` remain unchanged.
- If the pre-packet base arm itself fails its existing deterministic admission guards, the family
  can still be absent; F2 guarantees that an unaffordable replication increment does not cause
  that absence when the base package fits.
- No newly unreachable code was identified. The exported historical
  `FCOSE_CONTEST_SEEDS`/`TSNET_CONTEST_SEEDS` constants are compatibility labels and can be removed
  only after downstream imports are audited.

## Knowledge

- Before this packet, fCoSE and tsNET already generated three adjacent seeds, but admitted each
  separately and made raw variants mandatory, so partial families and excess full-referee work
  were possible.
- The two strict conversions disappear when `k=1` is forced, which supports replication as the
  mechanism on those rows but is not a byte-parity comparison with main at `0ee2db36`.

FIXUP: OK
