<task>
r76-C1+B2a: triage dagua's remaining classical_mds divergents (22) and gem divergents (7)
against honest references. RESEARCH/PROBE ONLY: no repo changes; scratch in /tmp; write only the
results file. Read r75_RESULTS.md first for context (DLA port landed + overlay fixed; gem's old
passes were stale-binary artifacts).

Repo: /home/jtaylor/projects/dagua (develop, read-only).
Data: eval_output/fidelity_definitive/r75_final.jsonl (rows: engine contains classical_mds or
classic_gem, quality_identical_raw=false). Positions: per-combo-freshest across
benchmark_100seed_* dirs (r75_fixes, r75_mds_topup, r75_topup2 newest).

MDS QUESTIONS:
1. The 22: split connected (expected: the 14 evidenced-floor rows from r75's E2 eigenvalue-tie
   probe -- confirm they are exactly those) vs disconnected (which legs still fail post-DLA?).
2. For still-failing disconnected rows: is it the crossings leg? Crossings ARE affected by DLA
   packing geometry (component placement changes edge crossings=0 mostly for disconnected...
   actually inter-component edges don't exist -- so crossings should be within-component only and
   near-identical. VERIFY: compute crossings per component for dagua vs reference on 2 failing
   combos). If stress: is the registered-pair sample including any cross-component pairs?
   Name the leg + mechanism per combo.
3. For the connected 14: assemble the formal floor dossier (eigengap table from the r75 E2 probe
   + a 1-seed demonstration that scipy driver choice changes coordinates) -- the disposition
   artifact r76 will cite.
GEM QUESTIONS:
4. The 7 (incl. 2 new honesty regressions): legs + gaps + Procrustes-vs-honest-ref sample.
5. First-divergence: gem is deterministic-given-seed via OGDF setSeed. For grid_5x5 seed 100,
   compare dagua's trajectory vs the honest runner output at rounds 20/100 (runner supports
   gemRounds now): does divergence grow with rounds (chaotic) or start at round ~0 (init/RNG
   mismatch)? Read ogdf-src GEMLayout.cpp (foxglove-202510) vs dagua/layout/ops/gem.py: list
   concrete behavioral deltas (RNG stream, update order, impulse formula) with file:line.
6. RECOMMEND per cluster: port-fix (effort) / floor-with-evidence / aggregate-tier.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r76_PROBE_mds_gem_triage.md
-- commands, tables, verdicts. ASCII. Budget ~45 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; if one question blocks, document and continue.
</default_follow_through_policy>
