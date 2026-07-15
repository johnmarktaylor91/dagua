<task>
r77-B3: r76_refs PROVENANCE PROBE (research/probe; oracle-integrity, high priority). The
runner parity probe (READ FIRST: .project-context/research/sprint_rng_matching/
r75_findings/r76_RUNNER_PARITY.md) proved scripts/ogdf_runner binary==source (byte-exact),
YET the stored benchmark_100seed_r76_refs tensors for random_dag_50 match NEITHER the
committed binary nor a fresh build when the prober reconstructed the payload/params. The
divergence must live in the PAYLOAD or PARAM MAPPING between (a) the benchmark adapter's
actual invocation (dagua/eval/competitors/ogdf_competitor.py + variants mapping, called by
scripts/run_benchmark.py) and (b) manual reconstruction. Find it and rule which is correct.

Repo: /home/jtaylor/projects/dagua (develop, read-only; output file only).

METHOD:
1. INSTRUMENT THE ADAPTER PATH (read-only monkeypatch in a scratch script, NOT an edit):
   wrap ogdf_competitor's runner invocation to DUMP the exact JSON payload + CLI args +
   env it sends for random_dag_50 x ogdf_fmmm__for__classic_fmmm_steps10, seed 100 --
   i.e., exactly what the r76_refs regen would have sent (same code path as run_benchmark
   --seed-refs; verify the variant mapping resolves fmmmFixedIterations=10 etc.).
2. RUN the committed runner on that dumped payload; compare output vs the STORED
   benchmark_100seed_r76_refs tensor (byte + Procrustes RMSD). Repeat for gem_iters2000
   and 2 more seeds.
3. If adapter-path output MATCHES the stored refs: the refs are faithful; the parity
   probe's manual payload was wrong -- DIFF the two payloads field-by-field and name what
   the manual reconstruction missed (this also explains MAAR attempt-3's failed gate --
   its port comparisons used the same wrong payload assumption; say so explicitly). If it
   does NOT match: bisect the drift (adapter code changed since 2026-07-03? graph builder
   changed? node-size inputs changed? -- git log the relevant files since 2026-07-02) and
   name exactly which stored dirs are affected.
4. VERDICT + implications: are the fmmm/gem/MAAR verdicts scored vs r76_refs SOUND? What
   (if anything) must be regenerated or rescored? Include the runner-hash tripwire
   recommendation (binary sha, source sha, ogdf lib shas, payload sha per tensor) as a
   concrete spec for the validator.

OUTPUT: .project-context/research/sprint_rng_matching/r75_findings/r76_REFS_PROVENANCE.md
-- payload diffs, comparison tables, verdict, affected-corpus list, commands. ASCII.
Budget ~45-60 min.
</task>
<default_follow_through_policy>
Most reasonable low-risk interpretation; document uncertainty honestly.
</default_follow_through_policy>
