# r75 ADVERSARIAL CRITIQUE -- verify/refute the research sweep before ANY code lands

You are the adversarial gatekeeper for dagua's r75 fidelity sprint. Ten research reports (5 from
OpenAI Codex agents, 5 from Anthropic Claude agents, 1:1 redundant per bucket) live in
/home/jtaylor/projects/dagua/.project-context/research/sprint_rng_matching/r75_findings/
as r75_{sfdp,sugiyama,fmmm,mds_tails,metrics}_{codex,sonnet}.md. In r74, this critique role
correctly predicted two fix reverts that the research agents got wrong. Your verdicts gate all
implementation. Be ruthless; read PRIMARY SOURCE, not the reports' summaries of it.

Repo: /home/jtaylor/projects/dagua (develop @ 89ed3c3, read-only for you except your report).
Reference trees: /home/jtaylor/projects/_references/{graphviz,igraph,ogdf}.

## VERSION PINNING (a sonnet caught this -- treat as your first duty)
The _references/graphviz working tree is checked out at ~14.1.5 HEAD but the benchmark's dot
binary is graphviz 7.0.5 (conda, built WITHOUT GTS so prism overlap compiles out). Any code
citation not verified against `git show 7.0.5:<path>` (run inside _references/graphviz) is
suspect. Do the same version audit for:
- igraph: installed python-igraph is 1.0.0; check what tag the _references/igraph tree is at
  (git describe) and reconcile any behavioral citations against the INSTALLED version's source.
- OGDF: find how the benchmark's ogdf runner binary is built (dagua/eval/competitors/ogdf*,
  scripts/ or docs mentioning ogdf build) and which OGDF source SHA it corresponds to.
State clearly per claim whether it was verified against the version actually running in the
benchmark.

## THE CONFLICTS YOU MUST SETTLE (ranked)
1. FMMM force path: codex claims dagua-exact-vs-OGDF-NMM affects ALL 29 OGDF-FMMM rows;
   sonnet claims NMM only activates >=175 particles so 25/32 targets never touch it, and instead
   blames (a) coincident-node repulsion: dagua zeroes force while OGDF numexcept jitters via the
   global RNG stream (desync), and (b) an oscillation-damping angle formula rounding difference
   feeding a ceil() sector lookup. Read OGDF FruchtermanReingold.cpp/NewMultipoleMethod.cpp
   particle thresholds + numexcept.cpp in the RUNNER'S OGDF version and rule: which mechanism
   explains the small-graph rows? Design the minimal decisive experiment if source alone cannot.
2. sfdp gv_random/permutation: codex says graphviz 7.0.5 gv_permutation uses rejection sampling
   and dagua ops/sfdp.py:247-253 raw-modulo is wrong; sonnet says (via git show 7.0.5:) dagua
   correctly splits rejection vs raw-modulo consumers for 7.0.5. Settle with `git show 7.0.5:`.
   (Impact is believed near-zero either way; settle it for correctness debt only.)
3. Crossings-metric policy: metrics codex frames small-count zero-spread failures as margin
   brittleness (up to 75-115 rows reclassifiable); metrics sonnet says most zero-spread failures
   are GENUINE deterministic 1-crossing differences (dagua always 3 vs ref always 2) that a
   layout fix, not a margin widening, should close -- only ~10-15 rows are true miscalibration.
   Rule on the honest policy per the north star ("statistically identical" -- a deterministic
   +1 crossing is NOT identical). Also reconcile their differing cross-fail counts (235 vs 163).
4. classical_mds connected rows: codex says eigensolver-basis HYPOTHESIS with a cheap driver
   experiment; sonnet says pre-existing documented dsyevr degenerate-eigenspace FLOOR. A floor
   claim needs FP-chaos evidence per project rules -- decide what evidence exists and what
   experiment would settle it.
5. sugiyama igraph LP objective: codex found reference-tree source has BOTH indegs and outdegs
   populated with IGRAPH_IN (sugiyama.c:589-592), conflicting with dagua's r74 objective fix
   (out-strength for sources). Verify against INSTALLED python-igraph 1.0.0 source version and
   rule whether dagua's objective is right.

## SPRINT-LEAD FINDINGS TO FOLD IN (verified by Fable, the sprint lead)
- STALE-VINTAGE: the r74 Phase-2 rescore (the "337 divergent" baseline) used pre-r74-fix
  positions. Verified: current code makes classic_sfdp_p_neg2 bit-identical to default (r74_fixes
  positions, 15/15 seeds x 4 graphs), while the rescore rows show different D-values -> all 52
  p_neg2 rows were scored on stale layouts. A true-baseline rescore (corrected metrics x 10-dir
  overlay incl r74_fixes) is running; its output will be
  eval_output/fidelity_definitive/r75_truebaseline.jsonl. If present when you run, USE it to
  re-rank all impact estimates in the reports.
- The sfdp codex empirically proved (probe rms 4e-16) that graphviz sfdp ignores theta/maxiter
  attrs -> 47 theta04/theta08/steps200 rows have no expressible canonical reference. Critique the
  proposed disposition: route them to a non-counting "no-canonical-reference" tier (needs JMT
  sign-off). Is there ANY way to make graphviz honor those knobs (env, build flag, cmdline) in
  7.0.5? Check before blessing the reclassification.
- The original brief's "83/126 sfdp hairline <=1%" was a bucketing bug by the sprint lead
  (negative gaps fell into the <=0.01 bucket); sonnet's strict recount says 8/126. Do not use
  the 83 figure.

## YOUR DELIVERABLE
Write .project-context/research/sprint_rng_matching/r75_findings/r75_ADVERSARIAL_VERDICTS.md:
For EVERY proposed fix across the 10 reports (dedup them), a verdict:
  APPROVE (implement as specced) / APPROVE-WITH-CHANGES (state them) / REJECT (why) /
  NEEDS-EXPERIMENT (give the exact command/script sketch + expected discriminating outcome).
Then a recommended implementation ORDER with dependency notes, per-fix regression gates
(which existing bit-exact/3Q combos to re-verify), and explicit blast-radius notes (the r74
failure mode was blanket fixes breaking 25 bit-exact combos -- flag any fix that is not gated
to a fidelity_mode/variant/graph-class).
Rules: cite file:line with version pin for every ruling; run cheap probes yourself where they
settle a conflict (<10 min each, /tmp scratch only); NO code changes to the repo; ASCII only.
