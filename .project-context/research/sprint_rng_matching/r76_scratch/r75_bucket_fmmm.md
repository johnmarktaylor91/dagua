# BUCKET: fmmm (32 divergent combos -- genuinely worse, needs root cause)

Target list: r75_targets_fmmm.json (BUCKET=fmmm). Reference: OGDF FMMM (seedable base ogdf_fmmm;
adapter dagua/eval/competitors/ogdf_competitor.py). Dagua side:
dagua/layout/ops/pipelines/fmmm.py + the fmmm/multilevel ops it composes.

Cluster structure to explain:
- Median stress excess +48.9%, median crossings excess +80.7% -- dagua is genuinely WORSE on most
  of this bucket (25 all-worse, 6 mixed, 1 all-better). This is NOT hairline; something
  algorithmic is different.
- 14 combos also fail neighborhood preservation (the np leg) -- the layouts are structurally
  different, not just noisier.
- Note r72 shipped "fmmm-multilevel" work and r74 kept a vectorize change (7cf7f83, perf-only);
  the divergence survives all of it.

Your job: find the FIRST algorithmic divergence between dagua's FMMM and OGDF's
(_references/ogdf/src/ogdf/energybased/fmmm/ -- FMMMLayout.cpp, multilevel/{MultilevelGraph,...},
FruchtermanReingold.cpp / NMM force approximation, initial placement, galaxy partitioning,
post-processing). Specifically check, in rough ROI order:
1. MULTILEVEL PIPELINE: galaxy partitioning (sun/planet/moon selection RNG + criteria), coarsest
   level layout, prolongation/initial placement of finer levels (position of planets around suns,
   angle sampling). A divergence at the coarsest level cascades everywhere. Which target combos
   are large enough to trigger multilevel vs single-level? (Check the threshold both sides --
   if dagua's threshold differs, small graphs may take entirely different paths.)
2. FORCE MODEL + SCHEDULE: repulsive-force approximation (NMM quadtree vs dagua's), force
   formulas, cooling, iteration counts per level, boundary handling ("force model" params in the
   adapter invocation vs OGDF defaults -- confirm the adapter's parameter mirroring is complete).
3. POST-PROCESSING: OGDF FMMM's postProcessing / fine-tuning rounds and componentLayout/packing
   (do the divergent combos include disconnected graphs? target JSON has a `disconnected` field).
4. RNG: OGDF uses its own RNG (double randomDouble etc.); does dagua's fmmm consume a matched
   stream? If the streams can be matched exactly, combos could go bit-exact -- assess feasibility
   the way r71 did for other engines (seeded refs proved 1006 combos bit-exact).
For each: CONFIRMED/HYPOTHESIS + decisive experiment + fix sketch + risk to the existing fmmm
bit-exact/3Q combos (33 fmmm combos were reclassified quality-identical in r74; do not break them).
