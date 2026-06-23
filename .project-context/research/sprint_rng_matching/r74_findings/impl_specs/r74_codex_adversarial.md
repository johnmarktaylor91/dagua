You are an ADVERSARIAL reviewer for the dagua r74 fidelity sprint plan. A separate Anthropic Claude/Opus
adversarial reviewer is tearing into the SAME plan in parallel -- find what it misses. Your job is to
BREAK the plan before any code is written: refute premises, expose double-counting, predict which
"wins" will fail re-benchmark, and flag regression risk. Reading actual source beats reasoning.

READ:
- The plan: /home/jtaylor/projects/dagua/.project-context/research/sprint_rng_matching/r74_PLAN.md
- The research it rests on: /tmp/r74_O{1..6}_findings.md (Opus) and /tmp/r74_CX{1..6}_findings.md (Codex).
- The actual code/source it cites (dagua pipelines + /home/jtaylor/projects/_references/{graphviz,ogdf,igraph}).
- Verdict data: /home/jtaylor/projects/dagua/eval_output/fidelity_definitive_r73/per_combo.json.

GUARDRAILS the plan must honor (call out any violation): NEVER LAUNDER (3Q must pass 0/40 controls);
NO RUNTIME DELEGATION (reimpl must not import/call the reference at runtime); VERIFY ON BENCHMARK PATH
not direct pipeline calls; MATCH params+seed to reference; a FLOOR claim needs FP-chaos EVIDENCE.

ANSWER CONCRETELY (cite file:line / per_combo.json fields / source):
1. p_neg2 (A1): is graphviz's clamp REAL in source (cite the lines)? Are sfdp_p_neg2 reference rows
   actually identical to sfdp_default reference rows in the data? After removing combos that are ALSO
   disconnected or floor, what is the HONEST flip count (not 52)?
2. DOUBLE-COUNTING: quantify overlap between A1 (p_neg2) and B1 (sfdp disconnected ~48). Net unique sfdp.
3. REGRESSION RISK: for B1/B2 disconnected packing and C1 sugiyama LP-objective change, can the fix
   regress a currently rung1-3 combo, or a currently-divergent-but-dagua-BETTER (D<R) combo, to worse?
   Name the specific guard each implementation needs.
4. SGD2 (D1): does a real, completable reference for sgd2_multi_with_crossing actually exist, or is it
   structural-by-design? Resolve from the data/store (sibling variants, ref row status) so we don't burn
   compute on a rerun that reproduces "no reference".
5. ORDERING/COLLISION: which fixes share files and MUST serialize; what's the safe implement+verify order.
6. KILL LIST: which proposed fixes are likely net-negative or not worth it, and which are safe high-conf
   wins. Per fix give KEEP / REVISE / KILL + one-line reason.

OUTPUT CONTRACT: READ-ONLY. Do NOT edit any repository file. Write your full critique to
/tmp/r74_CX_adversarial_findings.md, then return a tight (<=450 word) verdict: the KEEP/REVISE/KILL
table, the honest net-combo estimate (realistic, not summed), the regression guards, the sgd2 verdict,
and the safe implementation order. Be ruthless and specific.
