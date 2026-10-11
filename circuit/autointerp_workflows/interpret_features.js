export const meta = {
  name: 'autointerp-features',
  description: 'Interpret a batch of copy-task transcoder features (one agent each); a skeptic tries to refute each "explained" verdict',
  phases: [
    { title: 'Interpret', detail: 'one agent per feature: profile, predicates, held-out score, verdict' },
    { title: 'Skeptic', detail: 'refute "explained" verdicts with counterexamples' },
  ],
}

// Claude Code Workflow script: activation-based interpretation, one agent per transcoder feature.
// Run from Claude Code with the Workflow tool:
//   Workflow({scriptPath: "circuit/autointerp_workflows/interpret_features.js", args: {repo, python, store, out, items}})
// See circuit/autointerp_workflows/README.md.

// args: {repo, python, store, out, items}
const REPO = args.repo
const PY = args.python || 'python'
const STORE = args.store   // store path relative to repo
const OUT = args.out       // results directory relative to repo
const CLI = `${PY} -m circuit.autointerp`

const RESULT = {
  type: 'object',
  properties: {
    gate: { type: 'string' },
    feature: { type: 'integer' },
    verdict: { type: 'string', enum: ['explained', 'partially_explained', 'polysemantic', 'too_sparse', 'dense_uninterpretable', 'no_explanation'] },
    label: { type: 'string' },
    predicate: { type: ['string', 'null'] },
    explanation: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    test_precision: { type: ['number', 'null'] },
    test_recall: { type: ['number', 'null'] },
    test_f1: { type: ['number', 'null'] },
    test_mass_recall: { type: ['number', 'null'] },
    n_hypotheses_scored: { type: 'integer' },
  },
  required: ['gate', 'feature', 'verdict', 'label', 'predicate', 'explanation', 'confidence', 'n_hypotheses_scored'],
}

const SKEPTIC = {
  type: 'object',
  properties: {
    upheld: { type: 'boolean', description: 'true only if the label/predicate/explanation survive your checks' },
    issues: { type: 'array', items: { type: 'string' }, description: 'concrete problems found, each with evidence (scores or counterexample steps)' },
    revised_verdict: { type: 'string', enum: ['explained', 'partially_explained', 'polysemantic', 'no_explanation'] },
    revised_label: { type: ['string', 'null'], description: 'corrected label if the original is wrong or overclaims, else null' },
    better_predicate: { type: ['string', 'null'], description: 'a predicate that fixes a systematic miss, if you found one' },
    better_predicate_test_f1: { type: ['number', 'null'] },
  },
  required: ['upheld', 'issues', 'revised_verdict', 'revised_label', 'better_predicate', 'better_predicate_test_f1'],
}

const TOOLS = `Work in ${REPO} (cd there first) and use ONLY this tool; do not read store files directly and do not edit any files:
  ${CLI} variables                                   # predicate variables and helpers
  ${CLI} profile ${STORE} GATE FEATURE               # activation profile + examples (dev split)
  ${CLI} score ${STORE} GATE FEATURE "PREDICATE" [--split test]    # precision / recall / F1 / mass recall
  ${CLI} misses ${STORE} GATE FEATURE "PREDICATE"    # missed activations and false positives, rendered as sequences
Each command takes a few seconds. Useful helpers beyond the per-step variables: history lookups orig(k), rel_tok(d), out_at(d), and prefix sets seen([...]) / n_seen([...]) (inputs before this step) and in_orig([...]) (any original token) for features that depend on which tokens appeared earlier.`

const TASK = `Copy task: each sequence is L original tokens (letters a-z, digits 0-3; L = 3..9), then <DEL> at step L, then the model's own outputs fed back as inputs; at step t >= L the model outputs original token t - L. The feature is from a transcoder on one GRU gate (reset / update) or the candidate (hidden). Explain WHAT THE FEATURE FIRES ON, purely from its activations over 1M sequences, in the style of Anthropic's Towards Monosemanticity. No causal or circuit analysis.`

function interpretPrompt(f) {
  return `${TASK}

Feature: gate "${f.gate}", feature ${f.feature} (active on ${f.n_sequences} of 1M sequences, ${f.n_activations} activations).

${TOOLS}

Procedure:
1. Run variables and profile (GATE=${f.gate}, FEATURE=${f.feature}). Read the rate/lift/mass tables, including the history tables (prev2_tok, next_out, first_tok), and the examples from each activation quartile.
2. Form hypotheses as boolean predicates and score them on the dev split. Use misses on your best predicate to see what it gets wrong, then refine. Start simple and add conditions only when they clearly improve F1 or mass recall. Score at most ~12 predicates, and prefer the simplest one within a few points of the best.
3. Choose a verdict:
   - explained: one coherent predicate with F1 >= 0.8 (or mass recall >= 0.8 with high precision)
   - partially_explained: a coherent description captures most activation mass, but precision or recall is clearly incomplete
   - polysemantic: needs several unrelated disjoint clauses; record the disjunction and name the components
   - too_sparse: active on fewer than ~20 sequences
   - dense_uninterpretable: fires on a large fraction of all steps with no capturable structure
   - no_explanation: otherwise
   Do not inflate. In the explanation, say what modulates magnitude and what it does NOT fire on, and only make magnitude claims you have checked.
4. Record (this scores the predicate on the held-out TEST split, which is what gets reported):
  ${CLI} record ${STORE} ${f.gate} ${f.feature} '<JSON>' --out ${OUT}
  JSON = {"label": ..., "predicate": ... or null, "verdict": ..., "confidence": ..., "explanation": ...}. Mind shell quoting: inside the JSON write tok(\\"x\\") or use token indices.
5. Return the structured result, copying the test precision/recall/F1/mass recall printed by record.`
}

function skepticPrompt(f, r) {
  return `${TASK}

You are a SKEPTIC. Another agent concluded that feature gate "${f.gate}" feature ${f.feature} is EXPLAINED:
  label: ${r.label}
  predicate: ${r.predicate}
  explanation: ${r.explanation}
  reported test scores: precision ${r.test_precision}, recall ${r.test_recall}, F1 ${r.test_f1}, mass recall ${r.test_mass_recall}

${TOOLS}

Try to REFUTE it, using GATE=${f.gate}, FEATURE=${f.feature}:
- Check each factual claim in the label and explanation (for example "never fires in the copy phase" or "magnitude peaks at X") by scoring targeted predicates.
- Run misses on the predicate, with --split test as well. Look for SYSTEMATIC false positives or misses (a token, position, length or history pattern), not scattered noise.
- Check whether the label overclaims relative to the predicate.
- If you find a systematic gap, try a better predicate and report its TEST F1.
Uphold only if the claims check out and no systematic counterexample class remains. A small amount of scattered noise is fine. Default to not upheld if the evidence is mixed. Do not record anything; just return your verdict.`
}

// args: compact [gate initial, feature, n_sequences, n_activations] tuples
const GATES = { r: 'reset', u: 'update', h: 'hidden' }
// items: [[gate initial r|u|h, feature, n_sequences, n_activations], ...]
const ITEMS = args.items.map(([g, f, n, a]) => ({ gate: GATES[g] || g, feature: f, n_sequences: n, n_activations: a }))
log(`batch of ${ITEMS.length} features`)

const results = await pipeline(ITEMS,
  f => agent(interpretPrompt(f), { label: `${f.gate}/${f.feature}`, phase: 'Interpret', schema: RESULT }),
  (r, f) => {
    if (!r || r.verdict !== 'explained') return { interp: r, skeptic: null }
    return agent(skepticPrompt(f, r), { label: `skeptic ${f.gate}/${f.feature}`, phase: 'Skeptic', schema: SKEPTIC })
      .then(s => ({ interp: r, skeptic: s }))
  })

const done = results.filter(Boolean)
const counts = {}
for (const x of done) {
  const v = x.interp ? x.interp.verdict : 'failed'
  counts[v] = (counts[v] || 0) + 1
}
const sk = done.filter(x => x.skeptic)
const failed = ITEMS.length - done.filter(x => x.interp).length
log(`verdicts: ${JSON.stringify(counts)}; skeptic upheld ${sk.filter(x => x.skeptic.upheld).length}/${sk.length}; failed ${failed}`)
return { verdicts: counts, skeptic_upheld: sk.filter(x => x.skeptic.upheld).length, skeptic_total: sk.length, failed }
