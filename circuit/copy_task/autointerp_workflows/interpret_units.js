export const meta = {
  name: 'autointerp-units',
  description: 'Neuron baseline: interpret thresholded GRU gate units with the same agent protocol as transcoder features; skeptic on "explained"',
  phases: [
    { title: 'Interpret', detail: 'one agent per GRU unit' },
    { title: 'Skeptic', detail: 'refute "explained" verdicts' },
  ],
}

// Claude Code Workflow script: activation-based interpretation, one agent per thresholded GRU unit (neuron baseline).
// Run from Claude Code with the Workflow tool:
//   Workflow({scriptPath: "circuit/copy_task/autointerp_workflows/interpret_units.js", args: {repo, python, store, out, items}})
// See circuit/copy_task/autointerp_workflows/README.md.

// args: {repo, python, store, out, items}
const REPO = args.repo
const PY = args.python || 'python'
const STORE = args.store   // store path relative to repo
const OUT = args.out       // results directory relative to repo
const CLI = `${PY} -m circuit.copy_task.autointerp`

const RESULT = {
  type: 'object',
  properties: {
    gate: { type: 'string' }, feature: { type: 'integer' },
    verdict: { type: 'string', enum: ['explained', 'partially_explained', 'polysemantic', 'too_sparse', 'dense_uninterpretable', 'no_explanation'] },
    label: { type: 'string' }, predicate: { type: ['string', 'null'] }, explanation: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    test_precision: { type: ['number', 'null'] }, test_recall: { type: ['number', 'null'] },
    test_f1: { type: ['number', 'null'] }, test_mass_recall: { type: ['number', 'null'] },
    n_hypotheses_scored: { type: 'integer' },
  },
  required: ['gate', 'feature', 'verdict', 'label', 'predicate', 'explanation', 'confidence', 'n_hypotheses_scored'],
}
const SKEPTIC = {
  type: 'object',
  properties: {
    upheld: { type: 'boolean' }, issues: { type: 'array', items: { type: 'string' } },
    revised_verdict: { type: 'string', enum: ['explained', 'partially_explained', 'polysemantic', 'no_explanation'] },
    revised_label: { type: ['string', 'null'] }, better_predicate: { type: ['string', 'null'] },
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

const TASK = `Copy task: each sequence is L original tokens (letters a-z, digits 0-3; L = 3..9), then <DEL> at step L, then the model's own outputs fed back as inputs; at step t >= L the model outputs original token t - L. The unit you are interpreting is ONE COORDINATE of the GRU itself (not a dictionary feature): update gate z (neuron_update), reset gate r (neuron_reset), or candidate n at its high end (neuron_cand_pos) or low end (neuron_cand_neg). It is thresholded so it counts as "active" only on its top fraction of steps; the activation is how far it exceeds the threshold. Explain WHEN IT IS ACTIVE, purely from its activations over 200k sequences, in the style of Anthropic's Towards Monosemanticity. No causal or circuit analysis.`

function interpretPrompt(f) {
  return `${TASK}

Unit: gate "${f.gate}", index ${f.feature} (active on ${f.n_sequences} of 200k sequences, ${f.n_activations} activations).

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

You are a SKEPTIC. Another agent concluded that unit gate "${f.gate}" index ${f.feature} is EXPLAINED:
  label: ${r.label}
  predicate: ${r.predicate}
  explanation: ${r.explanation}
  reported test scores: precision ${r.test_precision}, recall ${r.test_recall}, F1 ${r.test_f1}, mass recall ${r.test_mass_recall}

${TOOLS}

Try to REFUTE it, using GATE=${f.gate}, FEATURE=${f.feature}:
- Check each factual claim in the label and explanation by scoring targeted predicates.
- Run misses on the predicate, with --split test as well. Look for SYSTEMATIC false positives or misses, not scattered noise.
- Check whether the label overclaims relative to the predicate.
- If you find a systematic gap, try a better predicate and report its TEST F1.
Uphold only if the claims check out and no systematic counterexample class remains. Default to not upheld if the evidence is mixed. Do not record anything; just return your verdict.`
}

// items: [[gate e.g. neuron_update, unit, n_sequences, n_activations], ...]
const ITEMS = args.items.map(([g, f, n, a]) => ({ gate: g, feature: f, n_sequences: n, n_activations: a }))
log(`batch of ${ITEMS.length} units`)
const results = await pipeline(ITEMS,
  f => agent(interpretPrompt(f), { label: `${f.gate}/${f.feature}`, phase: 'Interpret', schema: RESULT }),
  (r, f) => {
    if (!r || r.verdict !== 'explained') return { interp: r, skeptic: null }
    return agent(skepticPrompt(f, r), { label: `skeptic ${f.gate}/${f.feature}`, phase: 'Skeptic', schema: SKEPTIC })
      .then(s => ({ interp: r, skeptic: s }))
  })
const done = results.filter(Boolean)
const counts = {}
for (const x of done) { const v = x.interp ? x.interp.verdict : 'failed'; counts[v] = (counts[v] || 0) + 1 }
const sk = done.filter(x => x.skeptic)
log(`verdicts: ${JSON.stringify(counts)}; skeptic upheld ${sk.filter(x => x.skeptic.upheld).length}/${sk.length}`)
return { verdicts: counts, skeptic_upheld: sk.filter(x => x.skeptic.upheld).length, skeptic_total: sk.length, failed: ITEMS.length - done.filter(x => x.interp).length }
