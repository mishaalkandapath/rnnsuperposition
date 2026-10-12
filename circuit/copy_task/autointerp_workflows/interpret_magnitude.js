export const meta = {
  name: 'autointerp-magnitude',
  description: 'Magnitude explanations for dense candidate features (one agent each), skeptic check on explained verdicts',
  phases: [
    { title: 'Interpret', detail: 'one agent per dense feature: profile, terms, held-out R^2 beyond timing, verdict' },
    { title: 'Skeptic', detail: 'refute "explained" magnitude explanations' },
  ],
}

// Claude Code Workflow script: magnitude interpretation of dense features (see docs/autointerp_guide.md, section 4).
// Needs circuit/copy_task/magnitude_scores.py output for the gate (reference scores and tables in each profile).
//   Workflow({scriptPath: "circuit/copy_task/autointerp_workflows/interpret_magnitude.js",
//             args: {repo, python, store, out, features: [feature ids]}})
const REPO = args.repo
const PY = args.python
const STORE = args.store
const OUT = args.out
const CLI = `PYTHONPATH=. ${PY} -m circuit.copy_task.autointerp_magnitude`

const RESULT = {
  type: 'object',
  properties: {
    feature: { type: 'integer' },
    verdict: { type: 'string', enum: ['explained', 'partially_explained', 'no_explanation'] },
    label: { type: 'string', description: 'short magnitude label, at most ~12 words' },
    terms: { type: 'string', description: 'the recorded terms, separated by ;;' },
    explanation: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    test_r2_beyond_timing: { type: ['number', 'null'] },
    ceiling_all_tables: { type: ['number', 'null'], description: 'the "+all" reference from the profile' },
    n_attempts_scored: { type: 'integer' },
  },
  required: ['feature', 'verdict', 'label', 'terms', 'explanation', 'confidence', 'test_r2_beyond_timing', 'ceiling_all_tables', 'n_attempts_scored'],
}

const SKEPTIC = {
  type: 'object',
  properties: {
    upheld: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' }, description: 'concrete problems, each with evidence (scores or misses output)' },
    revised_verdict: { type: 'string', enum: ['explained', 'partially_explained', 'no_explanation'] },
    revised_label: { type: ['string', 'null'] },
    better_terms: { type: ['string', 'null'] },
    better_terms_test_r2_beyond_timing: { type: ['number', 'null'] },
  },
  required: ['upheld', 'issues', 'revised_verdict', 'revised_label', 'better_terms', 'better_terms_test_r2_beyond_timing'],
}

const TOOLS = `Work in ${REPO} (cd there first) and use ONLY these commands; do not read store files directly and do not edit any files:
  ${CLI} variables                                          # per-step variables and helpers
  ${CLI} profile ${STORE} hidden FEATURE                    # timing profile, fitted token tables, reference scores, examples
  ${CLI} score ${STORE} hidden FEATURE "TERM ;; TERM ;; ..." [--split test]   # fits one weight per term on dev, R^2 beyond timing
  ${CLI} misses ${STORE} hidden FEATURE "TERMS"             # worst-fit sequences and systematic errors
Each command takes 1-3 seconds. Terms are numpy expressions over the variables (t, L, phase, in_tok, out_tok, prev_tok, ...) and helpers: tok('x'), isin(col, [..]), rel_tok(d) (input d steps away; d <= 0 looks back), out_at(d) (token output d steps ahead; in the copy phase out_at(1), out_at(2), ... are the tokens still waiting in memory), orig(k), seen(S), n_seen(S), in_orig(S), table(col, {'a': 1.0, ...}), n_in(S, ds), w_in(S, {0: 1.0, -1: 0.8}). Booleans count as 0/1. Each term gets its own fitted weight (plus a timing baseline per (L, t) and an intercept), so write shapes, not scales.`

const TASK = `Copy task: each sequence is L original tokens (letters a-z, digits 0-3; L = 3..9), then <DEL> at step L, then the model's own outputs fed back as inputs; at step t >= L the model outputs original token t - L. The feature is a DENSE candidate-transcoder feature of the GRU: it fires on most steps, so what it encodes is in its MAGNITUDE. An earlier pass explained only when it fires. Your job: explain what drives its magnitude beyond timing, in the style of a token-table explanation (which tokens, at which relative positions or in which memory slots, raise or lower it, and how that differs by phase). No causal analysis.`

function interpretPrompt(f) {
  return `${TASK}

Feature: hidden ${f}.

${TOOLS}

Procedure:
1. Run variables and profile (FEATURE=${f}). Read the timing profile, the fitted token tables (which cue and phase matter; which tokens raise / lower it), the automatic token-set fit, the reference scores, and the examples. "+all" is the ceiling for additive token explanations; the token-set score is what a compact automatic label achieves.
2. Write a compact explanation as at most 8 terms and score it on the dev split. Typical shapes: a token set at a few relative inputs (w_in / n_in), phase-specific (multiply by (phase == 0) etc.); the current output and the queue ahead in the copy phase (out_at(0), out_at(1), out_at(2)); a signed axis (one set raises, another lowers); a sign flip between phases. Use misses to find systematic errors and refine. Score at most ~12 attempts. Prefer fewer, coherent terms; stop when more terms add little.
3. Verdict, from the TEST score you record:
   - explained: test R^2 beyond timing >= 0.6, or >= 0.45 and at least 80% of the "+all" ceiling, with a coherent description
   - partially_explained: >= 0.3, or a coherent description that captures a clear part
   - no_explanation: otherwise
   Do not inflate. Only state magnitude claims you have checked by scoring.
4. Record (scores on the held-out TEST split; that is what gets reported):
  ${CLI} record ${STORE} hidden ${f} '<JSON>' --out ${OUT}
  JSON = {"label": ..., "terms": "TERM ;; TERM", "verdict": ..., "confidence": ..., "explanation": ...}. The label is short (e.g. "remembers {e,b,g,h} among the last ~4 inputs; copy phase: queue"). Mind shell quoting: inside the JSON write tok(\\"x\\") (escaped double quotes) or token indices, never single quotes.
5. Return the structured result with the test R^2 beyond timing printed by record and the "+all" ceiling from the profile.`
}

function skepticPrompt(f, r) {
  return `${TASK}

You are a SKEPTIC. Another agent concluded that hidden feature ${f}'s magnitude is EXPLAINED:
  label: ${r.label}
  terms: ${r.terms}
  explanation: ${r.explanation}
  reported test R^2 beyond timing: ${r.test_r2_beyond_timing} (ceiling with full token tables: ${r.ceiling_all_tables})

${TOOLS}

Try to REFUTE it (FEATURE=${f}):
- Check each factual claim in the label and explanation (e.g. "lowered by s/1/i", "only in the original phase", "fades over ~4 inputs") by scoring targeted term variants, on --split test as well.
- Run misses on the terms. Look for SYSTEMATIC errors (a phase, a token class, a position) rather than scattered noise.
- Check whether the label overclaims relative to the terms and scores (e.g. names tokens the terms do not use).
- If you find a systematic gap, try better terms (at most 8) and report their TEST R^2 beyond timing.
Uphold only if the claims check out and no large systematic error class remains. Default to not upheld if the evidence is mixed. Do not record anything; just return your verdict.`
}

const ITEMS = args.features
log(`${ITEMS.length} dense candidate features`)
const results = await pipeline(ITEMS,
  f => agent(interpretPrompt(f), { label: `hidden/${f}`, phase: 'Interpret', schema: RESULT }),
  (r, f) => {
    if (!r || r.verdict !== 'explained') return { feature: f, interp: r, skeptic: null }
    return agent(skepticPrompt(f, r), { label: `skeptic hidden/${f}`, phase: 'Skeptic', schema: SKEPTIC })
      .then(s => ({ feature: f, interp: r, skeptic: s }))
  })

const done = results.filter(Boolean)
const counts = {}
for (const x of done) {
  const v = x.interp ? x.interp.verdict : 'failed'
  counts[v] = (counts[v] || 0) + 1
}
const sk = done.filter(x => x.skeptic)
log(`verdicts: ${JSON.stringify(counts)}; skeptic upheld ${sk.filter(x => x.skeptic.upheld).length}/${sk.length}`)
return { verdicts: counts, skeptic_upheld: sk.filter(x => x.skeptic.upheld).length, skeptic_total: sk.length,
         results: done.map(x => ({ feature: x.feature, interp: x.interp, skeptic: x.skeptic })) }
