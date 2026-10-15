export const meta = {
  name: 'autointerp-rl-magnitude',
  description: 'Magnitude explanations for RL transcoder features and GRU units (one agent each), skeptic check on explained verdicts',
  phases: [
    { title: 'Interpret', detail: 'one agent per component: profile, free fit, one-sentence label + short form, held-out R^2, verdict' },
    { title: 'Skeptic', detail: 'refute "explained" magnitude explanations' },
  ],
}

// Claude Code Workflow script: magnitude interpretation for the RL agent (docs/autointerp_guide.md, section 4).
// Same protocol as circuit/copy_task/autointerp_workflows/interpret_magnitude.js; the tool is circuit.rl_task.autointerp_magnitude.
//   Workflow({scriptPath: "circuit/rl_task/autointerp_workflows/interpret_magnitude.js",
//             args: {repo, python, store, out, items: [[gate, id], ...]}})
// gate: reset / update / hidden (transcoder features) or unit_reset / unit_update / unit_candidate (GRU units' raw values).
const REPO = args.repo
const PY = args.python
const STORE = args.store
const OUT = args.out
const CLI = `PYTHONPATH=. ${PY} -m circuit.rl_task.autointerp_magnitude`
// observed_only: terms may not read the model's own outputs (value, choice, model_act); the tool enforces it
const OBS = args.observed_only ? ' --observed_only' : ''
const OBS_TEXT = args.observed_only ? `
OBSERVED-ONLY RUN: terms may NOT use the model's own outputs (value, choice, model_act); the tool rejects them (always pass --observed_only). Instead use what the agent could infer: the variable belief (ideal-observer P(S1 good) from the episode's observed history; belief_lo = its log-odds) and belief_cover (how much history it rests on), plus trial history. Explain the component from observed quantities alone: the model's own outputs are what the components produce, not what drives them.` : ''

const RESULT = {
  type: 'object',
  properties: {
    gate: { type: 'string' },
    feature: { type: 'integer' },
    verdict: { type: 'string', enum: ['explained', 'partially_explained', 'no_explanation'] },
    label: { type: 'string', description: 'the interpretation: one plain sentence, at most ~15 words, that says what the short form says' },
    short_terms: { type: 'string', description: 'the recorded explanation: at most 3 simple terms separated by ;; (scored)' },
    terms: { type: 'string', description: 'the full fit used to find the hypothesis (up to 8 terms; supporting evidence only)' },
    explanation: { type: 'string' },
    uses_future: { type: 'boolean', description: 'true if a recorded term uses a FUTURE variable' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    test_r2_beyond_timing: { type: ['number', 'null'], description: 'of the SHORT form, as printed by record' },
    full_fit_test_r2_beyond_timing: { type: ['number', 'null'] },
    ceiling_all_tables: { type: ['number', 'null'], description: 'the "all" reference from the profile' },
    n_attempts_scored: { type: 'integer' },
  },
  required: ['gate', 'feature', 'verdict', 'label', 'short_terms', 'terms', 'explanation', 'uses_future', 'confidence',
             'test_r2_beyond_timing', 'full_fit_test_r2_beyond_timing', 'ceiling_all_tables', 'n_attempts_scored'],
}

const SKEPTIC = {
  type: 'object',
  properties: {
    upheld: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' }, description: 'concrete problems, each with evidence (scores or misses output)' },
    revised_verdict: { type: 'string', enum: ['explained', 'partially_explained', 'no_explanation'] },
    revised_label: { type: ['string', 'null'], description: 'a corrected one-sentence label (at most ~15 words), else null' },
    better_short_terms: { type: ['string', 'null'], description: 'a better SHORT form (passes --short), else null' },
    better_short_terms_test_r2_beyond_timing: { type: ['number', 'null'] },
  },
  required: ['upheld', 'issues', 'revised_verdict', 'revised_label', 'better_short_terms', 'better_short_terms_test_r2_beyond_timing'],
}

const TOOLS = `Work in ${REPO} (cd there first) and use ONLY these commands; do not read store or data files directly and do not edit any files:
  ${CLI} variables                                       # per-step variables (with when each is observable) and helpers
  ${CLI} profile ${STORE} GATE ID${OBS}                        # timing profile, residual tables, reference scores, examples
  ${CLI} score ${STORE} GATE ID "TERM ;; TERM ;; ..."${OBS} [--short] [--split test]   # fits one weight per term on dev, R^2 beyond timing
  ${CLI} misses ${STORE} GATE ID "TERMS"${OBS} [--short]       # worst-fit windows and systematic errors by phase x previous trial
Each command takes a few seconds. Terms are numpy expressions over the per-step variables (phase, in_rew, action, state, common, prev_reward, prev_action, prev_state, prev_common, stay, good, chose_good, since_rev, ep_trial, ep_dec, belief, belief_lo, ...) and helpers: tv('col', d) (trial-level column d trials away in the window), isin(col, [..]), last_is('a1 S2 u') (previous trial = that choice, state reached, r rewarded / u not), table(col, {value: weight}), D1 / GO / D2, S1 / S2, np. Booleans count as 0/1. Each term gets its own fitted weight (plus a timing baseline per (phase, episode decile) and an intercept), so write shapes, not scales; make a term phase-specific by multiplying by (phase == D1) etc.
SHORT FORM (--short): the recorded explanation. At most 3 terms; no hand-set non-integer constants; no table(); at most one last_is per term; at most three factors per term. The tool rejects anything else. A free fit is a way to find hypotheses, not an interpretation: the interpretation is a one-sentence label plus the short form that says the same thing.`

const TASK = `Two-step RL task (a GRU agent). Each trial has 3 steps: D1 (fixate; the previous trial's reward is shown), GO (choose a1 or a2), D2 (the reached state is shown; a1 -> S1 and a2 -> S2 with p = 0.8 "common", else "rare"). The reached state pays a reward with p = 0.9 if it is the currently good state, 0.1 otherwise; the reward is seen at the next D1. The good state reverses a few times per 100-trial episode. A model-based agent repeats after common-rewarded / rare-unrewarded trials and switches after common-unrewarded / rare-rewarded ones. Data: 5-trial windows from 12000 episodes.
The component is either a transcoder feature (gate reset / update / hidden = candidate) whose activation is graded (how strongly it fires carries information), or a GRU unit's RAW gate value (gate unit_reset = r, unit_update = z, unit_candidate = n; every step, no threshold). Your job: explain what drives its MAGNITUDE beyond timing (phase and time in the episode are in the baseline already): which trial histories, choices, states, rewards or internal quantities raise or lower it, by phase. FUTURE variables (this trial's reward, the state about to be reached, the next choice) may be used but the agent cannot observe them at that step; if you use one, also score the closest observed-only version and say which you recorded. No causal analysis.${OBS_TEXT}`

// args.evidence (optional): {"gate/id": text} intervention evidence for components being re-explained
const EVIDENCE = f => (args.evidence && args.evidence[`${f.gate}/${f.id}`]) ? `
INTERVENTION EVIDENCE (from tests on the real model; why this component is being re-explained):
${args.evidence[`${f.gate}/${f.id}`]}
Your explanation must capture what drives the component WITHIN the situation where the model needs it (score your short form there too, by multiplying terms by that situation's condition, and make sure the sentence says it).
` : ''

function interpretPrompt(f) {
  return `${TASK}

Component: gate "${f.gate}", id ${f.id}.${EVIDENCE(f)}

${TOOLS}

Procedure:
1. Run variables and profile (GATE=${f.gate}, ID=${f.id}). Read the timing profile, the residual tables (which previous-trial patterns, choices / states, policy or value deciles, hidden good state raise / lower it, by phase), the reference scores, and the examples. "all" is the ceiling for additive explanations over the profile's cues; "all observed" uses only observed cues.
2. Find the hypothesis with a free fit: up to 8 terms, scored on the dev split. Typical shapes: indicator terms for previous-trial patterns at a phase ((phase == D1) * last_is('a1 S2 u')); reward x state of the last trial; the trial before (tv('reward', -2) etc.); the choice being made or just made; the observer's belief (belief_lo) as a graded term; trials since a reversal. Use misses to find systematic errors. This step is for finding what matters.
3. Distil it into the INTERPRETATION: one plain sentence (at most ~15 words) that captures the crux, plus a SHORT form (at most 3 terms, --short) that says exactly what the sentence says. Score the short form with --short on dev, and compare it with the free fit. If the crux needs more than the short form allows, say so: that is a finding, not something to hide with more terms. Score at most ~15 attempts in total.
4. Verdict, from the TEST score of the SHORT form (printed by record):
   - explained: short-form test R^2 beyond timing >= 0.5, or >= 0.4 and at least 80% of your free fit's, and the sentence matches the short form
   - partially_explained: >= 0.25, or the sentence captures a clear part
   - no_explanation: otherwise
   Do not inflate. Only state claims you have checked by scoring.
5. Record (scores both on the held-out TEST split; the short form is what gets reported):
  ${CLI} record ${STORE} ${f.gate} ${f.id} '<JSON>' --out ${OUT}${OBS}
  JSON = {"label": ..., "short_terms": "TERM ;; TERM", "terms": "FULL FIT TERMS", "verdict": ..., "confidence": ..., "uses_future": true|false, "explanation": ...}. The explanation is about this component only (what drives it, what does not, what the free fit adds beyond the short form). Mind shell quoting: inside the JSON write last_is(\\"a1 S2 u\\") and tv(\\"reward\\", -1) (escaped double quotes), never single quotes.
6. Return the structured result with the short-form and free-fit test R^2 printed by record and the "all" ceiling from the profile.`
}

function skepticPrompt(f, r) {
  return `${TASK}

You are a SKEPTIC. Another agent concluded that the magnitude of gate "${f.gate}" id ${f.id} is EXPLAINED:
  label (the interpretation): ${r.label}
  short form (scored): ${r.short_terms}
  free fit (evidence): ${r.terms}
  explanation: ${r.explanation}
  uses future variables: ${r.uses_future}
  reported test R^2 beyond timing: short form ${r.test_r2_beyond_timing}, free fit ${r.full_fit_test_r2_beyond_timing} (additive ceiling: ${r.ceiling_all_tables})

${TOOLS}

Try to REFUTE it (GATE=${f.gate}, ID=${f.id}):
- Does the one-sentence label say what the short form says, and nothing it does not? A reader of the label alone should be able to predict when the component is high or low.
- Check each factual claim in the label and explanation (e.g. "raised after rare rewarded trials", "only at D1", "grows with the evidence for S2") by scoring targeted short-form variants with --short, on --split test as well.
- Run misses on the short form. Look for SYSTEMATIC errors (a phase, a previous-trial pattern, a belief range) rather than scattered noise.
- Is the short form a real description or disguised curve fitting (terms that only make sense as fitted corrections)?
- If a term uses a future variable, check whether an observed-only version does as well; if so, the label should say that.
- If you find a better short form (it must pass --short), report it and its TEST R^2 beyond timing; it will replace the recorded one, so give a matching revised label too.
Uphold only if the claims check out and no large systematic error class remains. Default to not upheld if the evidence is mixed. revised_verdict is your verdict on the component given your corrections. Do not record anything; just return your verdict.`
}

const ITEMS = args.items.map(([gate, id]) => ({ gate, id }))
log(`${ITEMS.length} components`)
const results = await pipeline(ITEMS,
  f => agent(interpretPrompt(f), { label: `${f.gate}/${f.id}`, phase: 'Interpret', schema: RESULT }),
  (r, f) => {
    if (!r || r.verdict !== 'explained') return { item: f, interp: r, skeptic: null }
    return agent(skepticPrompt(f, r), { label: `skeptic ${f.gate}/${f.id}`, phase: 'Skeptic', schema: SKEPTIC })
      .then(s => ({ item: f, interp: r, skeptic: s }))
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
         failed: ITEMS.length - done.filter(x => x.interp).length }
