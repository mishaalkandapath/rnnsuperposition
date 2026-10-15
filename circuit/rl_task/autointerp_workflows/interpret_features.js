export const meta = {
  name: 'autointerp-rl-features',
  description: 'Interpret a batch of RL (two-step task) transcoder features (one agent each); a skeptic tries to refute each "explained" verdict',
  phases: [
    { title: 'Interpret', detail: 'one agent per feature: profile, predicates, held-out score, verdict' },
    { title: 'Skeptic', detail: 'refute "explained" verdicts with counterexamples' },
  ],
}

// Claude Code Workflow script: activation-based interpretation of RL-task transcoder features.
// Same protocol as circuit/copy_task/autointerp_workflows/interpret_features.js; the tool is circuit.rl_task.autointerp.
//   Workflow({scriptPath: "circuit/rl_task/autointerp_workflows/interpret_features.js", args: {repo, python, store, out, items}})
// items: [[gate initial r|u|h, feature, n_windows, n_activations], ...] (python -m circuit.copy_task.autointerp_runs items STORE OUT)

const REPO = args.repo
const PY = args.python || 'python'
const STORE = args.store   // store path relative to repo (with its step table built: circuit.rl_task.autointerp build)
const OUT = args.out       // results directory relative to repo
const CLI = `PYTHONPATH=. ${PY} -m circuit.rl_task.autointerp`

const RESULT = {
  type: 'object',
  properties: {
    gate: { type: 'string' },
    feature: { type: 'integer' },
    verdict: { type: 'string', enum: ['explained', 'partially_explained', 'polysemantic', 'too_sparse', 'dense_uninterpretable', 'no_explanation'] },
    label: { type: 'string' },
    predicate: { type: ['string', 'null'] },
    explanation: { type: 'string' },
    uses_future: { type: 'boolean', description: 'true if the recorded predicate uses a FUTURE variable (not yet observable at that step)' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    test_precision: { type: ['number', 'null'] },
    test_recall: { type: ['number', 'null'] },
    test_f1: { type: ['number', 'null'] },
    test_mass_recall: { type: ['number', 'null'] },
    n_hypotheses_scored: { type: 'integer' },
  },
  required: ['gate', 'feature', 'verdict', 'label', 'predicate', 'explanation', 'uses_future', 'confidence', 'n_hypotheses_scored'],
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

const TOOLS = `Work in ${REPO} (cd there first) and use ONLY this tool; do not read store or trace files directly and do not edit any files:
  ${CLI} variables                                   # predicate variables (with when each is observable) and helpers
  ${CLI} profile ${STORE} GATE FEATURE               # activation profile + example windows (dev split)
  ${CLI} score ${STORE} GATE FEATURE "PREDICATE" [--split test]    # precision / recall / F1 / mass recall
  ${CLI} misses ${STORE} GATE FEATURE "PREDICATE"    # missed activations and false positives, rendered as windows
Each command takes about a second. Predicates are numpy expressions over per-step variables (phase, rel, in_rew, action, state, common, reward, prev_reward, prev_common, stay, next_stay, good, chose_good, value, choice, ...), combined with & | ~; tv('col', d) looks up a trial-level variable d trials back (d < 0) or ahead (d > 0) in the same window.`

const TASK = `Two-step RL task (a GRU agent). Each trial has 3 steps: D1 (DELAY_1: fixate; the previous trial's reward is shown), GO (choose action a1 or a2), D2 (DELAY_2: the reached state is shown; a1 leads to S1 and a2 to S2 with p = 0.8 "common", else the other state, "rare"). The reached state pays a reward with p = 0.9 if it is the currently good state, 0.1 otherwise; the reward is seen at the next trial's D1. Which state is good reverses a few times per episode, so a model-based agent should repeat a choice after common-rewarded or rare-unrewarded trials and switch after common-unrewarded or rare-rewarded ones. Data: 5-trial windows from 12000 episodes (3 context trials, a core trial, the next trial), each step a row. The feature is from a transcoder on one GRU gate (reset / update) or the candidate (hidden). Explain WHAT THE FEATURE FIRES ON, purely from its activations, in the style of Anthropic's Towards Monosemanticity. No causal or circuit analysis.

FUTURE variables (marked in variables: e.g. this trial's reward, the state about to be reached, the next choice) are allowed, but the agent cannot observe them at that step. If your best predicate uses one, also score the closest predicate with only observed variables (e.g. the observed history, or the hidden good state, which the agent may infer) and the agent's own upcoming choice; say in the explanation which you recorded and why. The agent's own upcoming choice (action at D1/GO, next_stay) is a legitimate thing for it to compute; a future random outcome (reward, rare vs common before D2) is not knowable, so a feature tracking it is really tracking an observed correlate.`

function interpretPrompt(f) {
  return `${TASK}

Feature: gate "${f.gate}", feature ${f.feature} (active in ${f.n_sequences} of 155737 windows, ${f.n_activations} activations).

${TOOLS}

Procedure:
1. Run variables and profile (GATE=${f.gate}, FEATURE=${f.feature}). Read the rate/lift/mass tables (phase, trial relative to the core, inputs, choice/state/transition/reward of this and the previous trial, stay/switch, the hidden good state, the model's value and policy) and the example windows from each activation quartile.
2. Form hypotheses as boolean predicates and score them on the dev split. Use misses on your best predicate to see what it gets wrong, then refine. Start simple and add conditions only when they clearly improve F1 or mass recall. Score at most ~12 predicates, and prefer the simplest one within a few points of the best.
3. Choose a verdict:
   - explained: one coherent predicate with F1 >= 0.8 (or mass recall >= 0.8 with high precision)
   - partially_explained: a coherent description captures most activation mass, but precision or recall is clearly incomplete
   - polysemantic: needs several unrelated disjoint clauses; record the disjunction and name the components
   - too_sparse: active in fewer than ~20 windows
   - dense_uninterpretable: fires on a large fraction of all steps with no capturable structure
   - no_explanation: otherwise
   Do not inflate. If the feature fires on most steps of a phase and the best predicate is only a phase / position, say so: it probably carries information in its magnitude. In the explanation, say what modulates magnitude and what it does NOT fire on, and only make magnitude claims you have checked.
4. Record (this scores the predicate on the held-out TEST split, which is what gets reported):
  ${CLI} record ${STORE} ${f.gate} ${f.feature} '<JSON>' --out ${OUT}
  JSON = {"label": ..., "predicate": ... or null, "verdict": ..., "confidence": ..., "uses_future": true|false, "explanation": ...}. Mind shell quoting: inside the JSON write tv(\\"reward\\", -1) (escaped double quotes), never single quotes.
5. Return the structured result, copying the test precision/recall/F1/mass recall printed by record.`
}

function skepticPrompt(f, r) {
  return `${TASK}

You are a SKEPTIC. Another agent concluded that feature gate "${f.gate}" feature ${f.feature} is EXPLAINED:
  label: ${r.label}
  predicate: ${r.predicate}
  explanation: ${r.explanation}
  uses future variables: ${r.uses_future}
  reported test scores: precision ${r.test_precision}, recall ${r.test_recall}, F1 ${r.test_f1}, mass recall ${r.test_mass_recall}

${TOOLS}

Try to REFUTE it, using GATE=${f.gate}, FEATURE=${f.feature}:
- Check each factual claim in the label and explanation (for example "never fires at GO" or "only after rare transitions") by scoring targeted predicates.
- Run misses on the predicate, with --split test as well. Look for SYSTEMATIC false positives or misses (a phase, a trial position, a choice / transition / reward pattern, the good state), not scattered noise.
- If the predicate uses a future variable, check whether an observed-only predicate explains the feature as well; if so, the label should say that.
- Check whether the label overclaims relative to the predicate.
- If you find a systematic gap, try a better predicate and report its TEST F1.
Uphold only if the claims check out and no systematic counterexample class remains. A small amount of scattered noise is fine. Default to not upheld if the evidence is mixed. Do not record anything; just return your verdict.`
}

const GATES = { r: 'reset', u: 'update', h: 'hidden' }
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
