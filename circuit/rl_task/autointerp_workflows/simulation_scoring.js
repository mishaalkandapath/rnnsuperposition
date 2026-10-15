export const meta = {
  name: 'rl-simulation-scoring',
  description: 'Words-only explanations scored by blind simulation, for RL transcoder features and GRU units alike',
  phases: [
    { title: 'Explain', detail: 'one agent per component: activations on example windows -> a short explanation in words' },
    { title: 'Simulate', detail: 'a separate agent sees only the explanation and held-out windows, predicts activations' },
  ],
}

// Claude Code Workflow script (Towards Monosemanticity-style automated interpretability; the tool is
// circuit.rl_task.simulation_scoring). The simulator never sees activations or the explainer's examples.
//   Workflow({scriptPath: "circuit/rl_task/autointerp_workflows/simulation_scoring.js",
//             args: {repo, python, store, out, items: [[gate, id], ...]}})
// gate: reset / update / hidden (transcoder features) or unit_reset / unit_update / unit_candidate (GRU units).
const REPO = args.repo
const PY = args.python
const STORE = args.store
const OUT = args.out
const CLI = `PYTHONPATH=. ${PY} -m circuit.rl_task.simulation_scoring`

const EXPLANATION = {
  type: 'object',
  properties: {
    explanation: { type: 'string', description: 'at most 2 sentences, plain words, no code; how the 0-10 value depends on what happens in the trials' },
  },
  required: ['explanation'],
}
const SIMULATION = {
  type: 'object',
  properties: {
    correlation: { type: ['number', 'null'], description: 'as printed by score' },
    correlation_beyond_timing: { type: ['number', 'null'] },
    recorded: { type: 'boolean' },
  },
  required: ['correlation', 'correlation_beyond_timing', 'recorded'],
}

const TASK = `Two-step RL task (a GRU agent). Each trial: D1 (fixate; the previous trial's reward is shown), GO (choose a1 or a2), D2 (the state reached is shown: a1 -> S1 and a2 -> S2 with p = 0.8, "common", otherwise "rare"). The reached state pays a reward with p = 0.9 if it is the currently good state, 0.1 otherwise; which state is good is hidden and reverses a few times per 100-trial episode.`

function explainPrompt(f) {
  return `${TASK}

Work in ${REPO} (cd there first). Run exactly this command and nothing else; do not read any files or run other commands:
  ${CLI} explain_sample ${STORE} ${f.gate} ${f.id}
It prints one component's activation (0-10) at every step of 30 example windows: its 12 highest-activating windows, 12 random ones and 6 with its lowest values.

Write an explanation of what determines the value, in at most 2 sentences of plain words (no code, no formulas). Another reader will use ONLY your explanation to guess the 0-10 value at every step of new windows, so say when it is high, when it is low, and how high: which step of the trial (D1 / GO / D2), which recent events (rewards, choices, common / rare transitions, earlier trials), and time in the episode if it matters. Be concrete and short; do not hedge. Return the explanation.`
}

function simulatePrompt(f, explanation) {
  return `${TASK}

You are predicting a hidden quantity from a description of it. The description:
  "${explanation}"

Work in ${REPO} (cd there first). Use only these two commands; do not read files or run anything else:
  ${CLI} simulate_sample ${STORE} ${f.gate} ${f.id}
prints 40 windows (w0 ... w39), each a few trials with steps marked [s0], [s1], ...; the values are NOT shown.
  ${CLI} score ${STORE} ${f.gate} ${f.id} '<JSON>' --explanation '<the description above>' --out ${OUT}
records your predictions. JSON = {"w0": [v_s0, v_s1, ...], "w1": [...], ...}: one number from 0 to 10 for EVERY step of EVERY window, in order, using only the description. Mind shell quoting: no single quotes inside the JSON or the description (replace them).

Run simulate_sample, predict every step from the description alone, run score once, and return the correlation and correlation_beyond_timing it prints (recorded = true).`
}

const ITEMS = args.items.map(([gate, id]) => ({ gate, id }))
log(`${ITEMS.length} components`)
const results = await pipeline(ITEMS,
  f => agent(explainPrompt(f), { label: `explain ${f.gate}/${f.id}`, phase: 'Explain', schema: EXPLANATION }),
  (e, f) => e ? agent(simulatePrompt(f, e.explanation), { label: `simulate ${f.gate}/${f.id}`, phase: 'Simulate', schema: SIMULATION })
    .then(s => ({ item: f, explanation: e.explanation, sim: s })) : null)
const done = results.filter(Boolean)
const ok = done.filter(x => x.sim && x.sim.recorded)
log(`recorded ${ok.length}/${ITEMS.length}`)
return { recorded: ok.length, total: ITEMS.length }
