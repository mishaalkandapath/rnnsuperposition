export const meta = {
  name: 'rl-simulate-shared',
  description: 'Blind simulation of existing words-only explanations on one shared held-out window set (features and GRU units alike)',
  phases: [{ title: 'Simulate', detail: 'one agent per component: explanation only -> predicted 0-10 at every step of 100 shared windows' }],
}

// Claude Code Workflow script. Every component is simulated on the SAME 100 held-out windows
// (circuit.rl_task.simulation_scoring shared_sample), from its existing words-only explanation alone, so features
// and units are scored on identical data and the predictions can replace the activations in the model.
//   Workflow({scriptPath: "circuit/rl_task/autointerp_workflows/simulate_shared.js",
//             args: {repo, python, store, out, items: [[gate, id, explanation], ...]}})
// or items: [[gate, id], ...] with args.explanations = the directory of recorded explanations (simulation_scoring
// score records); each simulator then reads only its own explanation text through show_explanation.
const REPO = args.repo
const PY = args.python
const STORE = args.store
const OUT = args.out
const CLI = `PYTHONPATH=. ${PY} -m circuit.rl_task.simulation_scoring`

const RESULT = {
  type: 'object',
  properties: {
    correlation: { type: ['number', 'null'], description: 'as printed by score_shared' },
    recorded: { type: 'boolean' },
  },
  required: ['correlation', 'recorded'],
}

const TASK = `Two-step RL task (a GRU agent). Each trial: D1 (fixate; the previous trial's reward is shown), GO (choose a1 or a2), D2 (the state reached is shown: a1 -> S1 and a2 -> S2 with p = 0.8, "common", otherwise "rare"). The reached state pays a reward with p = 0.9 if it is the currently good state, 0.1 otherwise; which state is good is hidden and reverses a few times per 100-trial episode.`

// optional shared definitions every simulator sees (e.g. the episode clock some explanations refer to)
const CONTEXT = args.context ? `\n\n${args.context}` : ''

function prompt(f) {
  const desc = f.explanation ? `The description:
  "${f.explanation}"` : `Get the description first with
  ${CLI} show_explanation ${f.gate} ${f.id} --source ${args.explanations}
(it prints only the text).`
  return `${TASK}${CONTEXT}

You are predicting a hidden quantity from a description of it. ${desc}

Work in ${REPO} (cd there first). Use only these commands${f.explanation ? '' : ' (and show_explanation)'}; do not read files or run anything else:
  ${CLI} shared_sample ${STORE} ${f.gate} ${f.id} --chunk K
prints chunk K (K = 0 ... 9) of 100 held-out windows (10 per chunk), each a few trials with steps marked [s0], [s1], ...; the values are NOT shown.
  ${CLI} record_chunk ${STORE} ${f.gate} ${f.id} --chunk K '<JSON>' --out ${OUT}
records your predictions for chunk K. JSON = {"w<i>": [v_s0, v_s1, ...], ...} for the chunk's windows: one number from 0 to 10 for EVERY step, in order, using only the description.
  ${CLI} score_shared ${STORE} ${f.gate} ${f.id} --explanation '<the description above>' --out ${OUT}
after all 10 chunks are recorded.

For each chunk K = 0 ... 9 in turn: run shared_sample, predict every step from the description alone, run record_chunk. Then run score_shared once and return the correlation it prints (recorded = true). Keep the same judgement across chunks. Mind shell quoting: no single quotes inside the JSON or the description (replace them).`
}

const ITEMS = args.items.map(([gate, id, explanation]) => ({ gate, id, explanation }))
log(`${ITEMS.length} components`)
const results = await pipeline(ITEMS,
  f => agent(prompt(f), { label: `shared ${f.gate}/${f.id}`, phase: 'Simulate', schema: RESULT }))
const ok = results.filter(r => r && r.recorded).length
log(`recorded ${ok}/${ITEMS.length}`)
return { recorded: ok, total: ITEMS.length }
