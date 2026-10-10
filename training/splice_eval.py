"""Replacement-model eval: splice one transcoder into a task's GRU.

Only the gate being trained is replaced; the other gates stay as computed by
the real GRU. The spliced hidden state is carried forward, so transcoder
errors compound through the recurrence as they would in a replacement model.
Inputs are teacher-forced from the stored traces, so the comparison is per
position on identical inputs.

Two tasks:
  copy: traces from datasets/transcoder_copy_datasets.py; outputs read at
        the last T // 2 steps; KL, top-1 agreement and task accuracy.
  rl:   trace windows from datasets/transcoder_rl_datasets.py; every step
        compared; policy KL at GO and at DELAY steps, a1/a2 choice agreement
        at GO, and value-head difference.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import StackDataset

from models.rnn import RNN

torch.serialization.add_safe_globals([StackDataset])

SPLICE_TARGETS = ("reset", "update", "candidate")
SPLICE_TASKS = ("copy", "rl")
RL_N_ACTIONS = 3  # RL readout: 3 policy logits, then the value


def build_task_rnn(task: str, input_size: int, hidden_size: int) -> RNN:
    """The GRU architecture each task's checkpoints were trained with."""
    if task == "copy":
        # input = vocab + delimiter, output = vocab
        return RNN(input_size=input_size, hidden_size=hidden_size, out_size=input_size - 1,
                   out_act=lambda x: x, use_gru=True)
    if task == "rl":
        return RNN(input_size=input_size, hidden_size=hidden_size, out_size=RL_N_ACTIONS + 1,
                   out_act=lambda x: x, use_gru=True, learn_init=True)
    raise ValueError(f"task must be one of {SPLICE_TASKS}, got {task!r}")


def load_task_traces(task: str, paths: List[str]) -> List[Dict]:
    return load_splice_traces(paths) if task == "copy" else load_rl_splice_traces(paths)


def make_splice_evaluator(task: str, rnn_model, target: str, traces: List[Dict], samples: float,
                          val_sequence_ids: Optional[torch.Tensor] = None, seed: int = 0, device="cpu"):
    cls = SpliceEvaluator if task == "copy" else RLSpliceEvaluator
    return cls(rnn_model, target, traces, samples, val_sequence_ids=val_sequence_ids, seed=seed, device=device)


def gru_step(gru, h: torch.Tensor, x_t: torch.Tensor, transcoder=None, target: Optional[str] = None) -> torch.Tensor:
    """One GRU step with the target gate optionally replaced by a transcoder."""
    gate_input = torch.cat([h, x_t], dim=-1)
    r = torch.sigmoid(gru.input_to_reset(x_t) + gru.hidden_to_reset(h))
    z = torch.sigmoid(gru.input_to_update(x_t) + gru.hidden_to_update(h))
    if transcoder is not None and target == "reset":
        r = transcoder(gate_input)[0]
    if transcoder is not None and target == "update":
        z = transcoder(gate_input)[0]
    n = torch.tanh(gru.input_to_new(x_t) + gru.hidden_to_new(r * h))
    if transcoder is not None and target == "candidate":
        n = transcoder(torch.cat([r * h, x_t], dim=-1))[0]
    return (1 - z) * h + z * n


def load_splice_traces(paths: List[str]) -> List[Dict[str, Optional[torch.Tensor]]]:
    """Load trace files keeping only what the splice eval needs.

    Trace files also hold h/z/r/n for every timestep, which is far larger
    than the inputs; files are loaded one at a time and the rest dropped.
    """
    traces = []
    for path in paths:
        columns = torch.load(path, map_location="cpu").datasets
        traces.append({"inputs": columns["inputs"], "seq_id": columns.get("seq_id")})
        del columns
    return traces


class SpliceEvaluator:
    def __init__(self, rnn_model, target: str, traces: List[Dict],
                 samples: float, val_sequence_ids: Optional[torch.Tensor] = None,
                 seed: int = 0, device="cpu", chunk_size: int = 4096):
        """
        Args:
            rnn_model: Trained RNN (single-layer GRU with a linear readout).
            target: Which gate the transcoder replaces: reset, update or candidate.
            traces: Output of load_splice_traces (one entry per sequence length).
            samples: Sequences to evaluate. Values <= 1 are a fraction of the
                eligible sequences; larger values are an absolute count.
            val_sequence_ids: If given (and traces carry seq_id), only these
                held-out sequences are eligible.
        """
        if target not in SPLICE_TARGETS:
            raise ValueError(f"splice target must be one of {SPLICE_TARGETS}, got {target!r}")
        if rnn_model.num_layers != 1 or not rnn_model.use_gru or not rnn_model.out_size:
            raise ValueError("Splice eval expects a single-layer GRU with an output layer")
        self.rnn = rnn_model.to(device).eval()
        self.gru = self.rnn.layers[0]
        self.readout = self.rnn.layers[-1]
        self.target = target
        self.device = device
        self.chunk_size = chunk_size

        # Gather eligible sequences per file; files hold one length each.
        pools = []
        held_out_only = val_sequence_ids is not None and all(
            trace["seq_id"] is not None for trace in traces)
        for trace in traces:
            inputs = trace["inputs"]
            if held_out_only:
                inputs = inputs[torch.isin(trace["seq_id"], val_sequence_ids)]
            pools.append(inputs)
        if not held_out_only:
            print("-- WARNING: no held-out sequence ids for the splice eval "
                  "(old dataset or traces without seq_id); sampling from all "
                  "sequences, which may include training sequences.")

        n_eligible = sum(p.shape[0] for p in pools)
        if n_eligible == 0:
            raise ValueError("No eligible sequences for splice eval")
        n_samples = round(samples * n_eligible) if samples <= 1 else int(samples)
        n_samples = max(1, min(n_samples, n_eligible))

        # Sample uniformly over all eligible sequences, then regroup by file.
        generator = torch.Generator().manual_seed(seed)
        chosen = torch.randperm(n_eligible, generator=generator)[:n_samples]
        self.inputs = []
        offset = 0
        for pool in pools:
            in_pool = chosen[(chosen >= offset) & (chosen < offset + pool.shape[0])] - offset
            if len(in_pool):
                self.inputs.append(pool[in_pool].float())
            offset += pool.shape[0]
        self.n_sequences = n_samples
        print(f"-- Splice eval: {n_samples} of {n_eligible} eligible sequences, target={target}")

        # The unspliced reference only depends on the frozen RNN; compute once.
        self.reference_logits = [self._run(inputs, transcoder=None) for inputs in self.inputs]

    @torch.no_grad()
    def _run(self, inputs: torch.Tensor, transcoder=None) -> torch.Tensor:
        """Logits at the output steps, chunked over sequences. Kept on CPU."""
        outputs = []
        for start in range(0, inputs.shape[0], self.chunk_size):
            x = inputs[start:start + self.chunk_size].to(self.device)
            outputs.append(self._run_chunk(x, transcoder).cpu())
        return torch.cat(outputs)

    def _run_chunk(self, x: torch.Tensor, transcoder) -> torch.Tensor:
        gru = self.gru
        batch, steps, _ = x.shape
        if self.rnn.learn_init:
            h = self.rnn.initial_states[0].unsqueeze(0).expand(batch, -1)
        else:
            h = torch.zeros(batch, gru.hidden_size, device=x.device, dtype=x.dtype)
        logits = []
        for t in range(steps):
            h = gru_step(gru, h, x[:, t], transcoder, self.target)
            if t >= steps - steps // 2:
                logits.append(self.readout(h))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def evaluate(self, transcoder) -> Dict[str, float]:
        """Fidelity of the spliced model to the original on the sampled sequences."""
        was_training = transcoder.training
        transcoder.eval()
        kl_sum = agree = orig_correct = splice_correct = n_positions = 0.0
        for inputs, reference in zip(self.inputs, self.reference_logits):
            spliced = self._run(inputs, transcoder)
            # Copy targets: the first L input tokens (excluding the delimiter dim).
            n_out = reference.shape[1]
            true_tokens = inputs[:, :n_out, :-1].argmax(-1)
            log_p = F.log_softmax(reference, dim=-1)
            log_q = F.log_softmax(spliced, dim=-1)
            kl_sum += (log_p.exp() * (log_p - log_q)).sum(-1).sum().item()
            ref_tokens, splice_tokens = reference.argmax(-1), spliced.argmax(-1)
            agree += (ref_tokens == splice_tokens).sum().item()
            orig_correct += (ref_tokens == true_tokens).sum().item()
            splice_correct += (splice_tokens == true_tokens).sum().item()
            n_positions += ref_tokens.numel()
        transcoder.train(was_training)
        return {
            "splice_kl": kl_sum / n_positions,
            "splice_top1_agreement": agree / n_positions,
            "splice_task_acc": splice_correct / n_positions,
            "original_task_acc": orig_correct / n_positions,
        }


def load_rl_splice_traces(paths: List[str]) -> List[Dict[str, Optional[torch.Tensor]]]:
    """Load RL trace-window files keeping only what the splice eval needs:
    inputs and h_prevs (trials, phases, ·) and the episode seq_id."""
    traces = []
    for path in paths:
        columns = torch.load(path, map_location="cpu").datasets
        traces.append({"inputs": columns["inputs"], "h_prevs": columns["h_prevs"],
                       "seq_id": columns.get("seq_id")})
        del columns
    return traces


class RLSpliceEvaluator:
    PAD = 5  # padding value of whole padded trials in RL trace windows

    def __init__(self, rnn_model, target: str, traces: List[Dict], samples: float,
                 val_sequence_ids: Optional[torch.Tensor] = None, seed: int = 0,
                 device="cpu", chunk_size: int = 4096):
        """
        Args:
            rnn_model: Trained RL GRU (3 policy logits + value readout).
            target: Which gate the transcoder replaces: reset, update or candidate.
            traces: Output of load_rl_splice_traces (one entry per window file).
            samples: Windows to evaluate. Values <= 1 are a fraction of the
                eligible windows; larger values are an absolute count.
            val_sequence_ids: If given (and windows carry seq_id), only windows
                from these held-out episodes are eligible.

        Windows start mid-episode, so each is run from its stored hidden state
        at its first real step. Phase = step % 3 (DELAY_1, GO, DELAY_2).
        """
        if target not in SPLICE_TARGETS:
            raise ValueError(f"splice target must be one of {SPLICE_TARGETS}, got {target!r}")
        if rnn_model.num_layers != 1 or not rnn_model.use_gru or rnn_model.out_size != RL_N_ACTIONS + 1:
            raise ValueError("RL splice eval expects a single-layer GRU with a 3-logit + value readout")
        self.rnn = rnn_model.to(device).eval()
        self.gru = self.rnn.layers[0]
        self.readout = self.rnn.layers[-1]
        self.target = target
        self.device = device
        self.chunk_size = chunk_size

        held_out_only = val_sequence_ids is not None and all(t["seq_id"] is not None for t in traces)
        inputs, h_prevs = [], []
        for trace in traces:
            keep = (torch.isin(trace["seq_id"], val_sequence_ids) if held_out_only
                    else torch.ones(len(trace["inputs"]), dtype=torch.bool))
            inputs.append(trace["inputs"][keep])
            h_prevs.append(trace["h_prevs"][keep])
        if not held_out_only:
            print("-- WARNING: no held-out episode ids for the RL splice eval (old trace files or "
                  "row-split dataset); sampling from all windows, which may include training episodes.")
        inputs, h_prevs = torch.cat(inputs), torch.cat(h_prevs)
        n_eligible = len(inputs)
        if n_eligible == 0:
            raise ValueError("No eligible windows for the RL splice eval")
        n_samples = round(samples * n_eligible) if samples <= 1 else int(samples)
        n_samples = max(1, min(n_samples, n_eligible))
        generator = torch.Generator().manual_seed(seed)
        chosen = torch.randperm(n_eligible, generator=generator)[:n_samples]
        inputs, h_prevs = inputs[chosen].float(), h_prevs[chosen].float()

        # Group by number of padded trials; flatten (trials, phases) to steps.
        pad = (inputs == self.PAD).flatten(2).all(-1).sum(-1)               # (window,)
        self.groups = []
        for n_pad in pad.unique().tolist():
            sel = pad == n_pad
            x = inputs[sel][:, n_pad:].flatten(1, 2)                         # (n, steps, X)
            h0 = h_prevs[sel][:, n_pad, 0]                                   # state entering the first real step
            self.groups.append((x, h0))
        self.n_windows = n_samples
        print(f"-- RL splice eval: {n_samples} of {n_eligible} eligible windows, target={target}")

        self.reference = [self._run(x, h0, None) for x, h0 in self.groups]

    @torch.no_grad()
    def _run(self, x: torch.Tensor, h0: torch.Tensor, transcoder) -> torch.Tensor:
        """Readout (policy logits + value) at every step, chunked. Kept on CPU."""
        outputs = []
        for start in range(0, x.shape[0], self.chunk_size):
            xs = x[start:start + self.chunk_size].to(self.device)
            h = h0[start:start + self.chunk_size].to(self.device)
            steps = []
            for t in range(xs.shape[1]):
                h = gru_step(self.gru, h, xs[:, t], transcoder, self.target)
                steps.append(self.readout(h))
            outputs.append(torch.stack(steps, dim=1).cpu())
        return torch.cat(outputs)

    @torch.no_grad()
    def evaluate(self, transcoder) -> Dict[str, float]:
        """Fidelity of the spliced agent to the original on the sampled windows.

        splice_kl_go / splice_kl_delay: KL(original || spliced) of the full
        3-way policy, averaged over GO steps / DELAY_1 and DELAY_2 steps.
        splice_go_agreement: same a1/a2 choice (argmax) at GO steps.
        splice_value_mae: mean |value_original - value_spliced| over all
        steps; splice_value_nmae divides it by the std of the original values.
        splice_nonfinite_frac: fraction of steps where the spliced readout is
        not finite (a gate transcoder whose outputs leave [0, 1] can make the
        hidden state diverge); the other metrics average the finite steps,
        except agreement, which counts a non-finite GO step as disagreeing.
        """
        was_training = transcoder.training
        transcoder.eval()
        kl = {"go": [0.0, 0], "delay": [0.0, 0]}
        agree = n_go = 0
        value_abs, n_value = 0.0, 0
        ref_values, n_steps, n_nonfinite = [], 0, 0
        for (x, h0), reference in zip(self.groups, self.reference):
            spliced = self._run(x, h0, transcoder)
            finite = torch.isfinite(spliced).all(-1)                         # (n, steps)
            log_p = F.log_softmax(reference[..., :RL_N_ACTIONS], dim=-1)
            log_q = F.log_softmax(spliced[..., :RL_N_ACTIONS], dim=-1)
            step_kl = (log_p.exp() * (log_p - log_q)).sum(-1)               # (n, steps)
            phase = torch.arange(step_kl.shape[1]) % 3
            go = (phase == 1)[None, :]
            for name, mask in (("go", go & finite), ("delay", ~go & finite)):
                kl[name][0] += step_kl[mask].sum().item(); kl[name][1] += mask.sum().item()
            choice = lambda out: out[..., 1:RL_N_ACTIONS].argmax(-1)
            agree += ((choice(reference) == choice(spliced)) & go & finite).sum().item()
            n_go += go.sum().item() * reference.shape[0]
            value_abs += (reference[..., -1] - spliced[..., -1]).abs()[finite].sum().item()
            n_value += finite.sum().item()
            ref_values.append(reference[..., -1].flatten())
            n_steps += finite.numel()
            n_nonfinite += (~finite).sum().item()
        transcoder.train(was_training)
        value_mae = value_abs / max(n_value, 1)
        return {
            "splice_kl_go": kl["go"][0] / max(kl["go"][1], 1),
            "splice_kl_delay": kl["delay"][0] / max(kl["delay"][1], 1),
            "splice_go_agreement": agree / max(n_go, 1),
            "splice_value_mae": value_mae,
            "splice_value_nmae": value_mae / max(torch.cat(ref_values).std().item(), 1e-12),
            "splice_nonfinite_frac": n_nonfinite / max(n_steps, 1),
        }
