"""Replacement-model eval: splice one transcoder into the copy-task GRU.

Only the gate being trained is replaced; the other gates stay as computed by
the real GRU. The spliced hidden state is carried forward, so transcoder
errors compound through the recurrence as they would in a replacement model.
Inputs are teacher-forced from the stored traces (the original model's own
autoregressive tokens), so the comparison is per position on identical
inputs.

Assumes the copy-trace layout written by datasets/transcoder_copy_datasets.py:
a single-layer GRU whose outputs are read at the last T // 2 steps.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import StackDataset

torch.serialization.add_safe_globals([StackDataset])

SPLICE_TARGETS = ("reset", "update", "candidate")


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
            x_t = x[:, t]
            gate_input = torch.cat([h, x_t], dim=-1)
            r = torch.sigmoid(gru.input_to_reset(x_t) + gru.hidden_to_reset(h))
            z = torch.sigmoid(gru.input_to_update(x_t) + gru.hidden_to_update(h))
            if transcoder is not None and self.target == "reset":
                r = transcoder(gate_input)[0]
            if transcoder is not None and self.target == "update":
                z = transcoder(gate_input)[0]
            n = torch.tanh(gru.input_to_new(x_t) + gru.hidden_to_new(r * h))
            if transcoder is not None and self.target == "candidate":
                n = transcoder(torch.cat([r * h, x_t], dim=-1))[0]
            h = (1 - z) * h + z * n
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
