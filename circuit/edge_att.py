from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class CircuitNode:
    """Represents a node in the circuit graph.
    name:
        Use patterns like: "x_{t}_{i}", "f_n_{t}_{j}", "e_n_{t}", "h_init",
        "o_{t}_{k}". Gate features ("f_z_{t}_{j}", "f_r_{t}_{j}") are not
        main-graph nodes; they name gate-view pieces and are the targets of
        rooted gate-feature graphs.
    node_type:
        'input' | 'feature' | 'initial' | 'error' | 'output'
    timestep:
        Integer time index.
    feature_idx:
        For feature nodes (index in the feature bank) or output nodes (index in output dim).
    input_dim:
        For input nodes (input dimension index).
    """
    name: str
    node_type: str
    timestep: int
    feature_idx: Optional[int] = None
    input_dim: Optional[int] = None
    hidden_dim: Optional[int] = None

class CircuitTracer:
    """Compute edge attribution weights for RNN transcoder circuits.

    Edges are prompt-specific linear attributions: the active scalar at the
    source times its local derivative on the target, with the update and
    reset gates frozen at their actual values (as attention patterns are
    frozen in Ameisen et al. 2025). Hidden coordinates are folded into the
    edges, as the paper folds the residual stream.

    Gate features are explained separately (as in Kamath et al. 2025's QK
    attributions): the transcoders output gate values directly, so each
    gate value is an exact sum over gate features plus bias and error. A
    folded edge passes through one gate per step it spans; it splits exactly
    by the gate features of any one step (split_edge_by_gate). Why a gate
    feature fired is traced by a graph rooted at it (gate_feature_graph).
    """

    def __init__(
        self,
        rnn_model: nn.Module,
        update_transcoder: nn.Module,
        hidden_transcoder: nn.Module,
        device: str = "cuda",
        reset_transcoder: Optional[nn.Module] = None,
        output_dims: Optional[int] = None,
        outputs_at: str = "second_half",
    ):
        """
        Args:
            output_dims: Treat only the first output_dims readout units as
                logits (RL: 3 policy logits, excluding the value unit).
                None uses all readout units.
            outputs_at: Timesteps with output nodes. "second_half" for the
                copy task (outputs at t >= T // 2), "all" for RL.
        """
        if outputs_at not in ("second_half", "all"):
            raise ValueError(f"outputs_at must be 'second_half' or 'all', got {outputs_at!r}")
        self.outputs_at = outputs_at
        self.rnn_model = rnn_model.to(device)
        self.update_transcoder = update_transcoder.to(device)
        self.hidden_transcoder = hidden_transcoder.to(device)
        self.reset_transcoder = reset_transcoder.to(device) if reset_transcoder else None
        self.device = device

        self.rnn_model.eval()
        self.update_transcoder.eval()
        self.hidden_transcoder.eval()
        if self.reset_transcoder:
            self.reset_transcoder.eval()

        # Update gate encoder splits: [h_{t-1}, x_t] -> pf^z_t -> ReLU -> f^z_t
        Wz_enc = self.update_transcoder.input_to_features.weight  # (Fz, H+X)
        Hz = rnn_model.hidden_size
        self.W_z_h = Wz_enc[:, :Hz]   # (Fz, H)
        self.W_z_x = Wz_enc[:, Hz:]   # (Fz, X)
        self.M_z = self.update_transcoder.features_to_outputs.weight  # (H, Fz), f^z -> z_hat

        # Hidden (new content) encoder: [r_t * h_{t-1}, x_t] -> pf^n_t -> ReLU -> f^n_t
        Wn_enc = self.hidden_transcoder.input_to_features.weight  # (Fn, H+X)
        self.W_n_h = Wn_enc[:, :Hz]   # (Fn, H)
        self.W_n_x = Wn_enc[:, Hz:]   # (Fn, X)
        self.M_n = self.hidden_transcoder.features_to_outputs.weight  # (H, Fn), f^n -> n_hat

        # Reset gate encoder: [h_{t-1}, x_t] -> f^r_t -> r_hat.  This is
        # optional so existing two-transcoder analyses remain loadable.
        if self.reset_transcoder:
            Wr_enc = self.reset_transcoder.input_to_features.weight
            self.W_r_h = Wr_enc[:, :Hz]
            self.W_r_x = Wr_enc[:, Hz:]
            self.M_r = self.reset_transcoder.features_to_outputs.weight

        # Output projection: o_t = W_o h_t (+ b)  [assumed linear]
        if hasattr(rnn_model, "layers") and len(rnn_model.layers) > rnn_model.num_layers:
            self.W_o = rnn_model.layers[-1].weight.to(device)  # (O, H)
            self.b_o = rnn_model.layers[-1].bias.to(device)
        else:
            print("No output weight?")
            self.W_o = torch.eye(Hz, device=device)  # (H, H)
            self.b_o = torch.zeros(Hz, device=device)
        if output_dims is not None:
            self.W_o, self.b_o = self.W_o[:output_dims], self.b_o[:output_dims]

    def _output_steps(self, T: int) -> List[int]:
        """Timesteps that have output nodes; row r of acts["logits"] is step r of this list."""
        return list(range(T // 2, T)) if self.outputs_at == "second_half" else list(range(T))

    @staticmethod
    def _flatten_rl_layout(acts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """RL items are (trials, phases, ·), left-padded with whole trials of 5s.

        Strip the padded trials and flatten to (trials * phases, ·), so
        h_prevs[0] is the state entering the first real trial.
        """
        lead = acts["inputs"].shape[:2]
        n_pad = 0
        while n_pad < lead[0] and torch.all(acts["inputs"][n_pad] == 5):
            n_pad += 1
        return {k: (v[n_pad:].reshape(-1, *v.shape[2:]) if v.shape[:2] == lead else v)
                for k, v in acts.items()}

    @torch.no_grad()
    def run_forward_pass(self, sequence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Copy tensors to device and cache per-timestep values we need.

        Expected keys in `sequence`:
          - inputs: (T, X)
          - h_prevs: (T, H)
          - h_new_ts: (T, H)      # n_hat (model's new content)
          - z_ts: (T, H)
          - r_ts: (T, H)          # reset gate (used upstream to make gated_hidden)
          - outputs: (T, O)
        RL items with a (trials, phases, ·) layout are flattened to (T, ·).
        """
        acts = {k: v.detach().to(self.device).clone() for k, v in sequence.items()
                if torch.is_tensor(v)}
        if acts["inputs"].dim() == 3:
            acts = self._flatten_rl_layout(acts)

        T = acts["inputs"].shape[0]

        z = acts["z_ts"]
        # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t  (using symbols from the user's code)
        acts["h_ts"] = (1.0 - z) * acts["h_prevs"] + z * acts["h_new_ts"]  # (T, H)

        # New copy trace datasets retain the actual output logits.  Older
        # datasets (and RL traces) do not; recover the logits from the
        # hidden states and the frozen RNN output projection instead of
        # softmaxing saved one-hot feedback tokens.
        if "logits" in acts:
            acts["logits"] = acts["logits"][..., :self.W_o.shape[0]]
        else:
            acts["logits"] = torch.nn.functional.linear(
                acts["h_ts"][self._output_steps(T)], self.W_o, self.b_o)

        # Pre/post feature activations are assumed to be produced by calling the transcoders.
        # We compute per-timestep masks needed for local linear maps.
        acts["pf_z"], acts["f_z"], acts["z_hat"], acts["e_z"] = [], [], [], []
        acts["pf_n"], acts["f_n"], acts["n_hat"], acts["e_n"] = [], [], [], []
        if self.reset_transcoder:
            acts["pf_r"], acts["f_r"], acts["r_hat"], acts["e_r"] = [], [], [], []
        else:
            # A two-transcoder graph still uses the real reset gate inside the
            # candidate input. Treat it as entirely unexplained residual so
            # that its effect is visible rather than silently frozen away.
            acts["e_r"] = acts["r_ts"].clone()

        for t in range(T):
            h_prev_t = acts["h_prevs"][t]            # (H,)
            x_t = acts["inputs"][t]                 # (X,)
            r_t = acts["r_ts"][t]                   # (H,)

            if self.reset_transcoder:
                r_in = torch.cat([h_prev_t, x_t], dim=0)
                r_hat_t, f_r_t, pf_r_t = self.reset_transcoder(r_in)
                acts["pf_r"].append(pf_r_t)
                acts["f_r"].append(f_r_t)
                acts["r_hat"].append(r_hat_t)
                acts["e_r"].append(r_t - r_hat_t)

            # Update gate transcoder input: concat[h_prev_t, x_t]
            z_in = torch.cat([h_prev_t, x_t], dim=0)
            z_hat_t, f_z_t, pf_z_t = self.update_transcoder(z_in)
            e_z_t = z[t] - z_hat_t  # model gate minus transcoder pred

            acts["pf_z"].append(pf_z_t)
            acts["f_z"].append(f_z_t)
            acts["z_hat"].append(z_hat_t)
            acts["e_z"].append(e_z_t)

            # Hidden/new-content transcoder input: concat[r_t * h_prev_t, x_t]
            gated_hidden = r_t * h_prev_t
            n_in = torch.cat([gated_hidden, x_t], dim=0)
            n_hat_t, f_n_t, pf_n_t = self.hidden_transcoder(n_in)
            e_n_t = acts["h_new_ts"][t] - n_hat_t

            acts["pf_n"].append(pf_n_t)
            acts["f_n"].append(f_n_t)
            acts["n_hat"].append(n_hat_t)
            acts["e_n"].append(e_n_t)

        # Stack lists to (T, ·)
        keys = ["pf_z", "f_z", "z_hat", "e_z", "pf_n", "f_n", "n_hat", "e_n"]
        if self.reset_transcoder:
            keys += ["pf_r", "f_r", "r_hat", "e_r"]
        for key in keys:
            acts[key] = torch.stack(acts[key], dim=0)

        return acts

    @staticmethod
    def _feature_mask(features: torch.Tensor) -> torch.Tensor:
        """Derivative mask of the actual JumpReLU output.

        A positive preactivation is not sufficient: JumpReLU activates only
        above its learned threshold.  The cached feature output is exactly
        zero when inactive, so this also works if the activation module is
        changed later.
        """
        return (features > 0).to(features.dtype)

    @staticmethod
    def _active_features_from_acts(acts: Dict[str, torch.Tensor]) -> Dict[str, List[Tuple[int, int, float]]]:
        """Return the features active in this exact traced forward pass.

        Circuit construction must not depend on a separately cached feature
        analysis, which may have been made with another dictionary checkpoint.
        """
        result = {}
        for kind, key in (("reset", "f_r"), ("update", "f_z"), ("hidden", "f_n")):
            if key not in acts:
                continue
            active = torch.nonzero(acts[key] > 0, as_tuple=False)
            result[kind] = [
                (int(t), int(j), float(acts[key][t, j])) for t, j in active.tolist()
            ]
        return result

    def get_active_features(self, sequence: Dict[str, torch.Tensor],
                            acts: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, List[Tuple[int, int, float]]]:
        """Public helper for visualizers: live, checkpoint-consistent features.

        Pass acts from run_forward_pass to avoid recomputing it.
        """
        return self._active_features_from_acts(acts if acts is not None else self.run_forward_pass(sequence))

    # ------------------------------------------------------------------
    # Folded edges
    # ------------------------------------------------------------------
    # With gates frozen the GRU is linear in what gets written into h:
    #   h_tau = sum_{s <= tau} C(s, tau) * w_s  +  C(-1, tau) * h_{-1}
    #   C(s, tau) = prod_{u = s+1}^{tau} (1 - z_u)        (C(s, s) = 1)
    # where w_s = z_s * (M_n f_s + b_n + e_n,s) is the write at step s.
    # Hidden coordinates are folded into the edges, as the paper folds the
    # residual stream: an edge from a source written at step s to a node that
    # reads h_tau is  sum_i read[i] * C(s, tau)[i] * write[i], so routes
    # through different coordinates cancel inside the edge. Every factor
    # (write gate z_s, carries 1 - z_u, read gate r_t) is a frozen constant.

    def _write_terms(self, node: CircuitNode, acts: Dict[str, torch.Tensor]
                     ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        """(step s written, write vector without its gate, write gate z_s or None)."""
        if node.node_type == "feature" and node.name.startswith("f_n_"):
            s, k = node.timestep, node.feature_idx
            return s, acts["f_n"][s, k] * self.M_n[:, k], acts["z_ts"][s]
        if node.node_type == "error":
            s = node.timestep
            return s, acts["e_n"][s], acts["z_ts"][s]
        if node.node_type == "initial":
            # State entering the window: learned initial state (RL) or zero (copy).
            return -1, acts["h_prevs"][0], None
        raise ValueError(f"{node.name} does not write into the hidden state")

    def _read_terms(self, node: CircuitNode, acts: Dict[str, torch.Tensor]
                    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        """(step tau of the h read, read vector without its gate, read gate r_t or None).

        Candidate features read h_{t-1} through the reset gate; gate features
        (only targets of rooted graphs) read h_{t-1} directly; outputs read h_t.
        """
        t = node.timestep
        if node.node_type == "output":
            W_c = self.W_o - self.W_o.mean(dim=0, keepdim=True)  # mean-centred logits
            return t, W_c[node.feature_idx], None
        j = node.feature_idx
        if node.name.startswith("f_n_"):
            return t - 1, self._feature_mask(acts["f_n"][t])[j] * self.W_n_h[j], acts["r_ts"][t]
        if node.name.startswith("f_z_"):
            return t - 1, self._feature_mask(acts["f_z"][t])[j] * self.W_z_h[j], None
        if node.name.startswith("f_r_") and self.reset_transcoder:
            return t - 1, self._feature_mask(acts["f_r"][t])[j] * self.W_r_h[j], None
        raise ValueError(f"{node.name} does not read the hidden state")

    def _input_row(self, node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encoder row (X,) mapping x_t into a feature read at t, times its JumpReLU mask."""
        t, j = node.timestep, node.feature_idx
        for prefix, W_x, key in (("f_n_", self.W_n_x, "f_n"), ("f_z_", self.W_z_x, "f_z"),
                                 ("f_r_", getattr(self, "W_r_x", None), "f_r")):
            if node.name.startswith(prefix) and W_x is not None:
                return self._feature_mask(acts[key][t])[j] * W_x[j]
        raise ValueError(f"{node.name} does not read the input")

    @staticmethod
    def _reads_hidden(node: CircuitNode) -> bool:
        return node.node_type in ("feature", "output")

    @staticmethod
    def _writes_hidden(node: CircuitNode) -> bool:
        return (node.node_type in ("error", "initial")
                or (node.node_type == "feature" and node.name.startswith("f_n_")))

    def compute_edge_weight(self, from_node: CircuitNode, to_node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return scalar edge weight from `from_node` to `to_node`.

        This is the per-edge reference definition; build_circuit_graph uses
        the equivalent batched computation in _compute_edges.

        Gates are frozen at their actual values, as the paper freezes
        attention patterns, so the replacement model is linear and edges into
        a node sum exactly to its value minus bias terms. Two kinds of edge:
          x_t[d] -> feature at t:   x_t[d] * W_x[j, d]  (times JumpReLU mask)
          writer at s -> reader of h_tau (s <= tau):
              sum_i read[i] * r_t[i] * prod_{u=s+1}^{tau} (1 - z_u[i]) * z_s[i] * write[i]
        Gate features get no edges; see gate_pieces / split_edge_by_gate.
        """
        zero = torch.tensor(0.0, device=self.device)
        if not self._reads_hidden(to_node):
            return zero
        if from_node.node_type == "input":
            if to_node.node_type != "feature" or from_node.timestep != to_node.timestep:
                return zero
            d = from_node.input_dim
            return acts["inputs"][from_node.timestep, d] * self._input_row(to_node, acts)[d]
        if not self._writes_hidden(from_node):
            return zero
        s, write, write_gate = self._write_terms(from_node, acts)
        tau, read, read_gate = self._read_terms(to_node, acts)
        if s > tau:
            return zero
        contribution = read * write
        if read_gate is not None:
            contribution = contribution * read_gate
        if write_gate is not None:
            contribution = contribution * write_gate
        for u in range(s + 1, tau + 1):
            contribution = contribution * (1.0 - acts["z_ts"][u])
        return contribution.sum()

    def _build_nodes(self, acts: Dict[str, torch.Tensor]) -> List[CircuitNode]:
        T = acts["inputs"].shape[0]
        active_features = self._active_features_from_acts(acts)
        nodes: List[CircuitNode] = []
        # Inputs: one node per nonzero input dimension (copy inputs are
        # one-hot; RL inputs mix one-hots with a reward scalar).
        for t in range(T):
            for d in torch.nonzero(acts["inputs"][t], as_tuple=False).flatten().tolist():
                nodes.append(CircuitNode(f"x_{t}_{d}", "input", t, input_dim=d))

        # Active candidate features. Gate (update/reset) features are not
        # nodes: with gates frozen they have no edges in this graph, and are
        # reported through gate_pieces instead.
        for t, j, mag in active_features["hidden"]:
            if mag < 1e-5:
                continue
            nodes.append(CircuitNode(f"f_n_{t}_{j}", "feature", t, feature_idx=j))

        # Hidden coordinates are folded into the edges (no h nodes). The state
        # entering the window is one vector-valued source node, like an error
        # node: zero for the copy task, the learned initial state (or the
        # carried state) for RL.
        nodes.append(CircuitNode("h_init", "initial", -1))

        # Frozen candidate reconstruction residuals are source-only nodes,
        # analogous to error nodes in a local replacement model. Gate errors
        # are frozen with the gates and show up in gate_pieces.
        for t in range(T):
            nodes.append(CircuitNode(f"e_n_{t}", "error", t))

        # Outputs
        sorted_outs = torch.argsort(acts["logits"], dim=-1, descending=True)
        for row, t in enumerate(self._output_steps(T)):
            for k in sorted_outs[row].tolist():
                nodes.append(CircuitNode(f"o_{t}_{k}", "output", t, feature_idx=k))
        return nodes

    def _compute_edges_reference(self, nodes: List[CircuitNode], acts: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str], float]:
        """O(N^2) loop over compute_edge_weight. Slow; kept to verify _compute_edges."""
        edge_weights: Dict[Tuple[str, str], float] = {}
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                w = self.compute_edge_weight(src, dst, acts)
                if not torch.isfinite(w) or abs(float(w)) < 1e-6:
                    continue
                edge_weights[(src.name, dst.name)] = float(w)
        return edge_weights

    @torch.no_grad()
    def _compute_edges(self, nodes: List[CircuitNode], acts: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str], float]:
        """Batched equivalent of _compute_edges_reference.

        Writers are grouped by the step s they write, readers by the step tau
        of h they read; each (s <= tau) pair is one matrix product through
        the carry C(s, tau). Edges are returned in the same order as the
        reference (source node order, then destination node order). Works on
        any node list, including rooted gate-feature graphs.
        """
        T, H = acts["h_ts"].shape
        dev = acts["h_ts"].device
        z = acts["z_ts"]

        # carry[s + 1, tau] = C(s, tau) = prod_{u=s+1}^{tau} (1 - z_u), for tau >= s
        carry = torch.zeros(T + 1, T, H, device=dev, dtype=z.dtype)
        for s in range(-1, T):
            running = torch.ones(H, device=dev, dtype=z.dtype)
            for tau in range(max(s, 0), T):
                if tau > s:
                    running = running * (1.0 - z[tau])
                carry[s + 1, tau] = running

        writers: Dict[int, Tuple[List[int], List[torch.Tensor]]] = {}
        readers: Dict[int, Tuple[List[int], List[torch.Tensor]]] = {}
        input_readers: Dict[int, Tuple[List[int], List[torch.Tensor]]] = {}
        inputs_at: Dict[int, List[int]] = {}
        for n, node in enumerate(nodes):
            if self._writes_hidden(node):
                s, write, gate = self._write_terms(node, acts)
                group = writers.setdefault(s, ([], []))
                group[0].append(n); group[1].append(write if gate is None else gate * write)
            if self._reads_hidden(node):
                tau, read, gate = self._read_terms(node, acts)
                group = readers.setdefault(tau, ([], []))
                group[0].append(n); group[1].append(read if gate is None else gate * read)
            if node.node_type == "feature":
                group = input_readers.setdefault(node.timestep, ([], []))
                group[0].append(n); group[1].append(self._input_row(node, acts))
            if node.node_type == "input":
                inputs_at.setdefault(node.timestep, []).append(n)

        srcs, dsts, ws = [], [], []

        def add(src_ids, dst_ids, w):
            # w is (n_dst, n_src)
            src_ids = torch.tensor(src_ids, dtype=torch.long, device=dev)
            dst_ids = torch.tensor(dst_ids, dtype=torch.long, device=dev)
            srcs.append(src_ids[None, :].expand_as(w).reshape(-1))
            dsts.append(dst_ids[:, None].expand_as(w).reshape(-1))
            ws.append(w.reshape(-1))

        # writer at s -> reader of h_tau, through the frozen carry
        for tau, (dst_ids, rows) in readers.items():
            R = torch.stack(rows)                                              # (n_dst, H)
            for s, (src_ids, cols) in writers.items():
                if s <= tau:
                    W = torch.stack(cols, dim=1)                               # (H, n_src)
                    # tau = -1 (readers at t = 0) only sees h_init, with C(-1, -1) = 1
                    C = carry[s + 1, tau] if tau >= 0 else torch.ones(H, device=dev, dtype=z.dtype)
                    add(src_ids, dst_ids, R @ (C[:, None] * W))

        # input x_t[d] -> feature read at t
        for t, (dst_ids, rows) in input_readers.items():
            if t not in inputs_at:
                continue
            d = torch.tensor([nodes[n].input_dim for n in inputs_at[t]], dtype=torch.long, device=dev)
            add(inputs_at[t], dst_ids, torch.stack(rows)[:, d] * acts["inputs"][t, d][None, :])

        if not ws:
            return {}
        src, dst, w = torch.cat(srcs), torch.cat(dsts), torch.cat(ws)
        keep = torch.isfinite(w) & (w.abs() >= 1e-6)
        src, dst, w = src[keep], dst[keep], w[keep]
        order = torch.argsort(src * len(nodes) + dst)
        names = [node.name for node in nodes]
        return {(names[s], names[d]): v
                for s, d, v in zip(src[order].tolist(), dst[order].tolist(), w[order].tolist())}

    @staticmethod
    def _normalize_by_group(nodes: List[CircuitNode], edge_weights: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        """Scale edges by the max |weight| within each (source group, target group).

        A group is node type + gate kind (features and error nodes) + timestep.
        Display aid only; pruning uses the raw weights.
        """
        group = {}
        for node in nodes:
            kind = node.name[2] if node.node_type in ("feature", "error") else ""
            group[node.name] = (node.node_type, kind, node.timestep)
        max_abs: Dict[Tuple, float] = {}
        for (src, dst), w in edge_weights.items():
            key = (group[src], group[dst])
            max_abs[key] = max(max_abs.get(key, 0.0), abs(w))
        return {(src, dst): w / max_abs[(group[src], group[dst])]
                for (src, dst), w in edge_weights.items()
                if max_abs[(group[src], group[dst])] >= 1e-6}

    def build_circuit_graph(
        self,
        sequence: Dict[str, torch.Tensor],
        active_features: Dict[str, List[Tuple[int, int, float]]],
        acts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        """Build edge maps {(from_name, to_name): weight} for relevant nodes.

        Returns raw and group-normalized edge weights. Pass acts from
        run_forward_pass(sequence) to avoid recomputing it.

        ``active_features`` is retained for call compatibility but deliberately
        ignored: features are recomputed from the loaded models on this exact
        sequence, preventing stale feature-cache/model mismatches.
        """
        if acts is None:
            acts = self.run_forward_pass(sequence)
        nodes = self._build_nodes(acts)
        edge_weights = self._compute_edges(nodes, acts)
        return edge_weights, self._normalize_by_group(nodes, edge_weights)

    # ------------------------------------------------------------------
    # Gate view: which gate features set each frozen gate
    # ------------------------------------------------------------------

    @staticmethod
    def _decoder_bias(transcoder: nn.Module) -> Optional[torch.Tensor]:
        return transcoder.features_to_outputs.bias

    @torch.no_grad()
    def _gate_piece_matrix(self, acts: Dict[str, torch.Tensor], gate: str, t: int
                           ) -> Tuple[List[str], torch.Tensor]:
        """Names and (H, n_pieces) matrix whose columns sum to the gate vector at t.

        gate is "update" (z_t) or "reset" (r_t). The transcoder outputs the
        gate value directly, so
            gate = sum_j f[t, j] * M[:, j] + bias + error[t]
        Columns: one per active gate feature, then bias, then error.
        """
        if gate == "update":
            transcoder, key, M, err, prefix = self.update_transcoder, "f_z", self.M_z, "e_z", "f_z"
        elif gate == "reset":
            if not self.reset_transcoder:
                return [f"r_{t} (no reset transcoder)"], acts["r_ts"][t][:, None]
            transcoder, key, M, err, prefix = self.reset_transcoder, "f_r", self.M_r, "e_r", "f_r"
        else:
            raise ValueError(f"gate must be 'update' or 'reset', got {gate!r}")
        active = torch.nonzero(acts[key][t] > 0, as_tuple=False).flatten()
        names = [f"{prefix}_{t}_{j}" for j in active.tolist()]
        cols = [M[:, active] * acts[key][t, active][None, :]]
        bias = self._decoder_bias(transcoder)
        if bias is not None:
            names.append(f"{gate}_bias")
            cols.append(bias[:, None])
        names.append(f"{err}_{t}")
        cols.append(acts[err][t][:, None])
        return names, torch.cat(cols, dim=1)

    @staticmethod
    def _sorted_pieces(names: List[str], values: torch.Tensor) -> List[Tuple[str, float]]:
        return sorted(zip(names, values.tolist()), key=lambda piece: -abs(piece[1]))

    @torch.no_grad()
    def gate_pieces(self, acts: Dict[str, torch.Tensor], gate: str, t: int, i: int) -> List[Tuple[str, float]]:
        """Exact decomposition of one gate value z_t[i] / r_t[i] into gate
        features, bias and error. Sums to the gate the GRU used; sorted by |value|."""
        names, P = self._gate_piece_matrix(acts, gate, t)
        return self._sorted_pieces(names, P[i])

    @staticmethod
    def node_from_name(name: str) -> CircuitNode:
        """Inverse of the node naming scheme (x_t_d, f_*_t_j, e_n_t, h_init, o_t_k)."""
        parts = name.split("_")
        if name == "h_init":
            return CircuitNode(name, "initial", -1)
        if parts[0] == "x":
            return CircuitNode(name, "input", int(parts[1]), input_dim=int(parts[2]))
        if parts[0] == "f":
            return CircuitNode(name, "feature", int(parts[2]), feature_idx=int(parts[3]))
        if parts[0] == "e":
            return CircuitNode(name, "error", int(parts[2]))
        if parts[0] == "o":
            return CircuitNode(name, "output", int(parts[1]), feature_idx=int(parts[2]))
        raise ValueError(f"Unrecognised node name {name!r}")

    def edge_gate_steps(self, src: str, dst: str) -> List[Tuple[int, str, str]]:
        """Gates a folded edge passes through: (step, "update"/"reset", role).

        role is "write" (z_s on the write), "carry" (1 - z_u while carried),
        or "read" (r_t on a candidate feature's read). One gate per step.
        Input edges are ungated and return [].
        """
        src_node, dst_node = self.node_from_name(src), self.node_from_name(dst)
        if not (self._writes_hidden(src_node) and self._reads_hidden(dst_node)):
            return []
        s = -1 if src_node.node_type == "initial" else src_node.timestep
        t = dst_node.timestep
        tau = t if dst_node.node_type == "output" else t - 1
        steps = []
        if src_node.node_type != "initial":
            steps.append((s, "update", "write"))
        steps += [(u, "update", "carry") for u in range(s + 1, tau + 1)]
        if dst_node.name.startswith("f_n_"):
            steps.append((t, "reset", "read"))
        return steps

    @torch.no_grad()
    def split_edge_by_gate(self, src: str, dst: str, acts: Dict[str, torch.Tensor],
                           step: int) -> List[Tuple[str, float]]:
        """Split a folded edge by the gate features at one step it passes through.

        Per coordinate the edge is a product of one gate per step; holding
        the other steps' gates fixed, it is linear in this step's gate, so
        the split is exact: shares sum to the edge weight. For a carry step
        the factor is (1 - z_u); "keep_u" is the share of the 1. Returns []
        if the edge has no gate at that step.
        """
        gates = {u: (gate, role) for u, gate, role in self.edge_gate_steps(src, dst)}
        if step not in gates:
            return []
        gate, role = gates[step]
        src_node, dst_node = self.node_from_name(src), self.node_from_name(dst)
        s, write, write_gate = self._write_terms(src_node, acts)
        tau, read, read_gate = self._read_terms(dst_node, acts)
        # Per-coordinate edge contribution with this step's gate factor left out.
        rest = read * write
        if read_gate is not None and role != "read":
            rest = rest * read_gate
        if write_gate is not None and role != "write":
            rest = rest * write_gate
        for u in range(s + 1, tau + 1):
            if not (role == "carry" and u == step):
                rest = rest * (1.0 - acts["z_ts"][u])
        names, P = self._gate_piece_matrix(acts, gate, step)
        shares = rest @ P
        if role == "carry":
            return self._sorted_pieces([f"keep_{step}"] + names,
                                       torch.cat([rest.sum()[None], -shares]))
        return self._sorted_pieces(names, shares)

    @torch.no_grad()
    def read_gate_pieces(self, acts: Dict[str, torch.Tensor], name: str) -> List[Tuple[str, float]]:
        """For candidate feature f_n_t_k: split its whole memory input
        mask * W_n_h[k] . (r_t * h_{t-1}) by the reset-gate pieces at t."""
        node = self.node_from_name(name)
        tau, read, read_gate = self._read_terms(node, acts)
        h_prev = acts["h_prevs"][node.timestep]
        names, P = self._gate_piece_matrix(acts, "reset", node.timestep)
        return self._sorted_pieces(names, (read * h_prev) @ P)

    @torch.no_grad()
    def gate_feature_graph(self, acts: Dict[str, torch.Tensor], name: str
                           ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[str, float]]:
        """Graph rooted at an active gate feature f_z_t_j / f_r_t_j: why did it fire?

        The gate feature's preactivation is linear in [h_{t-1}, x_t], and
        everything upstream is the frozen main graph, so this is an ordinary
        linear attribution graph with the gate feature as its target. Nodes:
        inputs up to t, candidate features and errors before t, h_init, and
        the target. Returns (edges, normalized edges, target weights for
        GraphPruner.prune_graph).
        """
        nodes = self._rooted_nodes(acts, name)
        edges = self._compute_edges(nodes, acts)
        return edges, self._normalize_by_group(nodes, edges), {name: 1.0}

    def _rooted_nodes(self, acts: Dict[str, torch.Tensor], name: str) -> List[CircuitNode]:
        """Node list of the graph rooted at gate feature `name` (validated)."""
        key = name[:3]
        try:
            target = self.node_from_name(name)
        except (ValueError, IndexError):
            target = None
        if (target is None or key not in ("f_z", "f_r") or key not in acts
                or not (0 <= target.timestep < acts[key].shape[0] and 0 <= target.feature_idx < acts[key].shape[1])
                or acts[key][target.timestep, target.feature_idx] <= 0):
            raise ValueError(f"{name!r} is not an active gate feature (expected f_z_t_j or f_r_t_j)")
        t = target.timestep
        nodes = [node for node in self._build_nodes(acts)
                 if node.node_type != "output"
                 and (node.timestep < t or (node.node_type == "input" and node.timestep == t))]
        nodes.append(target)
        return nodes

    def _node_value(self, node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Scalar value of a feature node (its JumpReLU output)."""
        return acts[node.name[:3]][node.timestep, node.feature_idx]

    @torch.no_grad()
    def gate_feature_ranking(self, acts: Dict[str, torch.Tensor], target_weights: Dict[str, float],
                             root: Optional[str] = None,
                             edges: Optional[Dict[Tuple[str, str], float]] = None
                             ) -> List[Tuple[str, float, str]]:
        """Rank gate pieces by their exact effect on a weighted target.

        score(piece) = how much sum_T target_weights[T] * value(T) would drop
        if the piece were removed from its gate, with all other gates and all
        JumpReLU on/off patterns fixed. Since the frozen model is linear in
        any one step's gate, this is exact:
            score = sum over edges e through that step of  share_piece(e) * D[dst(e)]
        with D[n] = d(weighted target)/d(value of n) through all paths. It is
        computed per step from aggregates rather than per edge, using the
        actual hidden states, so flow from bias writes is included too.

        An update-gate piece at step u acts twice: letting the candidate in
        (write) and erasing h_{u-1} (carry); its score is the sum of both.

        Args:
            target_weights: {node name: weight}, e.g. the pruner's logit
                weights, or {root: 1.0} for a rooted graph.
            root: If given, rank for the graph rooted at this gate feature.
            edges: Precomputed edges for the same node list (optional).
        Returns:
            [(piece name, score, "update@u" / "reset@u")], sorted by |score|.
        """
        nodes = self._rooted_nodes(acts, root) if root else self._build_nodes(acts)
        if edges is None:
            edges = self._compute_edges(nodes, acts)
        T, H = acts["h_ts"].shape
        dev, dtype = acts["h_ts"].device, acts["h_ts"].dtype

        # D: sensitivity of the weighted target to each feature/output value,
        # through all paths: D = w (I - J)^-1 with J the local derivatives
        # (edge weight / source value) between feature and output nodes.
        scalar = [n for n in nodes if n.node_type in ("feature", "output")]
        pos = {n.name: i for i, n in enumerate(scalar)}
        J = torch.zeros(len(scalar), len(scalar), device=dev, dtype=dtype)
        for (src, dst), w in edges.items():
            if src in pos and dst in pos:
                J[pos[dst], pos[src]] = w / self._node_value(scalar[pos[src]], acts)
        w_vec = torch.tensor([target_weights.get(n.name, 0.0) for n in scalar], device=dev, dtype=dtype)
        D = torch.linalg.solve((torch.eye(len(scalar), device=dev, dtype=dtype) - J).T, w_vec)

        # Aggregate reads per h step: R[tau] = sum_r D[r] * read_r (gated),
        # and per-step candidate reads without their reset gate.
        R = torch.zeros(T + 1, H, device=dev, dtype=dtype)      # row tau + 1
        R_read = torch.zeros(T, H, device=dev, dtype=dtype)     # f_n readers at t, without r_t
        for n in scalar:
            tau, read, gate = self._read_terms(n, acts)
            R[tau + 1] += D[pos[n.name]] * (read if gate is None else gate * read)
            if gate is not None:
                R_read[n.timestep] += D[pos[n.name]] * read

        # Q[u] = sum_{tau >= u} R[tau] * C(u, tau), built backwards.
        z = acts["z_ts"]
        Q = torch.zeros(T, H, device=dev, dtype=dtype)
        running = torch.zeros(H, device=dev, dtype=dtype)
        for u in range(T - 1, -1, -1):
            running = R[u + 1] + (1.0 - z[u + 1]) * running if u + 1 < T else R[u + 1].clone()
            Q[u] = running

        ranking = []
        for u in range(T):
            h_prev = acts["h_prevs"][u]                                         # h_{u-1}
            # update gate at u: write (lets n_u in) minus carry (erases h_{u-1})
            V = Q[u] * (acts["h_new_ts"][u] - h_prev)
            if V.abs().sum() > 0:
                names, P = self._gate_piece_matrix(acts, "update", u)
                ranking += [(self._step_name(name, u), float(s), f"update@{u}")
                            for name, s in zip(names, (V @ P).tolist())]
            V = R_read[u] * h_prev
            if V.abs().sum() > 0:
                names, P = self._gate_piece_matrix(acts, "reset", u)
                ranking += [(self._step_name(name, u), float(s), f"reset@{u}")
                            for name, s in zip(names, (V @ P).tolist())]
        return sorted(ranking, key=lambda item: -abs(item[1]))

    @staticmethod
    def _step_name(name: str, u: int) -> str:
        """Bias pieces carry no step in their name; add it for the ranking."""
        return f"{name}_{u}" if name.endswith("_bias") else name

if __name__ == "__main__":
    import pickle
    from models.rnn import RNN
    from models.transcoders import Transcoder
    from circuit.copy_find_features import CopyFeatureActivationAnalyzer
    from torch.utils.data import StackDataset
    torch.serialization.add_safe_globals([StackDataset])


    rnn_model = RNN(input_size=31, hidden_size=128, out_size=30, use_gru=True, num_layers=1)
    rnn_model.load_state_dict(torch.load("/w/150/lambda_squad/misc/rnnsuperposition/data/models/copy_train/copy_128_high/copy_128_high.ckpt"))
    update_transcoder = Transcoder(input_size=159, out_size=128, n_feats=64)
    hidden_transcoder = Transcoder(input_size=159, out_size=128, n_feats=128)
    hidden_transcoder.load_state_dict(torch.load("/w/150/lambda_squad/misc/rnnsuperposition/data/models/copy_transcoder/local_models/128_hctx_transcoder_hsparse_hc/final_model.ckpt")["transcoder"])
    update_transcoder.load_state_dict(torch.load("/w/150/lambda_squad/misc/rnnsuperposition/data/models/copy_transcoder/local_models/64_update_transcoder/final_model.ckpt")["transcoder"])

    datasets = torch.load("/w/nobackup/436/lambda/data/copy_transcoder/1M_128_seq4.pt")
    sequence_index = 84
    sequence_tensor = datasets[sequence_index]
    feature_analyzer = CopyFeatureActivationAnalyzer(rnn_model, update_transcoder, hidden_transcoder)
    
    tokens = feature_analyzer.convert_sequence_to_text(
        sequence_tensor["inputs"], sequence_tensor["outputs"]
    )
    
    # Get active features
    with open("/w/nobackup/436/lambda/data/copy_transcoder_features/h128_u64_features.p".replace("features.p", "sequences.p"), "rb") as f:
        analysis_dict_sequences = pickle.load(f)
    data_dict = analysis_dict_sequences
    feature_analyzer.sequence_activations = analysis_dict_sequences
    active_features = {
        'update': [(t, data_dict["update"][tokens][t]["features"][i], 
                data_dict["update"][tokens][t]["magnitudes"][i]) 
                for t in range(len(tokens)) 
                for i in range(len(data_dict["update"][tokens][t]["features"]))],
        'hidden': [(t, data_dict["hidden"][tokens][t]["features"][i], 
                data_dict["hidden"][tokens][t]["magnitudes"][i]) 
                for t in range(len(tokens)) 
                for i in range(len(data_dict["hidden"][tokens][t]["features"]))]
    }

    circuit_tracer = CircuitTracer(rnn_model, update_transcoder, hidden_transcoder)
    # with open("/w/150/lambda_squad/misc/rnnsuperposition/sequence_example.p", "rb") as f:
    #     sequences = pickle.load(f)
    
    # with open("/w/150/lambda_squad/misc/rnnsuperposition/active_features.p", "rb") as f:
    #     active_features = pickle.load(f)

    edge_weights, _ =circuit_tracer.build_circuit_graph(sequence_tensor, active_features)
    from circuit.graph_prune import GraphPruner
    pruner = GraphPruner(0.9, 0.95)
    pruner.prune_graph(edge_weights, circuit_tracer.run_forward_pass(sequence_tensor)["logits"].cpu())

#("f_n" in src.name or "f_z " in src.name) and dst.name in ("o_3_1", "o_4_28", "o_5_28") and src.name.split("_")[-2] == dst.name.split("_")[-2]
