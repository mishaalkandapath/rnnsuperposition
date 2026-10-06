from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class CircuitNode:
    """Represents a node in the circuit graph.
    name:
        Use patterns like: "x_{t}_{i}", "f_r_{t}_{j}", "f_z_{t}_{j}",
        "f_n_{t}_{j}", "o_{t}_{k}".
    node_type:
        'input' | 'feature' | 'hidden' | 'error' | 'output'
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
    source times its local derivative on the target. Hidden coordinates are
    explicit nodes, so direct recurrent carry and gate-mediated recurrence are
    represented as distinct paths.
    """

    def __init__(
        self,
        rnn_model: nn.Module,
        update_transcoder: nn.Module,
        hidden_transcoder: nn.Module,
        device: str = "cuda",
        reset_transcoder: Optional[nn.Module] = None,
    ):
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
        """
        acts = {k: v.detach().to(self.device).clone() for k, v in sequence.items()}

        T = acts["inputs"].shape[0]

        z = acts["z_ts"]
        # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t  (using symbols from the user's code)
        acts["h_ts"] = (1.0 - z) * acts["h_prevs"] + z * acts["h_new_ts"]  # (T, H)

        # New trace datasets retain the actual output logits.  Older datasets
        # saved only argmax one-hot tokens; recover their logits from the saved
        # hidden states and the frozen RNN output projection instead of
        # softmaxing those one-hot feedback tokens.
        if "logits" not in acts:
            output_steps = acts["outputs"].shape[0]
            acts["logits"] = torch.nn.functional.linear(
                acts["h_ts"][-output_steps:], self.W_o, self.b_o)

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

    def get_active_features(self, sequence: Dict[str, torch.Tensor]) -> Dict[str, List[Tuple[int, int, float]]]:
        """Public helper for visualizers: live, checkpoint-consistent features."""
        return self._active_features_from_acts(self.run_forward_pass(sequence))

    def _source_value(self, node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prompt-specific scalar represented by a source node."""
        t = node.timestep
        if node.node_type == "input":
            return acts["inputs"][t, node.input_dim]
        if node.node_type == "hidden":
            return acts["h_ts"][t, node.hidden_dim]
        if node.node_type == "feature":
            if node.name.startswith("f_r_"):
                return acts["f_r"][t, node.feature_idx]
            if node.name.startswith("f_z_"):
                return acts["f_z"][t, node.feature_idx]
            return acts["f_n"][t, node.feature_idx]
        if node.node_type == "error":
            raise ValueError("Error nodes are vector-valued source adjustments")
        raise ValueError(f"{node.name} cannot be an attribution-edge source")

    def _attribution_weight(
        self, from_node: CircuitNode, virtual_weight: torch.Tensor, acts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Prompt-specific edge attribution = source value × local weight."""
        return self._source_value(from_node, acts) * virtual_weight

    def _feature_to_hidden_weight(
        self, from_node: CircuitNode, to_node: CircuitNode, acts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Direct local edge from a gate/candidate feature to h_t[i]."""
        t, j, i = from_node.timestep, from_node.feature_idx, to_node.hidden_dim
        h_prev = acts["h_prevs"][t]
        Dz = -h_prev + acts["n_hat"][t] + acts["e_n"][t]
        Dn = acts["z_hat"][t] + acts["e_z"][t]
        if from_node.name.startswith("f_z_"):
            return Dz[i] * self.M_z[i, j]
        if from_node.name.startswith("f_n_"):
            return Dn[i] * self.M_n[i, j]
        # Reset features feed h_t only via reset -> candidate -> h_t.
        return torch.tensor(0.0, device=self.device)

    def _hidden_to_feature_weight(
        self, from_node: CircuitNode, to_node: CircuitNode, acts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Local h_t[i] -> gate feature at t+1 edge, holding sibling routes fixed."""
        t, i, j = to_node.timestep, from_node.hidden_dim, to_node.feature_idx
        if to_node.name.startswith("f_z_"):
            return self._feature_mask(acts["f_z"][t])[j] * self.W_z_h[j, i]
        if to_node.name.startswith("f_n_"):
            # The reset-mediated part has its own h -> f_r -> f_n path.
            return (self._feature_mask(acts["f_n"][t])[j]
                    * self.W_n_h[j, i] * acts["r_ts"][t, i])
        if self.reset_transcoder and to_node.name.startswith("f_r_"):
            return self._feature_mask(acts["f_r"][t])[j] * self.W_r_h[j, i]
        return torch.tensor(0.0, device=self.device)

    def compute_edge_weight(self, from_node: CircuitNode, to_node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return scalar edge weight from `from_node` to `to_node`.

        The graph decomposes each hidden transition into feature -> h,
        h -> h through the direct carry, and h -> next-gate-feature edges.
        This avoids double-counting routes through gate features.
        """
        t0, t1 = from_node.timestep, to_node.timestep
        if not (0 <=t1-t0 <= 1):
            # No backward-in-time causal edges, and nothing more than one time-step away
            return torch.tensor(0.0, device=self.device)
        elif from_node.node_type == "input":
            if to_node.node_type != "feature" or t0!=t1:
                return torch.tensor(0.0, device=self.device)
            else:
                if "f_z" in to_node.name:
                    return self._attribution_weight(from_node, self.W_z_x[to_node.feature_idx, from_node.input_dim] * self._feature_mask(acts["f_z"][t0])[to_node.feature_idx], acts)
                if "f_n" in to_node.name:
                    return self._attribution_weight(from_node, self.W_n_x[to_node.feature_idx, from_node.input_dim] * self._feature_mask(acts["f_n"][t0])[to_node.feature_idx], acts)
                if self.reset_transcoder and "f_r" in to_node.name:
                    return self._attribution_weight(from_node, self.W_r_x[to_node.feature_idx, from_node.input_dim] * self._feature_mask(acts["f_r"][t0])[to_node.feature_idx], acts)
            return torch.tensor(0.0, device=self.device)

        if from_node.node_type == "feature" and to_node.node_type == "hidden":
            if t0 == t1:
                return self._attribution_weight(from_node, self._feature_to_hidden_weight(from_node, to_node, acts), acts)
            return torch.tensor(0.0, device=self.device)

        if from_node.node_type == "hidden":
            if to_node.node_type == "output" and t0 == t1:
                return self._attribution_weight(from_node, (self.W_o[to_node.feature_idx, from_node.hidden_dim]
                        - self.W_o[:, from_node.hidden_dim].mean()), acts)
            if to_node.node_type == "hidden" and t1 == t0 + 1:
                if from_node.hidden_dim == to_node.hidden_dim:
                    return self._attribution_weight(from_node, 1.0 - acts["z_ts"][t1, from_node.hidden_dim], acts)
                return torch.tensor(0.0, device=self.device)
            if to_node.node_type == "feature" and t1 == t0 + 1:
                return self._attribution_weight(from_node, self._hidden_to_feature_weight(from_node, to_node, acts), acts)
            return torch.tensor(0.0, device=self.device)

        if (from_node.node_type == "feature" and to_node.node_type == "feature" and t1 == t0):
            # The reset factor creates the one same-timestep feature path:
            # f_r -> r -> (r*h_prev) -> candidate preactivation -> f_n.
            if (self.reset_transcoder and from_node.name.startswith("f_r_")
                    and to_node.name.startswith("f_n_")):
                t = t0
                j, k = from_node.feature_idx, to_node.feature_idx
                reset_delta = torch.diag(acts["h_prevs"][t]) @ self.M_r[:, j]
                return self._attribution_weight(from_node, (self._feature_mask(acts["f_n"][t])[k]
                        * torch.dot(self.W_n_h[k, :], reset_delta)), acts)
            return torch.tensor(0.0, device=self.device)

        if from_node.node_type == "error":
            if from_node.name.startswith("e_z_") and to_node.node_type == "hidden" and t0 == t1:
                i = to_node.hidden_dim
                h_prev = acts["h_prevs"][t0]
                virtual_weight = -h_prev[i] + acts["n_hat"][t0, i] + acts["e_n"][t0, i]
                return acts["e_z"][t0, i] * virtual_weight
            if from_node.name.startswith("e_n_") and to_node.node_type == "hidden" and t0 == t1:
                i = to_node.hidden_dim
                virtual_weight = acts["z_hat"][t0, i] + acts["e_z"][t0, i]
                return acts["e_n"][t0, i] * virtual_weight
            if (from_node.name.startswith("e_r_") and to_node.node_type == "feature"
                    and to_node.name.startswith("f_n_") and t0 == t1):
                j = to_node.feature_idx
                virtual_weight = (self._feature_mask(acts["f_n"][t0])[j]
                                  * self.W_n_h[j, :] * acts["h_prevs"][t0])
                return torch.dot(acts["e_r"][t0], virtual_weight)
            return torch.tensor(0.0, device=self.device)

        # All feature-to-output and feature-to-next-feature paths now factor
        # through h. Keeping shortcut edges would count them twice.
        return torch.tensor(0.0, device=self.device)

    def build_circuit_graph(
        self,
        sequence: Dict[str, torch.Tensor],
        active_features: Dict[str, List[Tuple[int, int, float]]],
    ) -> Dict[Tuple[str, str], float]:
        """Build edge map {(from_name, to_name): weight} for relevant nodes.

        ``active_features`` is retained for call compatibility but deliberately
        ignored: features are recomputed from the loaded models on this exact
        sequence, preventing stale feature-cache/model mismatches.
        """
        acts = self.run_forward_pass(sequence)
        T = sequence["inputs"].shape[0]
        active_features = self._active_features_from_acts(acts)
        nodes: List[CircuitNode] = []
        # Inputs # TODO ull need to change this for RL
        for t in range(T): # only the active one is required.
                active_dim = sequence["inputs"][t].argmax().item()
                nodes.append(CircuitNode(f"x_{t}_{active_dim}", "input", t, input_dim=active_dim))

        # Features (only active)
        for kind, feats in active_features.items():
            for t, j, mag in feats:
                if mag < 1e-5:
                    continue
                if kind == "reset" and self.reset_transcoder:
                    nodes.append(CircuitNode(f"f_r_{t}_{j}", "feature", t, feature_idx=j))
                elif kind == "update":
                    nodes.append(CircuitNode(f"f_z_{t}_{j}", "feature", t, feature_idx=j))
                elif kind == "hidden":
                    nodes.append(CircuitNode(f"f_n_{t}_{j}", "feature", t, feature_idx=j))

        # Post-update hidden coordinates. Scalar nodes are necessary because
        # the carry map is diagonal but gate and decoder maps mix coordinates.
        hidden_size = acts["h_ts"].shape[1]
        for t in range(T):
            for i in range(hidden_size):
                nodes.append(CircuitNode(f"h_{t}_{i}", "hidden", t, hidden_dim=i))

        # Frozen reconstruction residuals are source-only nodes, analogous to
        # error nodes in a local replacement model. They expose computation
        # not captured by the feature dictionaries.
        for t in range(T):
            nodes.append(CircuitNode(f"e_z_{t}", "error", t))
            nodes.append(CircuitNode(f"e_n_{t}", "error", t))
            nodes.append(CircuitNode(f"e_r_{t}", "error", t))

        # Outputs
        sorted_outs = torch.argsort(acts["logits"], dim=-1, descending=True)
        for t in range(T):
            if t < T//2: continue
            for k in sorted_outs[t-T//2].tolist():
                nodes.append(CircuitNode(f"o_{t}_{k}", "output", t, feature_idx=k))

        # Compute edges
        edge_weights: Dict[Tuple[str, str], float] = {}
        edge_weights_normalized: Dict[Tuple[str, str], float] = {}
        source_dest_types: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                w = self.compute_edge_weight(src, dst, acts)
                if not torch.isfinite(w) or abs(float(w)) < 1e-6:
                    continue
                edge_weights[(src.name, dst.name)] = float(w)
                src_type = src.node_type
                dst_type = dst.node_type
                if src_type == "feature":
                    src_type += src.name[2]
                if dst_type == "feature":
                    dst_type += dst.name[2] # n or z?
                src_t, dst_t = src.name.split("_")[-2], dst.name.split("_")[-2]
                src_type += src_t
                dst_type += dst_t
                if (src_type, dst_type) not in source_dest_types:
                    source_dest_types[(src_type, dst_type)] = []
                source_dest_types[(src_type, dst_type)].append((src.name, dst.name, float(w)))
        
        #normalize:
        for src_type, dst_type in source_dest_types:
            max_weight = max(max(source_dest_types[(src_type, dst_type)], key=lambda x: x[2])[2], abs(min(source_dest_types[(src_type, dst_type)], key=lambda x: x[2])[2]))
            if abs(float(max_weight)) < 1e-6: continue # nothing to add
            new_weights = [(x[0], x[1], x[2]/max_weight) for x in source_dest_types[(src_type, dst_type)]]
            for edge in new_weights:
                edge_weights_normalized[(edge[0], edge[1])] = edge[2]

        return edge_weights, edge_weights_normalized

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
