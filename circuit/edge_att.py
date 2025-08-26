from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class CircuitNode:
    """Represents a node in the circuit graph.
    name:
        Use patterns like: "x_{t}_{i}", "f_z_{t}_{j}", "f_n_{t}_{j}", "o_{t}_{k}".
    node_type:
        'input' | 'feature' | 'output'
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

class CircuitTracer:
    """Compute edge attribution weights for RNN transcoder circuits.

    This rewrite fixes shape inconsistencies and attribution logic:
    - Within a single timestep, edge weights are computed using the local
      linear maps only (no Jacobian chaining across that timestep).
    - Across time, we propagate *only* through the hidden-to-hidden linearized
      transition A_t = d h_t / d h_{t-1} and multiply the initial and final
      local maps at the endpoints.
    """

    def __init__(
        self,
        rnn_model: nn.Module,
        update_transcoder: nn.Module,
        hidden_transcoder: nn.Module,
        device: str = "cuda",
    ):
        self.rnn_model = rnn_model.to(device)
        self.update_transcoder = update_transcoder.to(device)
        self.hidden_transcoder = hidden_transcoder.to(device)
        self.device = device

        self.rnn_model.eval()
        self.update_transcoder.eval()
        self.hidden_transcoder.eval()

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

        # Output projection: o_t = W_o h_t (+ b)  [assumed linear]
        if hasattr(rnn_model, "layers") and len(rnn_model.layers) > rnn_model.num_layers:
            self.W_o = rnn_model.layers[-1].weight.to(device)  # (O, H)
        else:
            print("No output weight?")
            self.W_o = torch.eye(Hz, device=device)  # (H, H)

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
        H = self.rnn_model.hidden_size

        z = acts["z_ts"]
        # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t  (using symbols from the user's code)
        acts["h_ts"] = (1.0 - z) * acts["h_prevs"] + z * acts["h_new_ts"]  # (T, H)

        # Pre/post feature activations are assumed to be produced by calling the transcoders.
        # We compute per-timestep masks needed for local linear maps.
        acts["pf_z"], acts["f_z"], acts["z_hat"], acts["e_z"] = [], [], [], []
        acts["pf_n"], acts["f_n"], acts["n_hat"], acts["e_n"] = [], [], [], []

        for t in range(T):
            h_prev_t = acts["h_prevs"][t]            # (H,)
            x_t = acts["inputs"][t]                 # (X,)
            r_t = acts["r_ts"][t]                   # (H,)

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
        for key in ["pf_z", "f_z", "z_hat", "e_z", "pf_n", "f_n", "n_hat", "e_n"]:
            acts[key] = torch.stack(acts[key], dim=0)

        return acts

    def _relu_mask(self, v: torch.Tensor) -> torch.Tensor:
        """Elementwise ReLU' mask for pre-activations (1 if > 0 else 0)."""
        return (v > 0).to(v.dtype)

    def _local_influence_to_hidden(self, node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Column vector v (H,) giving the effect on h_t from a unit at `node` at its timestep.

        Cases handled:
          - x_{t,i} -> h_t via z and n branches with ReLU masks
          - f_z_{t,j} -> h_t via M_z and diag(-h_prev - n_hat - e_n)
          - f_n_{t,j} -> h_t via M_n and diag(z_hat + e_z)
          - o_{t,k} is not a valid source in this formulation (returns zeros)
        """
        t = node.timestep
        h_prev = acts["h_prevs"][t]        # (H,)
        z_hat = acts["z_hat"][t]          # (H,)
        n_hat = acts["n_hat"][t]          # (H,)
        e_z = acts["e_z"][t]              # (H,)
        e_n = acts["e_n"][t]              # (H,)

        Dz = torch.diag(-(h_prev + n_hat + e_n))  # (H,H), d h_t / d z_hat_t
        Dn = torch.diag(z_hat + e_z)              # (H,H), d h_t / d n_hat_t

        if node.node_type == "input":
            i = node.input_dim
            # z path: x -> pf^z -> f^z -> z_hat -> h
            mask_z = self._relu_mask(acts["pf_z"][t])  # (Fz,)
            # Column of W_z_x for input i: (Fz,)
            col_Wzx = self.W_z_x[:, i]
            zhat_from_x = self.M_z @ (mask_z * col_Wzx)  # (H,)

            # n path: x -> pf^n -> f^n -> n_hat -> h
            mask_n = self._relu_mask(acts["pf_n"][t])  # (Fn,)
            col_Wnx = self.W_n_x[:, i]                  # (Fn,)
            nhat_from_x = self.M_n @ (mask_n * col_Wnx)  # (H,)

            v = Dz @ zhat_from_x + Dn @ nhat_from_x      # (H,)
            return v

        if node.node_type == "feature":
            j = node.feature_idx
            if node.name.startswith("f_z_"):
                # z_hat from feature j is M_z[:, j]
                v = Dz @ self.M_z[:, j]
                return v
            if node.name.startswith("f_n_"):
                v = Dn @ self.M_n[:, j]
                return v

        # Outputs as sources aren't supported; return zeros
        H = acts["h_ts"].shape[1]
        return torch.zeros(H, device=self.device, dtype=acts["h_ts"].dtype)

    def _local_sensitivity_from_hidden(self, node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Row vector w (H,) giving d node / d h_t at node.timestep.

        Cases:
          - o_{t,k}: w = W_o[k, :]
          - f_z_{t+Δ,j} (when used as *target* at its own time):
                w = mask_z_j * W_z_h[j, :]  (but note: this maps h_t -> pf^z_{t+1};
                only meaningful if used with h_t at the *same* t feeding into t+1 feature)
          - f_n_{t+Δ,j}: analogous with W_n_h
          - x_{t,i} is not a valid *target* from hidden in the same timestep.
        """
        t = node.timestep
        if node.node_type == "output":
            k = node.feature_idx
            return self.W_o[k, :] - self.W_o.mean(dim=0)  # (H,)

        if node.node_type == "feature":
            j = node.feature_idx
            if node.name.startswith("f_z_"):
                mask = self._relu_mask(acts["pf_z"][t])
                # row vector: (1,H)
                return mask[j] * self.W_z_h[j, :]
            if node.name.startswith("f_n_"):
                mask = self._relu_mask(acts["pf_n"][t])
                return mask[j] * self.W_n_h[j, :]

        H = acts["h_ts"].shape[1]
        return torch.zeros(H, device=self.device, dtype=acts["h_ts"].dtype)

    # -----------------------------
    # Across-time propagation A_t = d h_t / d h_{t-1}
    # -----------------------------
    def _A_t(self, t: int, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Linearized hidden transition Jacobian A_t = d h_t / d h_{t-1} (H,H).

        A_t = diag(1 - (z_hat_t + e_z_t))
              + diag(-(h_{t-1} + n_hat_t + e_n_t)) @ M_z @ diag(ReLU'(pf^z_t)) @ W_z_h
              + diag( z_hat_t + e_z_t)              @ M_n @ diag(ReLU'(pf^n_t)) @ W_n_h
        """
        h_prev = acts["h_prevs"][t]
        z_hat = acts["z_hat"][t]
        n_hat = acts["n_hat"][t]
        e_z = acts["e_z"][t]
        e_n = acts["e_n"][t]

        Ddir = torch.diag(1.0 - (z_hat + e_z))
        Dz = torch.diag(-(h_prev + n_hat + e_n))
        Dn = torch.diag(z_hat + e_z)

        mask_z = self._relu_mask(acts["pf_z"][t])
        mask_n = self._relu_mask(acts["pf_n"][t])

        A = Ddir
        if self.M_z.numel() > 0:
            A = A + Dz @ (self.M_z @ torch.diag(mask_z) @ self.W_z_h)
        if self.M_n.numel() > 0:
            A = A + Dn @ (self.M_n @ torch.diag(mask_n) @ self.W_n_h)
        return A

    # -----------------------------
    # Public: compute a single edge weight
    # -----------------------------
    def compute_edge_weight(self, from_node: CircuitNode, to_node: CircuitNode, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return scalar edge weight from `from_node` to `to_node`.

        Within-timestep:  weight = w_t @ v_t
        Across-time (t0 < t1): weight = w_{t1} @ (A_{t1} ... A_{t0+1}) @ v_{t0}
        """
        t0, t1 = from_node.timestep, to_node.timestep
        if t0 > t1:
            # No backward-in-time causal edges
            return torch.tensor(0.0, device=self.device)

        v = self._local_influence_to_hidden(from_node, acts)  # (H,)
        w = self._local_sensitivity_from_hidden(to_node, acts)  # (H,)

        if torch.all(v == 0) or torch.all(w == 0):
            return torch.tensor(0.0, device=self.device)

        # Same timestep: just local maps
        if t0 == t1:
            return torch.dot(w, v)

        # Across-time propagation
        A = torch.eye(v.shape[0], device=self.device, dtype=v.dtype)
        for t in range(t0 + 1, t1 + 1):
            A = self._A_t(t, acts) @ A
        return torch.dot(w, A @ v)

    def build_circuit_graph(
        self,
        sequence: Dict[str, torch.Tensor],
        active_features: Dict[str, List[Tuple[int, int, float]]],
    ) -> Dict[Tuple[str, str], float]:
        """Build edge map {(from_name, to_name): weight} for relevant nodes.

        active_features: {
          'update': [(t, feat_idx, magnitude), ...],
          'hidden': [(t, feat_idx, magnitude), ...],
        }
        Only features with magnitude >= 1e-5 are included.
        """
        acts = self.run_forward_pass(sequence)
        T = sequence["inputs"].shape[0]
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
                if kind == "update":
                    nodes.append(CircuitNode(f"f_z_{t}_{j}", "feature", t, feature_idx=j))
                elif kind == "hidden":
                    nodes.append(CircuitNode(f"f_n_{t}_{j}", "feature", t, feature_idx=j))

        # Outputs
        top_3 = torch.argsort(sequence["outputs"], dim=-1, descending=True)
        for t in range(T):
            if t < T//2: continue
            for k in top_3[t-T//2].tolist():
                nodes.append(CircuitNode(f"o_{t}_{k}", "output", t, feature_idx=k))

        # Compute edges
        edge_weights: Dict[Tuple[str, str], float] = {}
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                w = self.compute_edge_weight(src, dst, acts)
                if torch.isfinite(w) and abs(float(w)) > 1e-6:
                    edge_weights[(src.name, dst.name)] = float(w)

        return edge_weights
