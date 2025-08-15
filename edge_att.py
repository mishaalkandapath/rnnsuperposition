import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CircuitNode:
    """Represents a node in the circuit graph"""
    name: str
    node_type: str  # 'feature', 'input', 'output'
    timestep: int
    feature_idx: Optional[int] = None  # For feature nodes
    input_dim: Optional[int] = None   # For input nodes
    
class CircuitTracer:
    """Compute edge attribution weights for RNN transcoder circuits"""
    
    def __init__(self, 
                 rnn_model: nn.Module,
                 update_transcoder: nn.Module, 
                 hidden_transcoder: nn.Module,
                 device: str = 'cuda'):
        """
        Args:
            rnn_model: Trained RNN model
            update_transcoder: Trained update gate transcoder  
            hidden_transcoder: Trained hidden context transcoder
            device: Device for computations
        """
        self.rnn_model = rnn_model.to(device)
        self.update_transcoder = update_transcoder.to(device)
        self.hidden_transcoder = hidden_transcoder.to(device)
        self.device = device
        
        # Set to eval mode
        self.rnn_model.eval()
        self.update_transcoder.eval()
        self.hidden_transcoder.eval()
        
        # Extract weight matrices
        self.W_z_h = self.update_transcoder.input_to_features.weight[:, :rnn_model.hidden_size]  # Hidden part
        self.W_z_x = self.update_transcoder.input_to_features.weight[:, rnn_model.hidden_size:]   # Input part
        self.M_z = self.update_transcoder.features_to_outputs.weight  # Decoder matrix
        self.b_z_enc = self.update_transcoder.input_to_features.bias
        self.b_z_dec = self.update_transcoder.features_to_outputs.bias
        
        self.W_n_h = self.hidden_transcoder.input_to_features.weight[:, :rnn_model.hidden_size]   # Hidden part  
        self.W_n_x = self.hidden_transcoder.input_to_features.weight[:, rnn_model.hidden_size:]    # Input part
        self.M_n = self.hidden_transcoder.features_to_outputs.weight  # Decoder matrix
        self.b_n_enc = self.hidden_transcoder.input_to_features.bias
        self.b_n_dec = self.hidden_transcoder.features_to_outputs.bias
        
        # Output weights (assuming final linear layer exists)
        if hasattr(rnn_model, 'layers') and len(rnn_model.layers) > rnn_model.num_layers:
            self.W_o = rnn_model.layers[-1].weight  # Output layer weights
        else:
            self.W_o = torch.eye(rnn_model.hidden_size, device=device)  # Identity if no output layer
            
    def run_forward_pass(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and collect all intermediate activations
        
        Args:
            sequence: Input sequence (1, seq_len, input_size) 
            
        Returns:
            Dictionary containing all intermediate values
        """
        with torch.no_grad():
            # Run RNN to get gates and hidden states
            outputs, final_hidden, r_records, z_records, h_new_records, h_records = self.rnn_model(
                sequence, record_gates=True
            )
            
            seq_len = sequence.shape[1]
            activations = {}
            
            # Store RNN activations (remove layer dimension since we have single layer)
            activations['r_t'] = r_records[0]  # (1, seq_len, hidden_size)
            activations['z_t'] = z_records[0]  # (1, seq_len, hidden_size) 
            activations['n_t'] = h_new_records[0]  # (1, seq_len, hidden_size)
            activations['h_t'] = outputs  # (1, seq_len, hidden_size)
            activations['h_prev'] = h_records[0]  # (1, seq_len, hidden_size)
            activations['x_t'] = sequence  # (1, seq_len, input_size)
            activations['o_t'] = torch.matmul(outputs, self.W_o.T)  # (1, seq_len, output_size)
            
            # Compute transcoder activations
            for t in range(seq_len):
                h_prev_t = activations['h_prev'][0, t]  # (hidden_size,)
                x_t = activations['x_t'][0, t]  # (input_size,)
                r_t = activations['r_t'][0, t]  # (hidden_size,)
                
                # Update gate transcoder
                z_input = torch.cat([h_prev_t, x_t])  # Concatenate hidden and input
                pf_z_t = torch.matmul(self.update_transcoder.input_to_features.weight, z_input) + self.b_z_enc
                f_z_t = torch.relu(pf_z_t)
                z_hat_t = torch.matmul(self.M_z, f_z_t) + self.b_z_dec
                e_z_t = activations['z_t'][0, t] - z_hat_t
                
                activations[f'pf_z_{t}'] = pf_z_t
                activations[f'f_z_{t}'] = f_z_t
                activations[f'z_hat_{t}'] = z_hat_t
                activations[f'e_z_{t}'] = e_z_t
                
                # Hidden context transcoder
                gated_hidden = r_t * h_prev_t
                n_input = torch.cat([gated_hidden, x_t])
                pf_n_t = torch.matmul(self.hidden_transcoder.input_to_features.weight, n_input) + self.b_n_enc
                f_n_t = torch.relu(pf_n_t)
                n_hat_t = torch.matmul(self.M_n, f_n_t) + self.b_n_dec
                e_n_t = activations['n_t'][0, t] - n_hat_t
                
                activations[f'pf_n_{t}'] = pf_n_t
                activations[f'f_n_{t}'] = f_n_t  
                activations[f'n_hat_{t}'] = n_hat_t
                activations[f'e_n_{t}'] = e_n_t
                
        return activations
        
    def get_node_vectors(self, node: CircuitNode, 
                         activations: Dict[str, torch.Tensor],
                         from_node = None, ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get input and output vectors for a node
        
        Returns:
            (input_vector, output_vector) - either can be None based on node type
        """
        if node.node_type == 'input':
            # Input nodes only have output vectors (columns of encoding matrices)
            if f'x_{node.timestep}' in node.name:
                if 'z' in node.name:
                    output_vector = self.W_z_x[:, node.input_dim]  # Column of input-to-features matrix
                else:  # 'n' in node.name
                    output_vector = self.W_n_x[:, node.input_dim]
                return None, output_vector
            
        elif node.node_type == 'feature':
            if 'f_z' in node.name:
                # Update gate feature node
                input_vector = self.W_z_h[:, node.feature_idx] if from_node and 'h' in from_node.name else self.W_z_x[:, node.feature_idx]
                output_vector = self.M_z[node.feature_idx, :]  # Row of decoder matrix
                return input_vector, output_vector
            elif 'f_n' in node.name:
                # Hidden context feature node  
                input_vector = self.W_n_h[:, node.feature_idx] if from_node and 'h' in from_node.name else self.W_n_x[:, node.feature_idx]
                output_vector = self.M_n[node.feature_idx, :]
                return input_vector, output_vector
                
        elif node.node_type == 'output':
            # Output nodes only have input vectors
            o_t = activations['o_t'][0, node.timestep]  # (output_size,)
            mean_o = torch.mean(o_t)
            
            # Gradient of (o_ti - mean(o_t)) w.r.t hidden state
            grad_input = self.W_o[node.feature_idx, :]  # Row of output weight matrix
            return grad_input, None
            
        return None, None
        
    def compute_jacobians(self, activations: Dict[str, torch.Tensor], timestep: int) -> Dict[str, torch.Tensor]:
        """Compute all Jacobian matrices for a given timestep"""
        jacobians = {}
        
        # Extract values for this timestep
        h_prev = activations['h_prev'][0, timestep]
        z_hat = activations[f'z_hat_{timestep}']
        n_hat = activations[f'n_hat_{timestep}'] 
        e_z = activations[f'e_z_{timestep}']
        e_n = activations[f'e_n_{timestep}']
        f_z = activations[f'f_z_{timestep}']
        f_n = activations[f'f_n_{timestep}']
        pf_z = activations[f'pf_z_{timestep}']
        pf_n = activations[f'pf_n_{timestep}']
        
        # ∂ẑ_t/∂f^z_t = M^z (element-wise, so just M^z for matrix multiply)
        jacobians['z_hat_to_f_z'] = self.M_z
        
        # ∂n̂_t/∂f^n_t = M^n  
        jacobians['n_hat_to_f_n'] = self.M_n
        
        # ∂ĥ_t/∂ẑ_t = -[h_{t-1} + n̂_t + ê_{n_t}] (diagonal)
        diag_h_z = -(h_prev + n_hat + e_n)
        jacobians['h_hat_to_z_hat'] = torch.diag(diag_h_z)
        
        # ∂ĥ_t/∂n̂_t = [ẑ_t + ê_{z_t}] (diagonal)
        diag_h_n = z_hat + e_z
        jacobians['h_hat_to_n_hat'] = torch.diag(diag_h_n)
        
        # ∂ĥ_t/∂h_{t-1} = 1 - [ẑ_t + ê_{z_t}] (diagonal)
        diag_h_prev = 1 - (z_hat + e_z)
        jacobians['h_hat_to_h_prev'] = torch.diag(diag_h_prev)
        
        # ∂ĥ_t/∂ê_{z_t} = -[h_{t-1} + n̂_t + ê_{n_t}] (diagonal)  
        jacobians['h_hat_to_e_z'] = torch.diag(-diag_h_z)
        
        # ∂ĥ_t/∂ê_{n_t} = [ẑ_t + ê_{z_t}] (diagonal)
        jacobians['h_hat_to_e_n'] = torch.diag(diag_h_n)
        
        # ∂f^z_t/∂pf^z_t = f^z_t (element-wise, diagonal with ReLU derivative)
        relu_deriv_z = (pf_z > 0).float()
        jacobians['f_z_to_pf_z'] = torch.diag(relu_deriv_z)
        
        # ∂f^n_t/∂pf^n_t = f^n_t (element-wise, diagonal with ReLU derivative) 
        relu_deriv_n = (pf_n > 0).float()
        jacobians['f_n_to_pf_n'] = torch.diag(relu_deriv_n)
        
        # ∂pf^n_t/∂h_{t-1} = W^n_h
        jacobians['pf_n_to_h_prev'] = self.W_n_h
        
        # ∂pf^z_t/∂h_{t-1} = W^z_h  
        jacobians['pf_z_to_h_prev'] = self.W_z_h
        
        # ∂pf^n_t/∂x_t = W^n_x
        jacobians['pf_n_to_x'] = self.W_n_x
        
        # ∂pf^z_t/∂x_t = W^z_x
        jacobians['pf_z_to_x'] = self.W_z_x
        
        return jacobians
        
    def compute_edge_weights(self, 
                           from_node: CircuitNode, 
                           to_node: CircuitNode,
                           activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute edge attribution weights between two nodes
        
        Args:
            from_node: Source node
            to_node: Target node  
            activations: All intermediate activations
            
        Returns:
            Edge weight tensor
        """
        # Get node vectors
        from_input_vec, from_output_vec = self.get_node_vectors(from_node, activations)
        to_input_vec, to_output_vec = self.get_node_vectors(to_node, activations, from_node=from_node)
        
        if from_output_vec is None or to_input_vec is None:
            return torch.tensor(0.0, device=self.device)
            
        # Compute Jacobians for the relevant timesteps
        jacobians = {}
        for t in range(max(from_node.timestep, to_node.timestep) + 1):
            jacobians[f't_{t}'] = self.compute_jacobians(activations, t)
            
        # Chain the Jacobians between nodes based on the circuit path
        # This is a simplified version - full implementation would need to trace
        # the specific path between any two nodes through the circuit
        
        if from_node.timestep == to_node.timestep:
            # Same timestep - direct connection through Jacobian
            t = from_node.timestep
            jacobian_chain = self._get_single_timestep_jacobian(from_node, to_node, jacobians[f't_{t}'])
        else:
            # Cross-timestep connection (e.g., h_{t-1} -> h_t)
            # Simplified: assume identity for now
            jacobian_chain = torch.eye(from_output_vec.shape[0], device=self.device)
            
        # Final edge weight: input_vector^T @ jacobian_chain @ output_vector
        edge_weight = torch.matmul(
            to_input_vec.unsqueeze(0),
            torch.matmul(jacobian_chain, from_output_vec.unsqueeze(1))
        ).squeeze()
        
        return edge_weight

    def _get_single_timestep_jacobian(self, from_node: CircuitNode, 
                                      to_node: CircuitNode, 
                                      jacobians_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get Jacobian for same-timestep connections"""
        if 'f_z' in from_node.name and 'h_hat' in to_node.name:
            # f_z -> z_hat -> h_hat path
            return torch.matmul(
                jacobians_t['h_hat_to_z_hat'],
                jacobians_t['z_hat_to_f_z']
            )
        elif 'f_n' in from_node.name and 'h_hat' in to_node.name:
            # f_n -> n_hat -> h_hat path  
            return torch.matmul(
                jacobians_t['h_hat_to_n_hat'],
                jacobians_t['n_hat_to_f_n']
            )
        elif 'x' in from_node.name and 'f_z' in to_node.name:
            # x -> pf_z -> f_z path
            return torch.matmul(
                jacobians_t['f_z_to_pf_z'],
                jacobians_t['pf_z_to_x']
            )
        elif 'x' in from_node.name and 'f_n' in to_node.name:
            # x -> pf_n -> f_n path
            return torch.matmul(
                jacobians_t['f_n_to_pf_n'], 
                jacobians_t['pf_n_to_x']
            )
        elif 'h' in from_node.name and 'f_z' in to_node.name:
            # h_{t-1} -> pf_z -> f_z path (same timestep case)
            return torch.matmul(
                jacobians_t['f_z_to_pf_z'],
                jacobians_t['pf_z_to_h_prev']
            )
        elif 'h' in from_node.name and 'f_n' in to_node.name:
            # h_{t-1} -> pf_n -> f_n path (same timestep case)  
            return torch.matmul(
                jacobians_t['f_n_to_pf_n'],
                jacobians_t['pf_n_to_h_prev']
            )
        else:
            # Default to identity for unrecognized paths
            return torch.eye(from_node.feature_idx or activations['h_t'].shape[2], device=self.device)

    def _chain_jacobians_across_time(self, from_node: CircuitNode, to_node: CircuitNode,
                                    jacobians: Dict[str, Dict[str, torch.Tensor]], 
                                    activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Chain Jacobians across multiple timesteps"""
        start_t = from_node.timestep
        end_t = to_node.timestep
        
        if start_t > end_t:
            # Backward connection (shouldn't happen in causal RNN, but handle gracefully)
            return torch.zeros((activations['h_t'].shape[2], activations['h_t'].shape[2]), device=self.device)
        
        # Initialize with identity matrix
        hidden_size = activations['h_t'].shape[2]
        jacobian_chain = torch.eye(hidden_size, device=self.device)
        
        # Chain from start_t to end_t
        for t in range(start_t + 1, end_t + 1):
            # Get the Jacobian ∂h_t/∂h_{t-1} for this timestep
            # This involves chaining through: h_{t-1} -> z_t, n_t -> h_t
            
            # Path 1: h_{t-1} -> z_t -> h_t (through update gate)
            # ∂h_t/∂h_{t-1} via update gate = ∂h_t/∂z_t * ∂z_t/∂h_{t-1}
            jacobian_h_to_z = jacobians[f't_{t}']['h_hat_to_z_hat']  # ∂h_t/∂z_t
            jacobian_z_to_h_prev = jacobians[f't_{t}']['pf_z_to_h_prev']  # ∂z_t/∂h_{t-1} (through encoder)
            
            # Need to account for ReLU nonlinearity in features
            jacobian_z_features = jacobians[f't_{t}']['f_z_to_pf_z']  # ReLU derivative
            jacobian_z_decode = jacobians[f't_{t}']['z_hat_to_f_z']   # Decoder
            
            # Chain: h_{t-1} -> pf_z -> f_z -> z_hat -> h_t
            path1 = torch.matmul(
                torch.matmul(jacobian_h_to_z, jacobian_z_decode),
                torch.matmul(jacobian_z_features, jacobian_z_to_h_prev)
            )
            
            # Path 2: h_{t-1} -> n_t -> h_t (through new content)
            # ∂h_t/∂h_{t-1} via new content = ∂h_t/∂n_t * ∂n_t/∂h_{t-1}
            jacobian_h_to_n = jacobians[f't_{t}']['h_hat_to_n_hat']  # ∂h_t/∂n_t
            jacobian_n_to_h_prev = jacobians[f't_{t}']['pf_n_to_h_prev']  # ∂n_t/∂h_{t-1} (through encoder)
            
            # Need to account for ReLU and reset gate
            jacobian_n_features = jacobians[f't_{t}']['f_n_to_pf_n']  # ReLU derivative  
            jacobian_n_decode = jacobians[f't_{t}']['n_hat_to_f_n']   # Decoder
            
            # Chain: h_{t-1} -> pf_n -> f_n -> n_hat -> h_t
            path2 = torch.matmul(
                torch.matmul(jacobian_h_to_n, jacobian_n_decode),
                torch.matmul(jacobian_n_features, jacobian_n_to_h_prev)
            )
            
            # Path 3: Direct h_{t-1} -> h_t (through 1-z term)
            # ∂h_t/∂h_{t-1} direct = diagonal(1 - z_t)
            path3 = jacobians[f't_{t}']['h_hat_to_h_prev']
            
            # Total Jacobian for this timestep: sum of all paths
            timestep_jacobian = path1 + path2 + path3
            
            # Chain with accumulated Jacobian
            jacobian_chain = torch.matmul(timestep_jacobian, jacobian_chain)
        
        # Handle connection from specific node type at start_t to specific node type at end_t
        
        # Get the initial connection from from_node to hidden state at start_t
        if from_node.node_type == 'feature':
            if 'f_z' in from_node.name:
                # f_z -> z_hat -> h_t
                initial_jacobian = torch.matmul(
                    jacobians[f't_{start_t}']['h_hat_to_z_hat'],
                    jacobians[f't_{start_t}']['z_hat_to_f_z']
                )
            elif 'f_n' in from_node.name:
                # f_n -> n_hat -> h_t
                initial_jacobian = torch.matmul(
                    jacobians[f't_{start_t}']['h_hat_to_n_hat'],
                    jacobians[f't_{start_t}']['n_hat_to_f_n']
                )
            else:
                initial_jacobian = torch.eye(hidden_size, device=self.device)
        elif from_node.node_type == 'input':
            if 'z' in from_node.name:
                # x -> pf_z -> f_z -> z_hat -> h_t
                initial_jacobian = torch.matmul(
                    torch.matmul(
                        jacobians[f't_{start_t}']['h_hat_to_z_hat'],
                        jacobians[f't_{start_t}']['z_hat_to_f_z']
                    ),
                    torch.matmul(
                        jacobians[f't_{start_t}']['f_z_to_pf_z'],
                        jacobians[f't_{start_t}']['pf_z_to_x']
                    )
                )
            else:  # 'n' in from_node.name
                # x -> pf_n -> f_n -> n_hat -> h_t
                initial_jacobian = torch.matmul(
                    torch.matmul(
                        jacobians[f't_{start_t}']['h_hat_to_n_hat'],
                        jacobians[f't_{start_t}']['n_hat_to_f_n']
                    ),
                    torch.matmul(
                        jacobians[f't_{start_t}']['f_n_to_pf_n'],
                        jacobians[f't_{start_t}']['pf_n_to_x']
                    )
                )
        else:  # from_node.node_type == 'output' or hidden
            initial_jacobian = torch.eye(hidden_size, device=self.device)
        
        # Get the final connection from hidden state at end_t to to_node
        if to_node.node_type == 'output':
            # h_t -> o_t
            final_jacobian = self.W_o  # (output_size, hidden_size)
        else:
            final_jacobian = torch.eye(hidden_size, device=self.device)
        
        # Chain everything together
        if from_node.timestep == end_t:
            # No time chaining needed, just initial connection
            full_jacobian = torch.matmul(final_jacobian, initial_jacobian)
        else:
            # Chain: initial -> time propagation -> final
            full_jacobian = torch.matmul(
                final_jacobian,
                torch.matmul(jacobian_chain, initial_jacobian)
            )
        
        return full_jacobian
            
    def build_circuit_graph(self, 
                        sequence: torch.Tensor,
                        active_features: Dict[str, List[Tuple[int, int, float]]]) -> Dict[Tuple[str, str], float]:
        """
        Build complete circuit graph with edge weights
        
        Args:
            sequence: Input sequence
            active_features: Dict mapping 'update'/'hidden' to list of (timestep, feature_idx, magnitude)
            
        Returns:
            Dictionary mapping (from_node_name, to_node_name) -> edge_weight
        """
        activations = self.run_forward_pass(sequence)
        seq_len = sequence.shape[1]
        
        # Create all nodes
        nodes = []
        
        # Input nodes
        for t in range(seq_len):
            for i in range(sequence.shape[2]):  # Input dimensions
                nodes.append(CircuitNode(f'x_{t}_{i}', 'input', t, input_dim=i))
                
        # Feature nodes (only active ones)
        for transcoder_type, features in active_features.items():
            for timestep, feature_idx, magnitude in features:
                if transcoder_type == 'update':
                    nodes.append(CircuitNode(f'f_z_{timestep}_{feature_idx}', 'feature', timestep, feature_idx))
                else:  # hidden
                    nodes.append(CircuitNode(f'f_n_{timestep}_{feature_idx}', 'feature', timestep, feature_idx))
                    
        # Output nodes
        for t in range(seq_len):
            for i in range(activations['o_t'].shape[2]):  # Output dimensions
                nodes.append(CircuitNode(f'o_{t}_{i}', 'output', t, feature_idx=i))
                
        # Compute edge weights between all relevant pairs
        edge_weights = {}
        
        for i, from_node in enumerate(nodes):
            for j, to_node in enumerate(nodes):
                if i != j:  # No self-loops
                    try:
                        weight = self.compute_edge_weights(from_node, to_node, activations)
                        if abs(weight.item()) > 1e-6:  # Only store significant weights
                            edge_weights[(from_node.name, to_node.name)] = weight.item()
                    except Exception as e:
                        # Skip problematic edge combinations
                        continue
                        
        return edge_weights

# Example usage
if __name__ == "__main__":
    # Example usage:
    # tracer = CircuitTracer(rnn_model, update_transcoder, hidden_transcoder)
    # sequence = torch.randn(1, 5, 31)  # Batch=1, seq_len=5, input_dim=31
    # active_features = {
    #     'update': [(0, 10, 0.8), (1, 15, 0.6)],  # (timestep, feature_idx, magnitude) 
    #     'hidden': [(0, 5, 0.9), (2, 20, 0.7)]
    # }
    # edge_weights = tracer.build_circuit_graph(sequence, active_features)
    pass