from typing import Dict, List, Tuple, Set

import torch
import numpy as np
import networkx as nx

class GraphPruner:
    """
    Graph pruning for RNN circuit analysis following the specification
    """
    
    def __init__(self, 
                 node_threshold: float = 0.8,
                 edge_threshold: float = 0.98,
                 top_k_logits: int = 10,
                 logit_prob_threshold: float = 0.95):
        """
        Args:
            node_threshold: Threshold for node pruning (keep nodes with cumulative influence <= threshold)
            edge_threshold: Threshold for edge pruning 
            top_k_logits: Maximum number of logit nodes to keep
            logit_prob_threshold: Keep logit nodes until cumulative prob > threshold
        """
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.top_k_logits = top_k_logits
        self.logit_prob_threshold = logit_prob_threshold
        
    def convert_to_adjacency_matrix(self, 
                                  edge_weights: Dict[Tuple[str, str], float],
                                  node_names: List[str]) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
        """
        Convert edge weights dict to adjacency matrix
        
        Args:
            edge_weights: Dict mapping (from_node, to_node) -> weight
            node_names: List of all node names
            
        Returns:
            adjacency_matrix: A[j,i] = weight from i to j (transposed for row normalization)
            name_to_idx: Dict mapping node name to matrix index
            idx_to_name: Dict mapping matrix index to node name
        """
        n_nodes = len(node_names)
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        idx_to_name = {i: name for i, name in enumerate(node_names)}
        
        # Initialize adjacency matrix
        A = torch.zeros(n_nodes, n_nodes)
        
        # Fill adjacency matrix: A[j,i] = weight from i to j
        for (from_node, to_node), weight in edge_weights.items():
            if from_node in name_to_idx and to_node in name_to_idx:
                i = name_to_idx[from_node]
                j = name_to_idx[to_node]
                A[j, i] = weight
                
        return A, name_to_idx, idx_to_name
        
    def compute_normalized_adjacency_matrix(self, A: torch.Tensor) -> torch.Tensor:
        """
        Convert to unsigned adjacency matrix and normalize rows to sum to 1
        
        Args:
            A: Raw adjacency matrix
            
        Returns:
            Normalized unsigned adjacency matrix
        """
        # Take absolute values
        A_abs = torch.abs(A)
        
        # Normalize each row to sum to 1
        row_sums = torch.sum(A_abs, dim=1)
        row_sums = torch.maximum(row_sums, torch.tensor(1e-8))  # Avoid division by zero
        
        # Create diagonal matrix and normalize
        A_norm = A_abs / row_sums.unsqueeze(1)
        
        return A_norm
        
    def compute_indirect_influence_matrix(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute indirect influence matrix B = (I - A)^-1 - I
        
        Args:
            A: Normalized adjacency matrix
            
        Returns:
            Indirect influence matrix B
        """
        n = A.shape[0]
        I = torch.eye(n, dtype=A.dtype, device=A.device)
        
        try:
            # B = (I - A)^-1 - I
            B = torch.inverse(I - A) - I
        except:
            # If matrix is singular, use pseudo-inverse
            B = torch.pinverse(I - A) - I
            
        return B
        
    def get_logit_weights(self, 
                         node_names: List[str], 
                         output_probs: torch.Tensor = None) -> torch.Tensor:
        """
        Get weights for logit nodes based on output probabilities
        
        Args:
            node_names: List of node names
            output_probs: Output probabilities for each timestep/token
            
        Returns:
            logit_weights: Vector where entry is prob for logit nodes, 0 for others
        """
        n_nodes = len(node_names)
        logit_weights = torch.zeros(n_nodes)
        
        if output_probs is None:
            # If no probabilities provided, use uniform weights for output nodes
            for i, name in enumerate(node_names):
                if name.startswith('o_'):  # Output/logit nodes
                    logit_weights[i] = 1.0 / sum(1 for n in node_names if n.startswith('o_'))
        else:
            # Use provided probabilities
            for i, name in enumerate(node_names):
                if name.startswith('o_'):  # Output/logit nodes
                    # Extract timestep and output dimension from name like 'o_2_5'
                    parts = name.split('_')
                    if len(parts) >= 3:
                        try:
                            t = int(parts[1])
                            dim = int(parts[2])
                            if t < output_probs.shape[0] and dim < output_probs.shape[1]:
                                logit_weights[i] = output_probs[t, dim]
                        except ValueError:
                            continue
                            
        return logit_weights
        
    def prune_nodes_by_indirect_influence(self, 
                                        edge_weights: Dict[Tuple[str, str], float],
                                        node_names: List[str],
                                        output_probs: torch.Tensor = None) -> Set[str]:
        """
        Prune nodes based on indirect influence on logits
        
        Args:
            edge_weights: Dict of edge weights
            node_names: List of all node names
            output_probs: Output probabilities
            
        Returns:
            Set of node names to keep
        """
        # Convert to adjacency matrix
        A, name_to_idx, idx_to_name = self.convert_to_adjacency_matrix(edge_weights, node_names)
        
        # Normalize adjacency matrix
        A_norm = self.compute_normalized_adjacency_matrix(A)
        
        # Compute indirect influence matrix
        B = self.compute_indirect_influence_matrix(A_norm)
        
        # Get logit weights
        logit_weights = self.get_logit_weights(node_names, output_probs)
        
        # Calculate influence on logits for each node
        influence_on_logits = torch.matmul(B, logit_weights)
        
        # Separate logit and non-logit nodes
        logit_nodes = []
        non_logit_nodes = []
        
        for i, name in enumerate(node_names):
            if name.startswith('o_'):  # Logit/output nodes
                logit_nodes.append((i, name))
            else:
                non_logit_nodes.append((i, name, influence_on_logits[i].item()))
                
        # Prune non-logit nodes by influence
        non_logit_nodes.sort(key=lambda x: x[2], reverse=True)  # Sort by influence descending
        
        total_influence = sum(x[2] for x in non_logit_nodes)
        cumulative_influence = 0.0
        nodes_to_keep = set()
        
        for i, name, influence in non_logit_nodes:
            cumulative_influence += influence
            nodes_to_keep.add(name)
            
            if total_influence > 0 and cumulative_influence / total_influence >= self.node_threshold:
                break
                
        # Prune logit nodes separately
        if output_probs is not None:
            # Sort logit nodes by probability
            logit_probs = [(name, logit_weights[i].item()) for i, name in logit_nodes]
            logit_probs.sort(key=lambda x: x[1], reverse=True)
            
            cumulative_prob = 0.0
            logit_count = 0
            
            for name, prob in logit_probs:
                if cumulative_prob >= self.logit_prob_threshold or logit_count >= self.top_k_logits:
                    break
                nodes_to_keep.add(name)
                cumulative_prob += prob
                logit_count += 1
        else:
            # Keep all logit nodes if no probabilities provided
            for i, name in logit_nodes:
                nodes_to_keep.add(name)
                
        # Always keep embedding and error nodes
        for name in node_names:
            if name.startswith('x_') or name.startswith('e_'):  # Input/embedding or error nodes
                nodes_to_keep.add(name)
                
        return nodes_to_keep
        
    def prune_edges_by_thresholded_influence(self,
                                           edge_weights: Dict[Tuple[str, str], float],
                                           kept_nodes: Set[str],
                                           output_probs: torch.Tensor = None) -> Dict[Tuple[str, str], float]:
        """
        Prune edges based on thresholded influence
        
        Args:
            edge_weights: Dict of edge weights
            kept_nodes: Set of nodes that survived node pruning
            output_probs: Output probabilities
            
        Returns:
            Pruned edge weights dict
        """
        # Filter edge weights to only include kept nodes
        filtered_edges = {
            (from_node, to_node): weight 
            for (from_node, to_node), weight in edge_weights.items()
            if from_node in kept_nodes and to_node in kept_nodes
        }
        
        if not filtered_edges:
            return {}
            
        node_names = list(kept_nodes)
        A, name_to_idx, idx_to_name = self.convert_to_adjacency_matrix(filtered_edges, node_names)
        
        # Normalize adjacency matrix
        A_norm = self.compute_normalized_adjacency_matrix(A)
        
        # Compute indirect influence matrix
        B = self.compute_indirect_influence_matrix(A_norm)
        
        # Get logit weights
        logit_weights = self.get_logit_weights(node_names, output_probs)
        
        # Calculate node influence scores
        node_scores = torch.matmul(B, logit_weights)
        
        # For logit nodes, set score to their probability
        for i, name in enumerate(node_names):
            if logit_weights[i] > 0:  # Logit node
                node_scores[i] = logit_weights[i]
                
        # Calculate edge scores: edge score = target_node_score * normalized_edge_weight
        edge_scores = A_norm * node_scores.unsqueeze(0)  # Broadcasting
        
        # Flatten edge scores and sort
        edge_scores_flat = edge_scores.flatten()
        edge_scores_nonzero = edge_scores_flat[edge_scores_flat > 0]
        
        if len(edge_scores_nonzero) == 0:
            return filtered_edges
            
        sorted_scores, sorted_indices = torch.sort(edge_scores_nonzero, descending=True)
        
        # Calculate cumulative scores
        cumulative_scores = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
        
        # Find threshold index
        threshold_indices = torch.where(cumulative_scores >= self.edge_threshold)[0]
        if len(threshold_indices) > 0:
            threshold_idx = threshold_indices[0].item()
            threshold_score = sorted_scores[threshold_idx].item()
        else:
            threshold_score = 0.0
            
        # Create edge mask
        edge_mask = edge_scores >= threshold_score
        
        # Filter edges based on mask
        pruned_edges = {}
        for (from_node, to_node), weight in filtered_edges.items():
            i = name_to_idx[from_node]
            j = name_to_idx[to_node]
            
            if edge_mask[j, i]:  # Remember A[j,i] convention
                pruned_edges[(from_node, to_node)] = weight
                
        return pruned_edges
        
    def prune_graph(self, 
                   edge_weights: Dict[Tuple[str, str], float],
                   output_probs: torch.Tensor = None) -> Tuple[Dict[Tuple[str, str], float], Set[str]]:
        """
        Full graph pruning pipeline
        
        Args:
            edge_weights: Dict of edge weights
            output_probs: Output probabilities for logit nodes
            
        Returns:
            (pruned_edge_weights, kept_nodes)
        """
        # Extract all unique node names
        all_nodes = set()
        for from_node, to_node in edge_weights.keys():
            all_nodes.add(from_node)
            all_nodes.add(to_node)
        node_names = list(all_nodes)
        
        # Step 1: Prune nodes
        kept_nodes = self.prune_nodes_by_indirect_influence(edge_weights, node_names, output_probs)
        
        # Step 2: Prune edges
        pruned_edges = self.prune_edges_by_thresholded_influence(edge_weights, kept_nodes, output_probs)
        
        return pruned_edges, kept_nodes