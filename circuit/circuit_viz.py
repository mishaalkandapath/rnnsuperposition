from typing import Dict, List, Tuple, Optional, Set
import pickle

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import networkx as nx

import json
import os
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import StackDataset

from circuit.edge_att import CircuitTracer
from circuit.copy_find_features import CopyFeatureActivationAnalyzer, checkpoint_fingerprint
from circuit.rl_find_features import RLFeatureActivationAnalyzer
from circuit.graph_prune import GraphPruner

from models.rnn import RNN
from models.transcoders import Transcoder
torch.serialization.add_safe_globals([StackDataset])

class InteractiveCircuitVisualizer:
    """Interactive web-based visualizer for RNN circuit graphs"""
    
    def __init__(self, circuit_tracer, feature_analyzer, datasets, pruner=None):
        self.circuit_tracer = circuit_tracer
        self.feature_analyzer = feature_analyzer
        self.datasets = datasets
        self.pruner = pruner
        self.app = dash.Dash(__name__)
        
        self.current_edge_weights = None 
        self.current_edge_weights_normalized = None  # Add this line
        self.current_tokens = None
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _parse_node_info(self, node_name: str) -> Dict:
        """Parse node name to extract type, timestep, and feature info"""
        parts = node_name.split('_')
        
        if node_name.startswith('x_'):
            return {'type': 'input', 'timestep': int(parts[1]), 'dimension': int(parts[2])}
        elif node_name.startswith('f_z_'):
            return {'type': 'feature_update', 'timestep': int(parts[2]), 'feature_idx': int(parts[3])}
        elif node_name.startswith('f_r_'):
            return {'type': 'feature_reset', 'timestep': int(parts[2]), 'feature_idx': int(parts[3])}
        elif node_name.startswith('f_n_'):
            return {'type': 'feature_hidden', 'timestep': int(parts[2]), 'feature_idx': int(parts[3])}
        elif node_name == 'h_init':
            return {'type': 'initial', 'timestep': -1}
        elif node_name.startswith('e_'):
            return {'type': 'error', 'gate': parts[1], 'timestep': int(parts[2])}
        elif node_name.startswith('o_'):
            return {'type': 'output', 'timestep': int(parts[1]), 'dimension': int(parts[2])}
        else:
            return {'type': 'unknown', 'timestep': 0}
    
    def _compute_graph_layout(self, edge_weights: Dict[Tuple[str, str], float]) -> Dict[str, Tuple[float, float]]:
        """Compute hierarchical layout for graph visualization with nodes sorted by contribution"""
        G = nx.DiGraph()
        for (from_node, to_node), weight in edge_weights.items():
            G.add_edge(from_node, to_node, weight=abs(weight))
        
        node_info = {node: self._parse_node_info(node) for node in G.nodes()}
        
        # Calculate total incoming weight (contribution) for each node
        node_contributions = {}
        for node in G.nodes():
            total_incoming = 0
            for (from_node, to_node), weight in edge_weights.items():
                if to_node == node:
                    total_incoming += abs(weight)
            node_contributions[node] = total_incoming
        
        # Group nodes by timestep and type
        timesteps = {}
        for node, info in node_info.items():
            t = info['timestep']
            if t not in timesteps:
                timesteps[t] = {'input': [], 'error': [], 'feature_reset': [], 'feature_update': [], 'feature_hidden': [], 'initial': [], 'output': []}
            timesteps[t][info['type']].append(node)
        
        # Sort nodes within each type by contribution (highest contribution at bottom)
        for t in timesteps:
            for node_type in timesteps[t]:
                timesteps[t][node_type].sort(key=lambda node: node_contributions.get(node, 0), reverse=True)
        
        positions = {}
        timestep_width = 200
        type_spacing = {'input': 80, 'error': 24, 'feature_reset': 60, 'feature_update': 60, 'feature_hidden': 60, 'initial': 24, 'output': 80}
        
        for t, nodes_by_type in timesteps.items():
            x_base = t * timestep_width
            
            # Place output nodes at the top (lowest y-values)
            # Sorted by contribution - highest contribution nodes at bottom of their group
            for i, node in enumerate(nodes_by_type['output']):
                positions[node] = (x_base, -100 - i * type_spacing['output'])
            
            # Place input nodes at the bottom
            # Sorted by contribution - highest contribution nodes at bottom of their group
            for i, node in enumerate(nodes_by_type['input']):
                positions[node] = (x_base - 50, 400 + i * type_spacing['input'])

            for i, node in enumerate(nodes_by_type['error']):
                positions[node] = (x_base - 80, 120 + i * type_spacing['error'])
            
            # Place feature nodes in the middle
            # Sorted by contribution - highest contribution nodes at bottom of their group
            for i, node in enumerate(nodes_by_type['feature_reset']):
                positions[node] = (x_base - 40, 120 + i * type_spacing['feature_reset'])

            for i, node in enumerate(nodes_by_type['feature_update']):
                positions[node] = (x_base, 150 + i * type_spacing['feature_update'])
                
            for i, node in enumerate(nodes_by_type['feature_hidden']):
                positions[node] = (x_base + 50, 150 + i * type_spacing['feature_hidden'])

            for i, node in enumerate(nodes_by_type['initial']):
                positions[node] = (x_base, 150 + i * type_spacing['initial'])
        
        return positions
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node based on type"""
        color_map = {
            'input': '#4CAF50',
            'error': '#616161',
            'feature_reset': '#E57373',
            'feature_update': '#2196F3', 
            'feature_hidden': '#FF9800',
            'initial': '#8E7CC3',
            'output': '#F44336'
        }
        return color_map.get(node_type, '#757575')
    
    def _get_node_activation_magnitude(self, node_name: str, 
                                        active_features: Dict) -> Optional[float]:
        """Get activation magnitude for a given node"""
        node_info = self._parse_node_info(node_name)
        
        if node_info['type'] == 'feature_reset':
            timestep = node_info['timestep']
            feature_idx = node_info['feature_idx']
            for t, feat_idx, magnitude in active_features.get('reset', []):
                if t == timestep and feat_idx == feature_idx:
                    return float(magnitude)
        elif node_info['type'] == 'feature_update':
            timestep = node_info['timestep']
            feature_idx = node_info['feature_idx']
            
            # Find matching activation in active_features
            for t, feat_idx, magnitude in active_features.get('update', []):
                if t == timestep and feat_idx == feature_idx:
                    return float(magnitude)
                    
        elif node_info['type'] == 'feature_hidden':
            timestep = node_info['timestep']
            feature_idx = node_info['feature_idx']
            
            # Find matching activation in active_features
            for t, feat_idx, magnitude in active_features.get('hidden', []):
                if t == timestep and feat_idx == feature_idx:
                    return float(magnitude)
        
        return None
    
    @staticmethod
    def _format_pieces(pieces, top_k: int = 5) -> str:
        shown = ", ".join(f"{name} {v:+.3f}" for name, v in pieces[:top_k])
        more = f" (+{len(pieces) - top_k} more)" if len(pieces) > top_k else ""
        return shown + more

    def _read_gate_hover(self, node: str, top_k: int = 5) -> str:
        """Gate-view line for a candidate feature: its memory input
        W_n_h . (r_t * h_{t-1}) split by the reset-gate pieces at t."""
        acts = getattr(self, 'current_acts', None)
        if acts is None or not node.startswith('f_n_'):
            return ""
        pieces = self.circuit_tracer.read_gate_pieces(acts, node)
        total = sum(v for _, v in pieces)
        return f"<br><b>memory input {total:+.3f} via r_t</b>: {self._format_pieces(pieces, top_k)}"

    def _gate_ranking_text(self, top_k: int = 8) -> List[str]:
        """Top gate pieces by exact effect on the current view's target
        (probability-weighted logits, or the root of a rooted graph)."""
        acts = getattr(self, 'current_acts', None)
        if acts is None or self.current_edge_weights is None:
            return []
        targets = getattr(self, 'current_target_weights', None)
        root = next(iter(targets)) if targets else None
        if targets is None:
            names = [n.name for n in self.circuit_tracer._build_nodes(acts)]
            pruner = self.pruner or GraphPruner()
            weights = pruner.get_logit_weights(names, self.current_output_logits)
            targets = {n: float(w) for n, w in zip(names, weights) if w > 0}
        ranking = self.circuit_tracer.gate_feature_ranking(
            acts, targets, root=root, edges=self.current_edge_weights)
        features = [r for r in ranking if r[0].startswith(('f_z_', 'f_r_'))]
        other = [r for r in ranking if not r[0].startswith(('f_z_', 'f_r_'))]
        fmt = lambda rows: ", ".join(f"{name} ({where}) {score:+.4f}" for name, score, where in rows[:top_k])
        target = f"root {root}" if root else "weighted logits"
        return [f"Top gate features by effect on {target} (drop if removed; paste into Explain):",
                "  " + (fmt(features) or "none"),
                "Largest error / bias gate pieces: " + (fmt(other[:3]) or "none")]

    def _split_edge_text(self, edge_text: str, top_k: int = 5) -> List[str]:
        """Per-step gate splits of one folded edge, for the 'Split edge' box."""
        acts = getattr(self, 'current_acts', None)
        if acts is None:
            return ["Generate a circuit first."]
        parts = (edge_text or "").split()
        if len(parts) != 2:
            return ["Enter: source_node target_node (e.g. f_n_0_5 f_n_6_3)"]
        src, dst = parts
        weights = self.current_edge_weights or {}
        if (src, dst) not in weights:
            return [f"No edge {src} -> {dst} in the current graph."]
        lines = [f"{src} -> {dst}: weight {weights[(src, dst)]:+.4f}"]
        for step, gate, role in self.circuit_tracer.edge_gate_steps(src, dst):
            pieces = self.circuit_tracer.split_edge_by_gate(src, dst, acts, step)
            symbol = "z" if gate == "update" else "r"
            factor = f"1-{symbol}_{step}" if role == "carry" else f"{symbol}_{step}"
            lines.append(f"step {step} ({role}, {factor}): {self._format_pieces(pieces, top_k)}")
        if len(lines) == 1:
            lines.append("Ungated edge (input -> feature).")
        return lines

    def _create_circuit_graph(self, edge_weights: Dict[Tuple[str, str], float],
                display_edge_weights: Dict[Tuple[str, str], float],
                kept_nodes: Optional[Set[str]] = None,
                active_features: Optional[Dict] = None) -> go.Figure:
        """Create interactive circuit graph visualization with hover activation magnitudes
        
        Args:
            edge_weights: Edge weights used for graph structure and pruning
            display_edge_weights: Edge weights to display on the graph (may be different from edge_weights)
            kept_nodes: Nodes to keep after pruning
            active_features: Active feature information for hover display
        """
        if kept_nodes:
            filtered_edges = {
                (from_node, to_node): weight 
                for (from_node, to_node), weight in edge_weights.items()
                if from_node in kept_nodes and to_node in kept_nodes
            }
            filtered_display_edges = {
                (from_node, to_node): display_edge_weights.get((from_node, to_node), weight)
                for (from_node, to_node), weight in filtered_edges.items()
            }
        else:
            filtered_edges = edge_weights
            filtered_display_edges = display_edge_weights
        
        if not filtered_edges:
            fig = go.Figure()
            fig.update_layout(title="No edges to display")
            return fig
        
        positions = self._compute_graph_layout(filtered_edges)
        
        all_nodes = set()
        for from_node, to_node in filtered_edges.keys():
            all_nodes.add(from_node)
            all_nodes.add(to_node)
        
        node_info = {node: self._parse_node_info(node) for node in all_nodes}
        
        fig = go.Figure()
        
        # Create edge mappings for highlighting
        node_to_outgoing = {}  # node -> [target_nodes]
        node_to_incoming = {}  # node -> [source_nodes]
        edge_to_coords = {}    # (from, to) -> (x0, y0, x1, y1)
        edge_to_weight = {}    # (from, to) -> weight (display weight)
        
        for (from_node, to_node) in filtered_edges.keys():
            if from_node in positions and to_node in positions:
                x0, y0 = positions[from_node]
                x1, y1 = positions[to_node]
                edge_to_coords[(from_node, to_node)] = (x0, y0, x1, y1)
                # Use display weight for labels
                edge_to_weight[(from_node, to_node)] = filtered_display_edges.get((from_node, to_node), filtered_edges[(from_node, to_node)])
                
                if from_node not in node_to_outgoing:
                    node_to_outgoing[from_node] = []
                if to_node not in node_to_incoming:
                    node_to_incoming[to_node] = []
                    
                node_to_outgoing[from_node].append(to_node)
                node_to_incoming[to_node].append(from_node)
        
        # Add default edges (gray)
        default_edge_x = []
        default_edge_y = []
        for coords in edge_to_coords.values():
            x0, y0, x1, y1 = coords
            default_edge_x.extend([x0, x1, None])
            default_edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=default_edge_x, y=default_edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            name='default_edges'
        ))
        
        # Add individual edge traces for each possible edge (hidden by default)
        for (from_node, to_node), coords in edge_to_coords.items():
            x0, y0, x1, y1 = coords
            weight = edge_to_weight[(from_node, to_node)]  # Use display weight
            
            # Calculate midpoint for label placement
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            # Purple trace for outgoing edges
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=3, color='purple'),
                hoverinfo='none',
                mode='lines',
                showlegend=False,
                visible=False,
                name=f'outgoing_{from_node}_{to_node}'
            ))
            
            # Yellow trace for incoming edges  
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=3, color='gold'),
                hoverinfo='none',
                mode='lines',
                showlegend=False,
                visible=False,
                name=f'incoming_{from_node}_{to_node}'
            ))
            
            # Purple weight label for outgoing edges
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y],
                mode='text',
                text=[f'{weight:.3f}'],
                textfont=dict(size=10, color='black'),
                showlegend=False,
                visible=False,
                hoverinfo='none',
                name=f'outgoing_label_{from_node}_{to_node}'
            ))
            
            # Yellow weight label for incoming edges
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y],
                mode='text',
                text=[f'{weight:.3f}'],
                textfont=dict(size=10, color='darkgoldenrod'),
                showlegend=False,
                visible=False,
                hoverinfo='none',
                name=f'incoming_label_{from_node}_{to_node}'
            ))
        
        # ... (rest of the node creation code remains the same) ...
        
        # Add nodes by type (this part remains unchanged)
        node_types = ['input', 'error', 'initial', 'feature_reset', 'feature_update', 'feature_hidden', 'output']
        
        for node_type in node_types:
            nodes_of_type = [node for node, info in node_info.items() if info['type'] == node_type]
            
            if not nodes_of_type:
                continue
                
            node_x = []
            node_y = []
            node_text = []
            hover_text = []
            node_ids = []
            
            for node in nodes_of_type:
                if node not in positions:
                    continue
                    
                info = node_info[node]
                x, y = positions[node]
                node_x.append(x)
                node_y.append(y)
                
                if info['type'] == 'input':
                    text = f"x_{info['timestep']}_{info['dimension']}"
                    hover_info = f"Input Node<br>Timestep: {info['timestep']}<br>Dimension: {info['dimension']}"
                elif info['type'] in ['feature_reset', 'feature_update', 'feature_hidden']:
                    text = f"f_{info.get('feature_idx', 0)}"
                    
                    # Get activation magnitude if available
                    activation_mag = None
                    if active_features:
                        activation_mag = self._get_node_activation_magnitude(node, active_features)
                    
                    feature_label = {'feature_reset': 'Feature Reset', 'feature_update': 'Feature Update', 'feature_hidden': 'Feature Hidden'}[info['type']]
                    hover_info = f"{feature_label}<br>" \
                            f"Timestep: {info['timestep']}<br>" \
                            f"Feature: {info.get('feature_idx', 0)}"
                    
                    if activation_mag is not None:
                        hover_info += f"<br><b>Activation Magnitude: {activation_mag:.4f}</b>"
                    else:
                        hover_info += "<br>Activation Magnitude: N/A"
                    hover_info += self._read_gate_hover(node)

                elif info['type'] == 'initial':
                    text = "h_init"
                    hover_info = ("Hidden state entering the window<br>"
                                  "Source-only: learned initial state (RL) or zero (copy)")

                elif info['type'] == 'error':
                    text = f"e_{info['gate']}_{info['timestep']}"
                    hover_info = ("Frozen candidate reconstruction residual<br>"
                                  f"Timestep: {info['timestep']}<br>"
                                  "Source-only: unexplained by the transcoder")
                        
                elif info['type'] == 'output':
                    text = f"o_{info['timestep']}_{info['dimension']}"
                    hover_info = (f"Output Node<br>Timestep: {info['timestep']}<br>Dimension: {info['dimension']}<br>"
                                  "Incoming edges are effects on the centered logit")
                else:
                    text = node
                    hover_info = f"Unknown Node: {node}"
                
                node_text.append(text)
                hover_text.append(hover_info)
                node_ids.append(node)
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=self._get_node_color(node_type),
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color='white'),
                name=node_type.replace('_', ' ').title(),
                hovertext=hover_text,
                hoverinfo='text',
                customdata=node_ids
            ))
        
        # Store edge mappings in the figure for the clientside callback
        fig.update_layout(
            title="RNN Circuit Graph",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            # Store edge mapping data for clientside callback
            uirevision='constant',  # Prevents layout reset
            meta={
                'node_to_outgoing': node_to_outgoing,
                'node_to_incoming': node_to_incoming,
                'all_edges': list(edge_to_coords.keys()),
                'edge_weights': {f"{from_node}_{to_node}": weight 
                for (from_node, to_node), weight in edge_to_weight.items()}
            }
        )
        
        return fig

    def _add_clientside_callbacks(self):
        """Add clientside callbacks for edge highlighting"""
        
        # Clientside callback for hover highlighting
        self.app.clientside_callback(
            """
            function(hoverData, figure) {
                if (!figure || !figure.data) {
                    return window.dash_clientside.no_update;
                }
                
                // Clone the figure to avoid mutation
                let newFig = JSON.parse(JSON.stringify(figure));
                
                // Hide all edge highlight traces and labels
                for (let i = 0; i < newFig.data.length; i++) {
                    if (newFig.data[i].name && 
                        (newFig.data[i].name.startsWith('outgoing_') || 
                        newFig.data[i].name.startsWith('incoming_'))) {
                        newFig.data[i].visible = false;
                    }
                }
                
                // If no hover data, just return with hidden highlights
                if (!hoverData || !hoverData.points || hoverData.points.length === 0) {
                    return newFig;
                }
                
                let point = hoverData.points[0];
                if (!point.customdata) {
                    return newFig;
                }
                
                let hoveredNode = point.customdata;
                let nodeToOutgoing = figure.layout.meta ? figure.layout.meta.node_to_outgoing : {};
                let nodeToIncoming = figure.layout.meta ? figure.layout.meta.node_to_incoming : {};
                
                // Show outgoing edges (purple) and their labels
                if (nodeToOutgoing[hoveredNode]) {
                    for (let targetNode of nodeToOutgoing[hoveredNode]) {
                        let edgeTraceName = 'outgoing_' + hoveredNode + '_' + targetNode;
                        let labelTraceName = 'outgoing_label_' + hoveredNode + '_' + targetNode;
                        
                        for (let i = 0; i < newFig.data.length; i++) {
                            if (newFig.data[i].name === edgeTraceName || 
                                newFig.data[i].name === labelTraceName) {
                                newFig.data[i].visible = true;
                            }
                        }
                    }
                }
                
                // Show incoming edges (yellow) and their labels
                if (nodeToIncoming[hoveredNode]) {
                    for (let sourceNode of nodeToIncoming[hoveredNode]) {
                        let edgeTraceName = 'incoming_' + sourceNode + '_' + hoveredNode;
                        let labelTraceName = 'incoming_label_' + sourceNode + '_' + hoveredNode;
                        
                        for (let i = 0; i < newFig.data.length; i++) {
                            if (newFig.data[i].name === edgeTraceName || 
                                newFig.data[i].name === labelTraceName) {
                                newFig.data[i].visible = true;
                            }
                        }
                    }
                }
                
                return newFig;
            }
            """,
            Output('circuit-graph', 'figure', allow_duplicate=True),
            [Input('circuit-graph', 'hoverData')],
            [State('circuit-graph', 'figure')],
            prevent_initial_call=True
        )
    
    def _setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = html.Div([
            html.H1("RNN Circuit Visualizer", 
                style={'text-align': 'center', 'margin-bottom': '20px'}),
            
            # Input controls
            html.Div([
                html.Label("Dataset Index and Sequence Index:", style={'font-weight': 'bold'}),
                dcc.Input(
                    id='sequence-input',
                    type='text',
                    placeholder='e.g., 0 42',
                    value='0 0',
                    style={'width': '200px', 'margin': '5px 10px'}
                ),
                html.Button('Generate Circuit', id='generate-button', 
                        style={'margin': '5px', 'padding': '10px'})
            ], style={'margin-bottom': '20px', 'text-align': 'center'}),
            
            # New controls for edge normalization and thresholds
            html.Div([
                # Pruning must preserve cross-bank attribution scale.
                html.Div([
                    html.Label("Pruning uses raw attribution edges (normalization is display-only).", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Checklist(
                        id='normalize-toggle',
                        options=[{'label': 'Normalized', 'value': 'normalized', 'disabled': True}],
                        value=[],
                        style={'display': 'inline-block'}
                    )
                ], style={'margin-bottom': '10px', 'text-align': 'center'}),
                
                # Toggle for displayed edge weights (independent of pruning)
                html.Div([
                    html.Label("Display Edge Weights As:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Checklist(
                        id='display-normalize-toggle',
                        options=[{'label': 'Normalized', 'value': 'normalized'}],
                        value=[],
                        style={'display': 'inline-block'}
                    )
                ], style={'margin-bottom': '10px', 'text-align': 'center'}),
                
                # Threshold controls
                html.Div([
                    html.Label("Node Threshold:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Input(
                        id='node-threshold-input',
                        type='number',
                        placeholder='0.8',
                        value=0.8,
                        step=0.01,
                        min=0,
                        max=1,
                        style={'width': '100px', 'margin-right': '20px'}
                    ),
                    html.Label("Edge Threshold:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Input(
                        id='edge-threshold-input',
                        type='number',
                        placeholder='0.98',
                        value=0.98,
                        step=0.01,
                        min=0,
                        max=1,
                        style={'width': '100px'}
                    )
                ], style={'margin-bottom': '10px', 'text-align': 'center'})
            ], style={'margin-bottom': '20px', 'padding': '10px', 'background-color': '#f8f9fa', 'border-radius': '5px'}),
            
            # Gate view: split an edge by the gate features of each step it
            # spans, or switch to a graph rooted at a gate feature.
            html.Div([
                html.Div([
                    html.Label("Split edge by gate:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Input(id='split-edge-input', type='text', placeholder='f_n_0_5 f_n_6_3',
                              style={'width': '220px', 'margin-right': '5px'}),
                    html.Button('Split', id='split-edge-button', style={'margin-right': '30px'}),
                    html.Label("Explain gate feature:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Input(id='gate-feature-input', type='text', placeholder='f_z_3_12',
                              style={'width': '120px', 'margin-right': '5px'}),
                    html.Button('Explain', id='explain-gate-button', style={'margin-right': '5px'}),
                    html.Button('Main graph', id='main-graph-button'),
                ], style={'text-align': 'center'}),
                html.Div(id='split-edge-output', style={'margin-top': '10px', 'font-family': 'monospace',
                                                        'font-size': '12px', 'text-align': 'left'}),
                html.Div(id='gate-ranking', style={'margin-top': '10px', 'font-family': 'monospace',
                                                   'font-size': '12px', 'text-align': 'left'}),
            ], style={'margin-bottom': '20px', 'padding': '10px', 'background-color': '#f8f9fa', 'border-radius': '5px'}),

            # Status display
            html.Div(id='graph-stats', style={'margin-bottom': '10px', 'padding': '10px',
                                            'background-color': '#f8f9fa', 'border-radius': '5px',
                                            'text-align': 'center'}),
            
            # Graph display
            dcc.Graph(id='circuit-graph', style={'height': '700px'})
        ])
    
    def _setup_callbacks(self):
        @self.app.callback(
            Output('split-edge-output', 'children'),
            [Input('split-edge-button', 'n_clicks')],
            [State('split-edge-input', 'value')],
            prevent_initial_call=True
        )
        def split_edge(n_clicks, edge_text):
            return [html.Div(line) for line in self._split_edge_text(edge_text)]

        @self.app.callback(
            Output('gate-ranking', 'children'),
            [Input('graph-stats', 'children')],
            prevent_initial_call=True
        )
        def rank_gates(stats):
            # Refreshes whenever the graph is redrawn (generate, view switch, toggles).
            return [html.Div(line) for line in self._gate_ranking_text()]

        @self.app.callback(
            [Output('circuit-graph', 'figure'),
            Output('graph-stats', 'children')],
            [Input('generate-button', 'n_clicks'),
            Input('display-normalize-toggle', 'value'),  # Add display toggle as input
            Input('normalize-toggle', 'value'),  # Add pruning toggle as input
            Input('explain-gate-button', 'n_clicks'),
            Input('main-graph-button', 'n_clicks')],
            [State('sequence-input', 'value'),
            State('node-threshold-input', 'value'),
            State('edge-threshold-input', 'value'),
            State('gate-feature-input', 'value'),
            State('circuit-graph', 'figure')]  # Keep current figure state
        )
        def generate_and_display_circuit(n_clicks, display_normalize_toggle, normalize_toggle,
                                        explain_clicks, main_clicks,
                                        sequence_text, node_threshold, edge_threshold, gate_feature,
                                        current_figure):
            """Generate and display circuit graph"""
            ctx = dash.callback_context
            triggered = [t['prop_id'] for t in ctx.triggered] if ctx.triggered else []

            # View switches: a graph rooted at a gate feature (its target
            # replaces the logits in pruning), or back to the main graph.
            if 'explain-gate-button.n_clicks' in triggered:
                if getattr(self, 'current_acts', None) is None:
                    return current_figure or go.Figure(), "Generate a circuit first."
                try:
                    edges, normalized, targets = self.circuit_tracer.gate_feature_graph(
                        self.current_acts, (gate_feature or "").strip())
                except ValueError as e:
                    return current_figure or go.Figure(), f"Error: {e}"
                self.current_edge_weights, self.current_edge_weights_normalized = edges, normalized
                self.current_target_weights = targets
            elif 'main-graph-button.n_clicks' in triggered:
                if getattr(self, 'main_edge_weights', None) is None:
                    return current_figure or go.Figure(), "Generate a circuit first."
                self.current_edge_weights, self.current_edge_weights_normalized = self.main_edge_weights
                self.current_target_weights = None
            view = (f"rooted at {next(iter(self.current_target_weights))}"
                    if getattr(self, 'current_target_weights', None) else "main graph")

            # Check if this is just a display toggle change (or a view switch)
            display_triggered = ctx.triggered and any(
                prop_id in ['display-normalize-toggle.value', 'normalize-toggle.value',
                            'explain-gate-button.n_clicks', 'main-graph-button.n_clicks']
                for prop_id in triggered
            )
            
            # Use cached data if available for display/normalization toggles
            if display_triggered and self.current_edge_weights is not None and self.current_edge_weights_normalized is not None:
                use_normalized_for_display = 'normalized' in display_normalize_toggle
                
                # Group-normalized weights are a display aid only. Using them
                # for pruning would arbitrarily change relative importance
                # across gate banks.
                selected_edge_weights = self.current_edge_weights
                
                # Choose which edge weights to display
                display_edge_weights = self.current_edge_weights_normalized if use_normalized_for_display else self.current_edge_weights
                
                # Get cached data for re-pruning if needed
                if hasattr(self, 'current_sequence_tensor') and hasattr(self, 'current_active_features'):
                    if self.pruner:
                        # Update pruner thresholds if provided
                        if node_threshold is not None:
                            self.pruner.node_threshold = node_threshold
                        if edge_threshold is not None:
                            self.pruner.edge_threshold = edge_threshold
                        
                        pruned_edges, kept_nodes = self.pruner.prune_graph(
                            selected_edge_weights, self.current_output_logits,
                            target_weights=getattr(self, 'current_target_weights', None))
                        fig = self._create_circuit_graph(pruned_edges, display_edge_weights, kept_nodes, self.current_active_features)

                        display_type = "normalized" if use_normalized_for_display else "raw"
                        stats = f"Circuit for '{' '.join(self.current_tokens)}', {view} (pruning: raw attribution, display: {display_type}): {len(kept_nodes)} nodes, {len(pruned_edges)} edges"
                    else:
                        fig = self._create_circuit_graph(selected_edge_weights, display_edge_weights, None, self.current_active_features)
                        all_nodes = set(sum(selected_edge_weights.keys(), ()))
                        display_type = "normalized" if use_normalized_for_display else "raw"
                        stats = f"Circuit for '{' '.join(self.current_tokens)}', {view} (pruning: raw attribution, display: {display_type}): {len(all_nodes)} nodes, {len(selected_edge_weights)} edges"
                    
                    return fig, stats
            
            # If no cached data or this is a generate button click
            if not n_clicks or not sequence_text:
                return current_figure or go.Figure(), "Enter dataset and sequence indices, then click 'Generate Circuit'"
            
            try:
                options = sequence_text.strip().split()
                if len(options) < 2:
                    return current_figure or go.Figure(), "Enter format: dataset_index sequence_index"

                dataset_idx, sequence_index = map(int, options)
                sequence_tensor = self.datasets[dataset_idx][sequence_index]
                
                if hasattr(self.feature_analyzer, "cur_type"):
                    self.feature_analyzer.cur_type = ["commonp", "common_p", "uncommonp", "uncommon_p"][dataset_idx]
                
                tokens = self.feature_analyzer.convert_sequence_to_text(
                    sequence_tensor["inputs"], sequence_tensor["outputs"]
                )
                
                # Recompute live activations from the checkpoint currently
                # loaded in the tracer. Cached analyses are for feature
                # browsing only and must not decide circuit membership.
                acts = self.circuit_tracer.run_forward_pass(sequence_tensor)
                active_features = self.circuit_tracer.get_active_features(sequence_tensor, acts=acts)

                print(f"Building circuit with {sum(len(v) for v in active_features.values())} active features")

                # Build circuit graph - get both normalized and raw edge weights
                edge_weights, edge_weights_normalized = self.circuit_tracer.build_circuit_graph(
                    sequence_tensor, active_features, acts=acts)

                # Cache both edge weight types and other data
                self.current_edge_weights = edge_weights
                self.current_edge_weights_normalized = edge_weights_normalized
                self.main_edge_weights = (edge_weights, edge_weights_normalized)
                self.current_target_weights = None
                self.current_sequence_tensor = sequence_tensor
                self.current_active_features = active_features
                self.current_tokens = tokens
                self.current_acts = acts
                self.current_output_logits = acts["logits"].detach().cpu()
                
                # Choose which edge weights to use for pruning
                use_normalized_for_display = 'normalized' in display_normalize_toggle
                
                selected_edge_weights = edge_weights
                display_edge_weights = edge_weights_normalized if use_normalized_for_display else edge_weights
                
                # Auto-prune if pruner exists
                if self.pruner:
                    # Update pruner thresholds if provided
                    if node_threshold is not None:
                        self.pruner.node_threshold = node_threshold
                    if edge_threshold is not None:
                        self.pruner.edge_threshold = edge_threshold
                    
                    print(f"Auto-pruning {len(selected_edge_weights)} edges with node_threshold={self.pruner.node_threshold}, edge_threshold={self.pruner.edge_threshold}")
                    pruned_edges, kept_nodes = self.pruner.prune_graph(selected_edge_weights, self.current_output_logits)
                    print(f"After pruning: {len(pruned_edges)} edges, {len(kept_nodes)} nodes")
                    
                    fig = self._create_circuit_graph(pruned_edges, display_edge_weights, kept_nodes, active_features)
                    display_type = "normalized" if use_normalized_for_display else "raw"
                    stats = f"Circuit for '{' '.join(tokens)}', main graph (pruning: raw attribution, display: {display_type}): {len(kept_nodes)} nodes, {len(pruned_edges)} edges (pruned from {len(selected_edge_weights)}) | Thresholds: node={self.pruner.node_threshold}, edge={self.pruner.edge_threshold}"
                else:
                    fig = self._create_circuit_graph(selected_edge_weights, display_edge_weights, None, active_features)
                    all_nodes = set(sum(selected_edge_weights.keys(), ()))
                    display_type = "normalized" if use_normalized_for_display else "raw"
                    stats = f"Circuit for '{' '.join(tokens)}', main graph (pruning: raw attribution, display: {display_type}): {len(all_nodes)} nodes, {len(selected_edge_weights)} edges (no pruning)"
                
                return fig, stats
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                return current_figure or go.Figure(), f"Error: {str(e)}"
        
        self._add_clientside_callbacks()
            
    def run(self, host='0.0.0.0', port=8051, debug=True):
        """Run the Dash app"""
        print(f"Starting circuit visualizer at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

def launch_circuit_visualizer(circuit_tracer, feature_analyzer, datasets, pruner=None):
    """Launch the interactive circuit visualizer"""
    visualizer = InteractiveCircuitVisualizer(circuit_tracer, feature_analyzer, datasets, pruner)
    visualizer.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--feature_dict_path")
    parser.add_argument("--rnn_path")
    parser.add_argument("--update_transcoder_path")
    parser.add_argument("--hidden_transcoder_path")
    parser.add_argument("--reset_transcoder_path")
    parser.add_argument("--n_feats_hidden", type=int)
    parser.add_argument("--n_feats_update", type=int)
    parser.add_argument("--n_feats_reset", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--dataset_paths", nargs="+")
    parser.add_argument("--allow_legacy_feature_cache", action="store_true",
                        help="Allow a cache without provenance metadata (unsafe; circuit membership is still recomputed live)")

    args = parser.parse_args()
    
    datasets = []
    for dataset in args.dataset_paths:
        datasets.append(torch.load(dataset, map_location=torch.device("cpu")))
        
    if args.rl:
        rnn_model = RNN(input_size=8, hidden_size=48, out_size=4, 
                    use_gru=True, num_layers=1, learn_init=True)
        update_transcoder = Transcoder(input_size=56, out_size=48, n_feats=args.n_feats_update)
        hidden_transcoder = Transcoder(input_size=56, out_size=48, n_feats=args.n_feats_hidden)
        reset_transcoder = Transcoder(input_size=56, out_size=48, n_feats=args.n_feats_reset) if args.reset_transcoder_path else None
        analyzer = RLFeatureActivationAnalyzer
    else:
        rnn_model = RNN(input_size=31, hidden_size=128, out_size=30, use_gru=True, num_layers=1)
        update_transcoder = Transcoder(input_size=159, out_size=128, n_feats=args.n_feats_update)
        hidden_transcoder = Transcoder(input_size=159, out_size=128, n_feats=args.n_feats_hidden)
        reset_transcoder = Transcoder(input_size=159, out_size=128, n_feats=args.n_feats_reset) if args.reset_transcoder_path else None
        analyzer = CopyFeatureActivationAnalyzer
    
    rnn_model.load_state_dict(torch.load(args.rnn_path))
    update_transcoder.load_state_dict(torch.load(args.update_transcoder_path)["transcoder"])
    hidden_transcoder.load_state_dict(torch.load(args.hidden_transcoder_path)["transcoder"])
    if reset_transcoder:
        reset_transcoder.load_state_dict(torch.load(args.reset_transcoder_path)["transcoder"])
    
    metadata_path = args.feature_dict_path.replace("_features.p", "_metadata.p")
    if not os.path.exists(metadata_path):
        if not args.allow_legacy_feature_cache:
            raise ValueError(
                "Feature cache has no provenance metadata. Re-run circuit.copy_find_features "
                "or pass --allow_legacy_feature_cache explicitly."
            )
    else:
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        expected_fingerprints = {
            "rnn": checkpoint_fingerprint(args.rnn_path),
            "update": checkpoint_fingerprint(args.update_transcoder_path),
            "hidden": checkpoint_fingerprint(args.hidden_transcoder_path),
            "reset": checkpoint_fingerprint(args.reset_transcoder_path) if args.reset_transcoder_path else None,
        }
        if metadata.get("checkpoint_fingerprints") != expected_fingerprints:
            raise ValueError("Feature cache was generated with different checkpoint(s); refusing to mix models.")

    with open(args.feature_dict_path, "rb") as f:
        analysis_dict = pickle.load(f)
    with open(args.feature_dict_path.replace("features.p", "sequences.p"), "rb") as f:
        analysis_dict_sequences = pickle.load(f)

    feature_analyzer = analyzer(rnn_model, update_transcoder, hidden_transcoder, reset_transcoder=reset_transcoder)
    pruner = GraphPruner()
    feature_analyzer.feature_activations = analysis_dict
    feature_analyzer.sequence_activations = analysis_dict_sequences

    # RL: the readout is 3 policy logits + 1 value unit, and the agent acts at
    # every step. Copy: all readout units are logits, emitted in the second half.
    tracer_kwargs = dict(output_dims=3, outputs_at="all") if args.rl else {}
    circuit_tracer = CircuitTracer(rnn_model, update_transcoder, hidden_transcoder, reset_transcoder=reset_transcoder,
                                   device="cpu", **tracer_kwargs)
    
    launch_circuit_visualizer(circuit_tracer, feature_analyzer, datasets, pruner)
