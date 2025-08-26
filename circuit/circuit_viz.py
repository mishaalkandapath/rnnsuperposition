from typing import Dict, List, Tuple, Optional, Set
import pickle

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import networkx as nx

import json
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import StackDataset

from circuit.edge_att import CircuitTracer
from circuit.copy_find_features import CopyFeatureActivationAnalyzer
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
        elif node_name.startswith('f_n_'):
            return {'type': 'feature_hidden', 'timestep': int(parts[2]), 'feature_idx': int(parts[3])}
        elif node_name.startswith('o_'):
            return {'type': 'output', 'timestep': int(parts[1]), 'dimension': int(parts[2])}
        else:
            return {'type': 'unknown', 'timestep': 0}
    
    def _compute_graph_layout(self, edge_weights: Dict[Tuple[str, str], float]) -> Dict[str, Tuple[float, float]]:
        """Compute hierarchical layout for graph visualization"""
        G = nx.DiGraph()
        for (from_node, to_node), weight in edge_weights.items():
            G.add_edge(from_node, to_node, weight=abs(weight))
        
        node_info = {node: self._parse_node_info(node) for node in G.nodes()}
        
        # Group nodes by timestep and type
        timesteps = {}
        for node, info in node_info.items():
            t = info['timestep']
            if t not in timesteps:
                timesteps[t] = {'input': [], 'feature_update': [], 'feature_hidden': [], 'output': []}
            timesteps[t][info['type']].append(node)
        
        positions = {}
        timestep_width = 200
        type_spacing = {'input': 80, 'feature_update': 60, 'feature_hidden': 60, 'output': 80}
        
        for t, nodes_by_type in timesteps.items():
            x_base = t * timestep_width
            
            for i, node in enumerate(nodes_by_type['input']):
                positions[node] = (x_base - 50, 50 + i * type_spacing['input'])
            
            for i, node in enumerate(nodes_by_type['feature_update']):
                positions[node] = (x_base, 200 + i * type_spacing['feature_update'])
                
            for i, node in enumerate(nodes_by_type['feature_hidden']):
                positions[node] = (x_base + 50, 200 + i * type_spacing['feature_hidden'])
            
            for i, node in enumerate(nodes_by_type['output']):
                positions[node] = (x_base, 500 + i * type_spacing['output'])
        
        return positions
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node based on type"""
        color_map = {
            'input': '#4CAF50',
            'feature_update': '#2196F3', 
            'feature_hidden': '#FF9800',
            'output': '#F44336'
        }
        return color_map.get(node_type, '#757575')
    
    def _get_node_activation_magnitude(self, node_name: str, 
                                        active_features: Dict) -> Optional[float]:
        """Get activation magnitude for a given node"""
        node_info = self._parse_node_info(node_name)
        
        if node_info['type'] == 'feature_update':
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
    
    def _create_circuit_graph(self, edge_weights: Dict[Tuple[str, str], float], 
                        kept_nodes: Optional[Set[str]] = None,
                        active_features: Optional[Dict] = None) -> go.Figure:
        """Create interactive circuit graph visualization with hover activation magnitudes"""
        if kept_nodes:
            filtered_edges = {
                (from_node, to_node): weight 
                for (from_node, to_node), weight in edge_weights.items()
                if from_node in kept_nodes and to_node in kept_nodes
            }
        else:
            filtered_edges = edge_weights
        
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
        
        # Add edges
        edge_x = []
        edge_y = []
        
        for (from_node, to_node), weight in filtered_edges.items():
            if from_node in positions and to_node in positions:
                x0, y0 = positions[from_node]
                x1, y1 = positions[to_node]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes by type
        node_types = ['input', 'feature_update', 'feature_hidden', 'output']
        
        for node_type in node_types:
            nodes_of_type = [node for node, info in node_info.items() if info['type'] == node_type]
            
            if not nodes_of_type:
                continue
                
            node_x = [positions[node][0] for node in nodes_of_type if node in positions]
            node_y = [positions[node][1] for node in nodes_of_type if node in positions]
            node_text = []
            hover_text = []
            
            for node in nodes_of_type:
                if node not in positions:
                    continue
                    
                info = node_info[node]
                
                if info['type'] == 'input':
                    text = f"x_{info['timestep']}_{info['dimension']}"
                    hover_info = f"Input Node<br>Timestep: {info['timestep']}<br>Dimension: {info['dimension']}"
                elif info['type'] in ['feature_update', 'feature_hidden']:
                    text = f"f_{info.get('feature_idx', 0)}"
                    
                    # Get activation magnitude if available
                    activation_mag = None
                    if active_features:
                        activation_mag = self._get_node_activation_magnitude(node, active_features)
                    
                    hover_info = f"{'Feature Update' if info['type'] == 'feature_update' else 'Feature Hidden'}<br>" \
                            f"Timestep: {info['timestep']}<br>" \
                            f"Feature: {info.get('feature_idx', 0)}"
                    
                    if activation_mag is not None:
                        hover_info += f"<br><b>Activation Magnitude: {activation_mag:.4f}</b>"
                    else:
                        hover_info += "<br>Activation Magnitude: N/A"
                        
                elif info['type'] == 'output':
                    text = f"o_{info['timestep']}_{info['dimension']}"
                    hover_info = f"Output Node<br>Timestep: {info['timestep']}<br>Dimension: {info['dimension']}"
                else:
                    text = node
                    hover_info = f"Unknown Node: {node}"
                
                node_text.append(text)
                hover_text.append(hover_info)
            
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
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="RNN Circuit Graph",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
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
            
            # Status display
            html.Div(id='graph-stats', style={'margin-bottom': '10px', 'padding': '10px', 
                                             'background-color': '#f8f9fa', 'border-radius': '5px',
                                             'text-align': 'center'}),
            
            # Graph display
            dcc.Graph(id='circuit-graph', style={'height': '700px'})
        ])
    
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('circuit-graph', 'figure'),
            Output('graph-stats', 'children')],
            [Input('generate-button', 'n_clicks')],
            [State('sequence-input', 'value')]
        )
        def generate_and_display_circuit(n_clicks, sequence_text):
            """Generate and display circuit graph"""
            if not n_clicks or not sequence_text:
                return go.Figure(), "Enter dataset and sequence indices, then click 'Generate Circuit'"
            
            try:
                options = sequence_text.strip().split()
                if len(options) < 2:
                    return go.Figure(), "Enter format: dataset_index sequence_index"

                dataset_idx, sequence_index = map(int, options)
                sequence_tensor = self.datasets[dataset_idx][sequence_index]
                
                if hasattr(self.feature_analyzer, "cur_type"):
                    self.feature_analyzer.cur_type = ["commonp", "common_p", "uncommonp", "uncommon_p"][dataset_idx]
                
                tokens = self.feature_analyzer.convert_sequence_to_text(
                    sequence_tensor["inputs"], sequence_tensor["outputs"]
                )
                
                # Get active features
                data_dict = self.feature_analyzer.sequence_activations
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
                
                print(f"Building circuit with {sum(len(v) for v in active_features.values())} active features")
                
                # Build circuit graph
                edge_weights = self.circuit_tracer.build_circuit_graph(sequence_tensor, active_features)
                
                # Auto-prune if pruner exists
                if self.pruner:
                    print(f"Auto-pruning {len(edge_weights)} edges..")
                    pruned_edges, kept_nodes = self.pruner.prune_graph(edge_weights, sequence_tensor["outputs"])
                    print(f"After pruning: {len(pruned_edges)} edges, {len(kept_nodes)} nodes")
                    import pickle
                    with open("sequence_example.p", "wb") as f:
                        pickle.dump(sequence_tensor, f)
                    with open("sequence_weights_example.p", "wb") as f:
                        pickle.dump(edge_weights, f)
                    
                    # Pass active_features to the graph creation function
                    fig = self._create_circuit_graph(pruned_edges, kept_nodes, active_features)
                    stats = f"Circuit for '{' '.join(tokens)}': {len(kept_nodes)} nodes, {len(pruned_edges)} edges (pruned from {len(edge_weights)})"
                else:
                    # Pass active_features to the graph creation function
                    fig = self._create_circuit_graph(edge_weights, None, active_features)
                    all_nodes = set(sum(edge_weights.keys(), ()))
                    stats = f"Circuit for '{' '.join(tokens)}': {len(all_nodes)} nodes, {len(edge_weights)} edges (no pruning)"
                
                return fig, stats
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                return go.Figure(), f"Error: {str(e)}"
            
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
    parser.add_argument("--n_feats_hidden", type=int)
    parser.add_argument("--n_feats_update", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--dataset_paths", nargs="+")

    args = parser.parse_args()
    
    datasets = []
    for dataset in args.dataset_paths:
        datasets.append(torch.load(dataset, map_location=torch.device("cpu")))
        
    if args.rl:
        rnn_model = RNN(input_size=8, hidden_size=48, out_size=4, 
                    use_gru=True, num_layers=1, learn_init=True)
        update_transcoder = Transcoder(input_size=56, out_size=48, n_feats=args.n_feats_update)
        hidden_transcoder = Transcoder(input_size=56, out_size=48, n_feats=args.n_feats_hidden)
        analyzer = RLFeatureActivationAnalyzer
    else:
        rnn_model = RNN(input_size=31, hidden_size=128, out_size=30, use_gru=True, num_layers=1)
        update_transcoder = Transcoder(input_size=159, out_size=128, n_feats=args.n_feats_update)
        hidden_transcoder = Transcoder(input_size=159, out_size=128, n_feats=args.n_feats_hidden)
        analyzer = CopyFeatureActivationAnalyzer
    
    rnn_model.load_state_dict(torch.load(args.rnn_path))
    update_transcoder.load_state_dict(torch.load(args.update_transcoder_path)["transcoder"])
    hidden_transcoder.load_state_dict(torch.load(args.hidden_transcoder_path)["transcoder"])
    
    with open(args.feature_dict_path, "rb") as f:
        analysis_dict = pickle.load(f)
    with open(args.feature_dict_path.replace("features.p", "sequences.p"), "rb") as f:
        analysis_dict_sequences = pickle.load(f)

    feature_analyzer = analyzer(rnn_model, update_transcoder, hidden_transcoder)
    pruner = GraphPruner()
    feature_analyzer.feature_activations = analysis_dict
    feature_analyzer.sequence_activations = analysis_dict_sequences

    circuit_tracer = CircuitTracer(rnn_model, update_transcoder, hidden_transcoder, device="cpu")
    
    launch_circuit_visualizer(circuit_tracer, feature_analyzer, datasets, pruner)