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
            
            # Place output nodes at the top (lowest y-values)
            for i, node in enumerate(nodes_by_type['output']):
                positions[node] = (x_base, -100 - i * type_spacing['output'])
            
            # Place input nodes at the bottom
            for i, node in enumerate(nodes_by_type['input']):
                positions[node] = (x_base - 50, 400 + i * type_spacing['input'])
            
            # Place feature nodes in the middle
            for i, node in enumerate(nodes_by_type['feature_update']):
                positions[node] = (x_base, 150 + i * type_spacing['feature_update'])
                
            for i, node in enumerate(nodes_by_type['feature_hidden']):
                positions[node] = (x_base + 50, 150 + i * type_spacing['feature_hidden'])
        
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
        
        # Create edge mappings for highlighting
        node_to_outgoing = {}  # node -> [target_nodes]
        node_to_incoming = {}  # node -> [source_nodes]
        edge_to_coords = {}    # (from, to) -> (x0, y0, x1, y1)
        edge_to_weight = {}    # (from, to) -> weight
        
        for (from_node, to_node), weight in filtered_edges.items():
            if from_node in positions and to_node in positions:
                x0, y0 = positions[from_node]
                x1, y1 = positions[to_node]
                edge_to_coords[(from_node, to_node)] = (x0, y0, x1, y1)
                edge_to_weight[(from_node, to_node)] = weight
                
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
            weight = edge_to_weight[(from_node, to_node)]
            
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
        
        # Add nodes by type
        node_types = ['input', 'feature_update', 'feature_hidden', 'output']
        
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
                for (from_node, to_node), weight in filtered_edges.items()}
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
                # Toggle for normalized edges
                html.Div([
                    html.Label("Use Normalized Edge Weights:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Checklist(
                        id='normalize-toggle',
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
                        placeholder='0.98',
                        value=0.98,
                        step=0.01,
                        min=0,
                        max=1,
                        style={'width': '100px', 'margin-right': '20px'}
                    ),
                    html.Label("Edge Threshold:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                    dcc.Input(
                        id='edge-threshold-input',
                        type='number',
                        placeholder='0.99',
                        value=0.99,
                        step=0.01,
                        min=0,
                        max=1,
                        style={'width': '100px'}
                    )
                ], style={'margin-bottom': '10px', 'text-align': 'center'})
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
            [Output('circuit-graph', 'figure'),
            Output('graph-stats', 'children')],
            [Input('generate-button', 'n_clicks')],
            [State('sequence-input', 'value'),
            State('normalize-toggle', 'value'),
            State('node-threshold-input', 'value'),
            State('edge-threshold-input', 'value')]
        )
        def generate_and_display_circuit(n_clicks, sequence_text, normalize_toggle, node_threshold, edge_threshold):
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
                edge_weights, edge_weights_normalized = self.circuit_tracer.build_circuit_graph(sequence_tensor, active_features)
                
                # Choose which edge weights to use based on toggle
                use_normalized = 'normalized' in normalize_toggle
                selected_edge_weights = edge_weights_normalized if use_normalized else edge_weights
                
                # Auto-prune if pruner exists
                if self.pruner:
                    # Update pruner thresholds if provided
                    if node_threshold is not None:
                        self.pruner.node_threshold = node_threshold
                    if edge_threshold is not None:
                        self.pruner.edge_threshold = edge_threshold
                    
                    print(f"Auto-pruning {len(selected_edge_weights)} edges with node_threshold={self.pruner.node_threshold}, edge_threshold={self.pruner.edge_threshold}")
                    pruned_edges, kept_nodes = self.pruner.prune_graph(selected_edge_weights, sequence_tensor["outputs"])
                    print(f"After pruning: {len(pruned_edges)} edges, {len(kept_nodes)} nodes")
                    
                    import pickle
                    with open("sequence_example.p", "wb") as f:
                        pickle.dump(sequence_tensor, f)
                    with open("sequence_weights_example.p", "wb") as f:
                        pickle.dump(selected_edge_weights, f)
                    with open("active_features.p", "wb") as f:
                        pickle.dump(active_features, f)
                    
                    # Pass active_features to the graph creation function
                    fig = self._create_circuit_graph(pruned_edges, kept_nodes, active_features)
                    edge_type = "normalized" if use_normalized else "raw"
                    stats = f"Circuit for '{' '.join(tokens)}' ({edge_type} edges): {len(kept_nodes)} nodes, {len(pruned_edges)} edges (pruned from {len(selected_edge_weights)}) | Thresholds: node={self.pruner.node_threshold}, edge={self.pruner.edge_threshold}"
                else:
                    # Pass active_features to the graph creation function
                    fig = self._create_circuit_graph(selected_edge_weights, None, active_features)
                    all_nodes = set(sum(selected_edge_weights.keys(), ()))
                    edge_type = "normalized" if use_normalized else "raw"
                    stats = f"Circuit for '{' '.join(tokens)}' ({edge_type} edges): {len(all_nodes)} nodes, {len(selected_edge_weights)} edges (no pruning)"
                
                return fig, stats
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                return go.Figure(), f"Error: {str(e)}"
        
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