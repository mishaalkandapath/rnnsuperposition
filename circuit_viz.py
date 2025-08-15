import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
import colorsys
import json
import torch 

class InteractiveCircuitVisualizer:
    """Interactive web-based visualizer for RNN circuit graphs with feature activation examples"""
    
    def __init__(self, 
                 circuit_tracer,
                 feature_analyzer,
                 pruner=None):
        """
        Args:
            circuit_tracer: CircuitTracer instance
            feature_analyzer: FeatureActivationAnalyzer instance with collected data
            pruner: Optional GraphPruner instance
        """
        self.circuit_tracer = circuit_tracer
        self.feature_analyzer = feature_analyzer
        self.pruner = pruner
        self.app = dash.Dash(__name__)
        
        # Store current graph data
        self.current_graph = None
        self.current_sequence = None
        self.layout_positions = {}
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _parse_node_info(self, node_name: str) -> Dict:
        """Parse node name to extract type, timestep, and feature info"""
        parts = node_name.split('_')
        
        if node_name.startswith('x_'):
            # Input node: x_t_dim
            return {
                'type': 'input',
                'timestep': int(parts[1]) if len(parts) > 1 else 0,
                'dimension': int(parts[2]) if len(parts) > 2 else 0,
                'feature_idx': None
            }
        elif node_name.startswith('f_z_'):
            # Update gate feature: f_z_t_idx
            return {
                'type': 'feature_update',
                'timestep': int(parts[2]) if len(parts) > 2 else 0,
                'dimension': None,
                'feature_idx': int(parts[3]) if len(parts) > 3 else 0
            }
        elif node_name.startswith('f_n_'):
            # Hidden context feature: f_n_t_idx
            return {
                'type': 'feature_hidden',
                'timestep': int(parts[2]) if len(parts) > 2 else 0,
                'dimension': None,
                'feature_idx': int(parts[3]) if len(parts) > 3 else 0
            }
        elif node_name.startswith('o_'):
            # Output node: o_t_dim
            return {
                'type': 'output',
                'timestep': int(parts[1]) if len(parts) > 1 else 0,
                'dimension': int(parts[2]) if len(parts) > 2 else 0,
                'feature_idx': None
            }
        elif node_name.startswith('e_'):
            # Error node: e_z_t or e_n_t
            return {
                'type': 'error',
                'timestep': int(parts[2]) if len(parts) > 2 else 0,
                'dimension': None,
                'feature_idx': None
            }
        else:
            return {
                'type': 'unknown',
                'timestep': 0,
                'dimension': None,
                'feature_idx': None
            }
    
    def _compute_graph_layout(self, edge_weights: Dict[Tuple[str, str], float]) -> Dict[str, Tuple[float, float]]:
        """Compute hierarchical layout for graph visualization"""
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add edges
        for (from_node, to_node), weight in edge_weights.items():
            G.add_edge(from_node, to_node, weight=abs(weight))
        
        # Parse node information for layout
        node_info = {}
        for node in G.nodes():
            node_info[node] = self._parse_node_info(node)
        
        # Group nodes by timestep and type
        timesteps = {}
        for node, info in node_info.items():
            t = info['timestep']
            if t not in timesteps:
                timesteps[t] = {'input': [], 'feature_update': [], 'feature_hidden': [], 'output': [], 'error': []}
            timesteps[t][info['type']].append(node)
        
        positions = {}
        
        # Layout parameters
        timestep_width = 200
        type_spacing = {'input': 80, 'feature_update': 60, 'feature_hidden': 60, 'output': 80, 'error': 100}
        
        for t, nodes_by_type in timesteps.items():
            x_base = t * timestep_width
            
            # Position input nodes at top
            for i, node in enumerate(nodes_by_type['input']):
                positions[node] = (x_base - 50, 50 + i * type_spacing['input'])
            
            # Position feature nodes in middle
            for i, node in enumerate(nodes_by_type['feature_update']):
                positions[node] = (x_base, 200 + i * type_spacing['feature_update'])
                
            for i, node in enumerate(nodes_by_type['feature_hidden']):
                positions[node] = (x_base + 50, 200 + i * type_spacing['feature_hidden'])
            
            # Position output nodes at bottom
            for i, node in enumerate(nodes_by_type['output']):
                positions[node] = (x_base, 500 + i * type_spacing['output'])
                
            # Position error nodes on the side
            for i, node in enumerate(nodes_by_type['error']):
                positions[node] = (x_base + 100, 350 + i * type_spacing['error'])
        
        return positions
    
    def _get_node_color(self, node_name: str, node_info: Dict) -> str:
        """Get color for node based on type"""
        color_map = {
            'input': '#4CAF50',          # Green
            'feature_update': '#2196F3', # Blue  
            'feature_hidden': '#FF9800', # Orange
            'output': '#F44336',         # Red
            'error': '#9C27B0'          # Purple
        }
        return color_map.get(node_info['type'], '#757575')
    
    def _get_edge_color(self, weight: float) -> str:
        """Get edge color based on weight (positive = red, negative = blue)"""
        if weight > 0:
            intensity = min(abs(weight), 1.0)
            return f'rgba(255, 0, 0, {0.3 + 0.7 * intensity})'
        else:
            intensity = min(abs(weight), 1.0) 
            return f'rgba(0, 0, 255, {0.3 + 0.7 * intensity})'
    
    def _get_feature_examples(self, node_name: str, n_examples: int = 10) -> List[Dict]:
        """Get top activating examples for a feature node"""
        node_info = self._parse_node_info(node_name)
        
        if node_info['type'] not in ['feature_update', 'feature_hidden']:
            return []
        
        transcoder_type = 'update' if node_info['type'] == 'feature_update' else 'hidden'
        feature_idx = node_info['feature_idx']
        
        if feature_idx not in self.feature_analyzer.feature_activations[transcoder_type]:
            return []
        
        # Collect all activations with their sequences
        examples = []
        for seq_tuple, activations_list in self.feature_analyzer.feature_activations[transcoder_type][feature_idx].items():
            for activation in activations_list:
                for pos, magnitude in zip(activation['positions'], activation['magnitudes']):
                    examples.append({
                        'sequence': list(seq_tuple),
                        'position': pos,
                        'magnitude': magnitude,
                        'context_start': max(0, pos - 3),
                        'context_end': min(len(seq_tuple), pos + 4)
                    })
        
        # Sort by magnitude and take top examples
        examples.sort(key=lambda x: x['magnitude'], reverse=True)
        return examples[:n_examples]
    
    def _create_circuit_graph(self, edge_weights: Dict[Tuple[str, str], float], 
                            kept_nodes: Optional[Set[str]] = None) -> go.Figure:
        """Create interactive circuit graph visualization"""
        if kept_nodes:
            # Filter edges to only include kept nodes
            filtered_edges = {
                (from_node, to_node): weight 
                for (from_node, to_node), weight in edge_weights.items()
                if from_node in kept_nodes and to_node in kept_nodes
            }
        else:
            filtered_edges = edge_weights
        
        if not filtered_edges:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No edges to display",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            return fig
        
        # Compute layout
        positions = self._compute_graph_layout(filtered_edges)
        self.layout_positions = positions
        
        # Extract nodes and parse info
        all_nodes = set()
        for from_node, to_node in filtered_edges.keys():
            all_nodes.add(from_node)
            all_nodes.add(to_node)
        
        node_info = {node: self._parse_node_info(node) for node in all_nodes}
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for (from_node, to_node), weight in filtered_edges.items():
            if from_node in positions and to_node in positions:
                x0, y0 = positions[from_node]
                x1, y1 = positions[to_node]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_info.append(f"{from_node} → {to_node}: {weight:.3f}")
        
        # Add edge trace
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes by type for better legend
        node_types = ['input', 'feature_update', 'feature_hidden', 'output', 'error']
        
        for node_type in node_types:
            nodes_of_type = [node for node, info in node_info.items() if info['type'] == node_type]
            
            if not nodes_of_type:
                continue
                
            node_x = [positions[node][0] for node in nodes_of_type if node in positions]
            node_y = [positions[node][1] for node in nodes_of_type if node in positions]
            node_text = []
            node_hover = []
            
            for node in nodes_of_type:
                if node not in positions:
                    continue
                    
                info = node_info[node]
                
                # Create display text
                if info['type'] == 'input':
                    text = f"x_{info['timestep']}_{info['dimension']}"
                elif info['type'] in ['feature_update', 'feature_hidden']:
                    text = f"f_{info['feature_idx']}"
                elif info['type'] == 'output':
                    text = f"o_{info['timestep']}_{info['dimension']}"
                else:
                    text = node
                
                node_text.append(text)
                
                # Create hover info
                hover_info = f"<b>{node}</b><br>"
                hover_info += f"Type: {info['type']}<br>"
                hover_info += f"Timestep: {info['timestep']}<br>"
                
                if info['feature_idx'] is not None:
                    hover_info += f"Feature: {info['feature_idx']}<br>"
                    
                    # Add top examples for feature nodes
                    examples = self._get_feature_examples(node, 5)
                    if examples:
                        hover_info += "<br><b>Top Activations:</b><br>"
                        for i, example in enumerate(examples[:3]):
                            context = example['sequence'][example['context_start']:example['context_end']]
                            context_str = ' '.join(context)
                            hover_info += f"{i+1}. {context_str} ({example['magnitude']:.3f})<br>"
                
                node_hover.append(hover_info)
            
            # Add node trace
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=self._get_node_color(nodes_of_type[0] if nodes_of_type else '', 
                                             node_info[nodes_of_type[0]] if nodes_of_type else {}),
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color='white'),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=node_hover,
                name=node_type.replace('_', ' ').title(),
                customdata=[node for node in nodes_of_type if node in positions]
            ))
        
        # Update layout
        fig.update_layout(
            title="RNN Circuit Graph",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ 
                dict(
                    text="Hover over feature nodes to see top activating examples",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12, color='gray')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_feature_detail_panel(self, node_name: str) -> html.Div:
        """Create detailed panel for selected feature node"""
        if not node_name:
            return html.Div("Click on a feature node to see detailed examples")
        
        node_info = self._parse_node_info(node_name)
        
        if node_info['type'] not in ['feature_update', 'feature_hidden']:
            return html.Div(f"Selected node {node_name} is not a feature node")
        
        examples = self._get_feature_examples(node_name, 10)
        
        if not examples:
            return html.Div(f"No activations found for {node_name}")
        
        # Create examples display
        example_divs = []
        
        for i, example in enumerate(examples):
            sequence = example['sequence']
            pos = example['position']
            magnitude = example['magnitude']
            
            # Create colored token display
            token_spans = []
            for j, token in enumerate(sequence):
                if j == pos:
                    # Highlight the activating position
                    color = f'rgba(255, 0, 0, {0.3 + 0.7 * min(magnitude, 1.0)})'
                    token_spans.append(
                        html.Span(
                            token,
                            style={
                                'background-color': color,
                                'padding': '2px 4px',
                                'margin': '1px',
                                'border-radius': '3px',
                                'font-family': 'monospace',
                                'font-weight': 'bold',
                                'border': '2px solid red'
                            }
                        )
                    )
                else:
                    token_spans.append(
                        html.Span(
                            token,
                            style={
                                'padding': '2px 4px',
                                'margin': '1px',
                                'font-family': 'monospace',
                                'color': '#666'
                            }
                        )
                    )
            
            example_div = html.Div([
                html.Div([
                    html.Strong(f"Example {i+1}: "),
                    html.Span(f"Magnitude: {magnitude:.3f}, Position: {pos}")
                ], style={'margin-bottom': '5px', 'font-size': '12px'}),
                html.Div(token_spans, style={'margin-bottom': '10px'})
            ], style={'border': '1px solid #ddd', 'padding': '8px', 'margin': '5px', 'border-radius': '3px'})
            
            example_divs.append(example_div)
        
        return html.Div([
            html.H4(f"Feature {node_info['feature_idx']} ({node_info['type'].replace('_', ' ').title()})"),
            html.P(f"Top {len(examples)} activating examples:"),
            html.Div(example_divs, style={'max-height': '400px', 'overflow-y': 'auto'})
        ])
    
    def _setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = html.Div([
            html.H1("RNN Circuit Visualizer", 
                   style={'text-align': 'center', 'margin-bottom': '20px'}),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Input Sequence (space-separated):", style={'font-weight': 'bold'}),
                    dcc.Input(
                        id='sequence-input',
                        type='text',
                        placeholder='Enter sequence tokens separated by spaces',
                        value='the quick brown fox jumps',
                        style={'width': '100%', 'margin': '5px 0'}
                    )
                ], style={'width': '70%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Button('Generate Circuit', id='generate-button', 
                              style={'margin': '5px', 'padding': '10px'}),
                    html.Button('Prune Graph', id='prune-button',
                              style={'margin': '5px', 'padding': '10px'})
                ], style={'width': '28%', 'float': 'right', 'text-align': 'right'})
            ], style={'margin-bottom': '20px'}),
            
            # Graph Statistics
            html.Div(id='graph-stats', style={'margin-bottom': '10px', 'padding': '10px', 
                                             'background-color': '#f8f9fa', 'border-radius': '5px'}),
            
            # Main content area
            html.Div([
                # Graph display
                html.Div([
                    dcc.Graph(id='circuit-graph', style={'height': '600px'})
                ], style={'width': '70%', 'display': 'inline-block'}),
                
                # Feature detail panel
                html.Div([
                    html.Div(id='feature-details', style={'height': '580px', 'overflow-y': 'auto'})
                ], style={'width': '28%', 'float': 'right', 'border': '1px solid #ddd', 
                         'padding': '10px', 'margin-left': '2%'})
            ]),
            
            # Hidden div to store graph data
            html.Div(id='graph-data', style={'display': 'none'})
        ])
    
    def _setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            [Output('graph-data', 'children'),
             Output('graph-stats', 'children')],
            [Input('generate-button', 'n_clicks')],
            [State('sequence-input', 'value')]
        )
        def generate_circuit(n_clicks, sequence_text):
            """Generate circuit graph from input sequence"""
            if not n_clicks or not sequence_text:
                return "", "Enter a sequence and click 'Generate Circuit'"
            
            # Parse input sequence
            tokens = sequence_text.strip().split()
            if not tokens:
                return "", "Please enter a valid sequence"
            
            # Convert to tensor (this is simplified - you'd need proper tokenization)
            # For now, assume each token maps to an index
            try:
                # Create dummy sequence tensor - replace with your actual tokenization
                sequence_tensor = torch.randn(1, len(tokens), 10)  # Adjust input_size as needed
                self.current_sequence = sequence_tensor
                
                # Identify active features (simplified - you'd use your actual feature detection)
                active_features = {
                    'update': [(t, f, 0.5) for t in range(len(tokens)) for f in range(5)],
                    'hidden': [(t, f, 0.5) for t in range(len(tokens)) for f in range(5)]
                }
                
                # Build circuit graph
                edge_weights = self.circuit_tracer.build_circuit_graph(sequence_tensor, active_features)
                
                # Store graph data
                graph_data = {
                    'edge_weights': edge_weights,
                    'kept_nodes': None,
                    'tokens': tokens
                }
                
                stats = f"Generated circuit with {len(edge_weights)} edges for sequence: '{sequence_text}'"
                
                return json.dumps(graph_data), stats
                
            except Exception as e:
                return "", f"Error generating circuit: {str(e)}"
        
        @self.app.callback(
            Output('graph-data', 'children', allow_duplicate=True),
            [Input('prune-button', 'n_clicks')],
            [State('graph-data', 'children')],
            prevent_initial_call=True
        )
        def prune_circuit(n_clicks, graph_data_json):
            """Prune the circuit graph"""
            if not n_clicks or not graph_data_json or not self.pruner:
                return graph_data_json
            
            try:
                graph_data = json.loads(graph_data_json)
                edge_weights = graph_data['edge_weights']
                
                # Convert string keys back to tuples
                edge_weights = {eval(k): v for k, v in edge_weights.items()}
                
                # Prune graph
                pruned_edges, kept_nodes = self.pruner.prune_graph(edge_weights)
                
                # Update graph data
                graph_data['edge_weights'] = {str(k): v for k, v in pruned_edges.items()}
                graph_data['kept_nodes'] = list(kept_nodes)
                
                return json.dumps(graph_data)
                
            except Exception as e:
                return graph_data_json
        
        @self.app.callback(
            [Output('circuit-graph', 'figure'),
             Output('graph-stats', 'children', allow_duplicate=True)],
            [Input('graph-data', 'children')],
            prevent_initial_call=True
        )
        def update_graph_display(graph_data_json):
            """Update graph visualization"""
            if not graph_data_json:
                return go.Figure(), "No graph data"
            
            try:
                graph_data = json.loads(graph_data_json)
                edge_weights = graph_data['edge_weights']
                kept_nodes = set(graph_data['kept_nodes']) if graph_data['kept_nodes'] else None
                
                # Convert string keys back to tuples
                edge_weights = {eval(k): v for k, v in edge_weights.items()}
                
                fig = self._create_circuit_graph(edge_weights, kept_nodes)
                
                n_edges = len(edge_weights)
                n_nodes = len(kept_nodes) if kept_nodes else len(set(sum(edge_weights.keys(), ())))
                stats = f"Displaying {n_nodes} nodes and {n_edges} edges"
                if kept_nodes:
                    stats += " (pruned)"
                
                return fig, stats
                
            except Exception as e:
                return go.Figure(), f"Error displaying graph: {str(e)}"
        
        @self.app.callback(
            Output('feature-details', 'children'),
            [Input('circuit-graph', 'clickData')]
        )
        def update_feature_details(click_data):
            """Update feature details panel when node is clicked"""
            if not click_data:
                return self._create_feature_detail_panel(None)
            
            try:
                # Extract clicked node information
                point = click_data['points'][0]
                if 'customdata' in point:
                    node_name = point['customdata']
                    return self._create_feature_detail_panel(node_name)
                else:
                    return self._create_feature_detail_panel(None)
            except:
                return self._create_feature_detail_panel(None)
    
    def run(self, host='127.0.0.1', port=8051, debug=True):
        """Run the Dash app"""
        print(f"Starting circuit visualizer at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

# Usage example
def launch_circuit_visualizer(circuit_tracer, feature_analyzer, pruner=None):
    """Launch the interactive circuit visualizer"""
    visualizer = InteractiveCircuitVisualizer(circuit_tracer, feature_analyzer, pruner)
    visualizer.run()

# Example usage:
if __name__ == "__main__":
    # Assuming you have the required components:
    # circuit_tracer = CircuitTracer(rnn_model, update_transcoder, hidden_transcoder)
    # feature_analyzer = FeatureActivationAnalyzer(rnn_model, update_transcoder, hidden_transcoder)
    # pruner = GraphPruner()
    
    # Launch visualizer:
    # launch_circuit_visualizer(circuit_tracer, feature_analyzer, pruner)
    pass