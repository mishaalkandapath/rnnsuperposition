from typing import Dict, List, Optional

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import colorsys

import pandas as pd
import numpy as np

class InteractiveFeatureVisualizer:
    """Interactive web-based visualizer for RNN transcoder feature activations"""
    
    def __init__(self, analyzer):
        """
        Args:
            analyzer: FeatureActivationAnalyzer instance with collected data
        """
        self.analyzer = analyzer
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _get_color_intensity(self, magnitude: float, max_magnitude: float) -> str:
        """Convert activation magnitude to color intensity"""
        if magnitude == 0:
            return "rgba(200, 200, 200, 0.3)"  # Light gray for inactive
        
        # Normalize magnitude to 0-1 range
        intensity = min(magnitude / max_magnitude, 1.0) if max_magnitude > 0 else 0
        
        # Create color: light blue to dark red based on intensity
        # HSV: Hue from 240° (blue) to 0° (red), high saturation and value
        hue = (1 - intensity) * 240 / 360  # 240° to 0° 
        saturation = 0.8
        value = 0.9
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = tuple(int(c * 255) for c in rgb)
        
        return f"rgba({rgb_255[0]}, {rgb_255[1]}, {rgb_255[2]}, {0.3 + 0.7 * intensity})"
        
    def _create_sequence_display(self, sequences_data: List[Dict], feature_idx: int) -> List[html.Div]:
        """Create colored token displays for sequences"""
        if not sequences_data:
            return [html.Div("No activations found for this feature.", className="no-data")]
        
        # Calculate max magnitude across all sequences for normalization
        all_magnitudes = []
        for seq_data in sequences_data:
            all_magnitudes.extend(seq_data['magnitudes'])
        max_magnitude = max(all_magnitudes) if all_magnitudes else 1.0
        
        sequence_divs = []
        
        for seq_idx, seq_data in enumerate(sequences_data[:50]):  # Limit to first 50 sequences
            sequence = seq_data['sequence']
            positions = seq_data['positions']
            magnitudes = seq_data['magnitudes']
            
            # Create position -> magnitude mapping
            pos_to_magnitude = {pos: mag for pos, mag in zip(positions, magnitudes)}
            
            # Create token spans
            token_spans = []
            for pos, token in enumerate(sequence):
                magnitude = pos_to_magnitude.get(pos, 0)
                color = self._get_color_intensity(magnitude, max_magnitude)
                
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
                            'border': '1px solid #ccc'
                        },
                        title=f"Position {pos}: {magnitude:.3f}" if magnitude > 0 else f"Position {pos}: inactive"
                    )
                )
            
            # Add sequence info
            total_activation = sum(magnitudes)
            seq_div = html.Div([
                html.Div([
                    html.Strong(f"Sequence {seq_idx + 1}: "),
                    html.Span(f"Total activation: {total_activation:.3f}, Positions: {len(positions)}")
                ], style={'margin-bottom': '5px', 'font-size': '12px', 'color': '#666'}),
                html.Div(token_spans, style={'margin-bottom': '15px'})
            ], 
            style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '5px', 'border-radius': '5px'},
            id={'type': 'sequence', 'index': seq_idx}
            )
            
            sequence_divs.append(seq_div)
            
        return sequence_divs
        
    def _setup_layout(self):
        """Setup the Dash app layout"""
        
        # Get available features for dropdowns
        update_features = list(self.analyzer.feature_activations['update'].keys())
        hidden_features = list(self.analyzer.feature_activations['hidden'].keys())
        
        self.app.layout = html.Div([
            html.H1("RNN Transcoder Feature Activation Visualizer", 
                   style={'text-align': 'center', 'margin-bottom': '30px'}),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Transcoder Type:", style={'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='transcoder-type',
                        options=[
                            {'label': 'Update Gate', 'value': 'update'},
                            {'label': 'Hidden Context', 'value': 'hidden'}
                        ],
                        value='update',
                        inline=True,
                        style={'margin': '10px 0'}
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Feature Index:", style={'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=[{'label': f'Feature {f}', 'value': f} for f in update_features[:100]],  # Limit options
                        value=update_features[0] if update_features else None,
                        style={'margin': '10px 0'}
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={'margin-bottom': '20px'}),
            
            # Feature Statistics
            html.Div(id='feature-stats', style={'margin-bottom': '20px', 'padding': '15px', 
                                               'background-color': '#f8f9fa', 'border-radius': '5px'}),
            
            # Sorting Controls
            html.Div([
                html.Label("Sort sequences by:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.RadioItems(
                    id='sort-option',
                    options=[
                        {'label': 'Total Activation', 'value': 'total'},
                        {'label': 'Max Activation', 'value': 'max'},
                        {'label': 'Number of Positions', 'value': 'positions'},
                        {'label': 'Sequence Length', 'value': 'length'}
                    ],
                    value='total',
                    inline=True
                )
            ], style={'margin-bottom': '20px'}),
            
            # Color Legend
            html.Div([
                html.H4("Color Legend:", style={'margin-bottom': '10px'}),
                html.Div([
                    html.Span("Inactive", style={'background-color': 'rgba(200, 200, 200, 0.3)', 
                                               'padding': '2px 8px', 'margin-right': '10px', 'border-radius': '3px'}),
                    html.Span("Low", style={'background-color': 'rgba(100, 100, 255, 0.5)', 
                                          'padding': '2px 8px', 'margin-right': '10px', 'border-radius': '3px'}),
                    html.Span("Medium", style={'background-color': 'rgba(255, 100, 100, 0.7)', 
                                             'padding': '2px 8px', 'margin-right': '10px', 'border-radius': '3px'}),
                    html.Span("High", style={'background-color': 'rgba(230, 0, 0, 1.0)', 
                                           'padding': '2px 8px', 'border-radius': '3px'})
                ])
            ], style={'margin-bottom': '20px', 'padding': '10px', 'background-color': '#f0f0f0', 'border-radius': '5px'}),
            
            # Main visualization area
            html.Div(id='sequence-display', style={'max-height': '600px', 'overflow-y': 'auto', 
                                                  'border': '1px solid #ddd', 'padding': '10px'})
        ])
        
    def _setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            [Output('feature-dropdown', 'options'),
             Output('feature-dropdown', 'value')],
            [Input('transcoder-type', 'value')]
        )
        def update_feature_dropdown(transcoder_type):
            """Update feature dropdown based on selected transcoder type"""
            if transcoder_type == 'update':
                features = list(self.analyzer.feature_activations['update'].keys())
            else:
                features = list(self.analyzer.feature_activations['hidden'].keys())
            
            options = [{'label': f'Feature {f}', 'value': f} for f in features[:100]]  # Limit to first 100
            value = features[0] if features else None
            
            return options, value
            
        @self.app.callback(
            Output('feature-stats', 'children'),
            [Input('transcoder-type', 'value'),
             Input('feature-dropdown', 'value')]
        )
        def update_feature_stats(transcoder_type, feature_idx):
            """Update feature statistics display"""
            if feature_idx is None:
                return "No feature selected"
                
            stats = self.analyzer.get_feature_summary(transcoder_type, feature_idx)
            
            if stats['n_sequences'] == 0:
                return "This feature has no activations."
            
            stats_div = html.Div([
                html.H4(f"{transcoder_type.title()} Gate Feature {feature_idx}"),
                html.P(f"Active in {stats['n_sequences']} sequences"),
                html.P(f"Total activations: {stats['n_activations']}"),
                html.P(f"Average activations per sequence: {stats['avg_activations_per_sequence']:.2f}"),
            ])
            
            if stats['magnitude_stats']:
                mag_stats = stats['magnitude_stats']
                stats_div.children.extend([
                    html.P(f"Activation magnitude - Mean: {mag_stats['mean']:.3f}, "
                          f"Std: {mag_stats['std']:.3f}, Range: [{mag_stats['min']:.3f}, {mag_stats['max']:.3f}]")
                ])
            
            return stats_div
            
        @self.app.callback(
            Output('sequence-display', 'children'),
            [Input('transcoder-type', 'value'),
             Input('feature-dropdown', 'value'),
             Input('sort-option', 'value')]
        )
        def update_sequence_display(transcoder_type, feature_idx, sort_option):
            """Update main sequence display"""
            if feature_idx is None:
                return [html.Div("No feature selected")]
            
            if feature_idx not in self.analyzer.feature_activations[transcoder_type]:
                return [html.Div("This feature has no activations.")]
            
            sequences_data = []
            for seq_tuple, activations_list in self.analyzer.feature_activations[transcoder_type][feature_idx].items():
                # Combine all activations for this sequence
                all_positions = []
                all_magnitudes = []
                for activation in activations_list:
                    all_positions.extend(activation['positions'])
                    all_magnitudes.extend(activation['magnitudes'])
                
                sequences_data.append({
                    'sequence': list(seq_tuple),
                    'positions': all_positions,
                    'magnitudes': all_magnitudes
                })
            
            # Sort sequences based on selected option
            if sort_option == 'total':
                sequences_data.sort(key=lambda x: sum(x['magnitudes']), reverse=True)
            elif sort_option == 'max':
                sequences_data.sort(key=lambda x: max(x['magnitudes']) if x['magnitudes'] else 0, reverse=True)
            elif sort_option == 'positions':
                sequences_data.sort(key=lambda x: len(x['positions']), reverse=True)
            elif sort_option == 'length':
                sequences_data.sort(key=lambda x: len(x['sequence']), reverse=True)
            
            return self._create_sequence_display(sequences_data, feature_idx)
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the Dash app"""
        print(f"Starting interactive visualizer at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

# Usage example
def launch_visualizer(analyzer):
    """Launch the interactive visualizer"""
    visualizer = InteractiveFeatureVisualizer(analyzer)
    visualizer.run()

# Example usage:
if __name__ == "__main__":
    # Assuming you have an analyzer with collected data:
    # rnn_model, u
    # analyzer = FeatureActivationAnalyzer(rnn_model, update_transcoder, hidden_transcoder)
    # analyzer.analyze_all_sequences(...)
    
    # Launch visualizer:
    # launch_visualizer(analyzer)
    pass