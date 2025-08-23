from typing import Dict, List, Optional
import pickle 
import traceback

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import colorsys

import pandas as pd
import numpy as np

from circuit.copy_find_features import CopyFeatureActivationAnalyzer

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
        
        # Create color: blue to red interpolation (matching legend)
        # Low activation = blue (100, 100, 255)
        # High activation = red (230, 0, 0)
        
        # Interpolate between blue and red
        blue_rgb = (100, 100, 255)
        red_rgb = (230, 0, 0)
        
        # Linear interpolation between blue and red
        r = int(blue_rgb[0] + (red_rgb[0] - blue_rgb[0]) * intensity)
        g = int(blue_rgb[1] + (red_rgb[1] - blue_rgb[1]) * intensity)
        b = int(blue_rgb[2] + (red_rgb[2] - blue_rgb[2]) * intensity)
        
        # Alpha increases with intensity (0.3 to 1.0)
        alpha = 0.3 + 0.7 * intensity
        
        return f"rgba({r}, {g}, {b}, {alpha})"
        
    def _create_sequence_display(self, sequences_data: List[Dict], 
                                 feature_idx: int) -> List[html.Div]:
        """Create colored token displays for sequences"""
        try:
            if not sequences_data:
                return [html.Div("No activations found for this feature.", className="no-data")]
            
            # Calculate max magnitude across all sequences for normalization
            all_magnitudes = []
            for seq_data in sequences_data:
                if isinstance(seq_data, dict) and 'magnitudes' in seq_data:
                    all_magnitudes.extend(seq_data['magnitudes'])
            max_magnitude = max(all_magnitudes) if all_magnitudes else 1.0
            
            sequence_divs = []
            
            for seq_idx, seq_data in enumerate(sequences_data[:20]):  # Show up to 20 sequences
                try:
                    if not isinstance(seq_data, dict):
                        continue
                        
                    sequence = seq_data.get('sequence', [])
                    positions = seq_data.get('positions', [])
                    magnitudes = seq_data.get('magnitudes', [])
                    
                    # Create position -> magnitude mapping
                    pos_to_magnitude = {pos: mag for pos, mag in zip(positions, magnitudes)}
                    
                    # Create token spans
                    token_spans = []
                    for pos, token in enumerate(sequence):
                        magnitude = pos_to_magnitude.get(pos, 0)
                        color = self._get_color_intensity(magnitude, max_magnitude)
                        
                        token_spans.append(
                            html.Span(
                                str(token),  # Convert to string to be safe
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
                    total_activation = sum(magnitudes) if magnitudes else 0
                    max_activation = max(magnitudes) if magnitudes else 0
                    seq_length = len(sequence)
                    
                    seq_div = html.Div([
                        html.Div([
                            html.Strong(f"Sequence {seq_idx + 1} (Length {seq_length}): "),
                            html.Span(f"Total: {total_activation:.3f}, Max: {max_activation:.3f}, Positions: {len(positions)}")
                        ], style={'margin-bottom': '5px', 'font-size': '12px', 'color': '#666'}),
                        html.Div(token_spans, style={'margin-bottom': '15px'})
                    ], 
                    style={'border': '1px solid #ddd', 'padding': '10px', 'margin': '5px', 'border-radius': '5px'},
                    id={'type': 'sequence', 'index': seq_idx}
                    )
                    
                    sequence_divs.append(seq_div)
                except Exception as e:
                    print(f"Error processing sequence {seq_idx}: {e}")
                    continue
                
            return sequence_divs
        except Exception as e:
            print(f"Error in _create_sequence_display: {e}")
            traceback.print_exc()
            return [html.Div(f"Error creating sequence display: {str(e)}")]
        
    def _setup_layout(self):
        """Setup the Dash app layout"""
        
        try:
            # Get available features for dropdowns with error handling
            update_features = []
            hidden_features = []
            
            if hasattr(self.analyzer, 'feature_activations') and isinstance(self.analyzer.feature_activations, dict):
                if 'update' in self.analyzer.feature_activations:
                    update_features = list(self.analyzer.feature_activations['update'].keys())
                if 'hidden' in self.analyzer.feature_activations:
                    hidden_features = list(self.analyzer.feature_activations['hidden'].keys())
            
            # Fallback if no features found
            if not update_features and not hidden_features:
                update_features = [0]  # Default feature
                
        except Exception as e:
            print(f"Error getting features: {e}")
            update_features = [0]
            hidden_features = [0]
        
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
                        value=update_features[0] if update_features else 0,
                        style={'margin': '10px 0'}
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={'margin-bottom': '20px'}),
            
            # Debug info
            html.Div(id='debug-info', style={'margin-bottom': '10px', 'padding': '10px', 
                                           'background-color': '#fff3cd', 'border-radius': '5px'}),
            
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
            ], style={'margin-bottom': '10px'}),
            
            # Category Controls
            html.Div([
                html.Label("Show sequences:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.RadioItems(
                    id='category-option',
                    options=[
                        {'label': 'Top 20 Best (High Activation)', 'value': 'best'},
                        {'label': 'Top 20 Worst (Low Activation)', 'value': 'worst'}
                    ],
                    value='best',
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
            try:
                if (hasattr(self.analyzer, 'feature_activations') and 
                    isinstance(self.analyzer.feature_activations, dict) and
                    transcoder_type in self.analyzer.feature_activations):
                    
                    features = list(self.analyzer.feature_activations[transcoder_type].keys())
                else:
                    features = [0]  # Default fallback
                
                options = [{'label': f'Feature {f}', 'value': f} for f in features[:100]]  # Limit to first 100
                value = features[0] if features else 0
                
                return options, value
            except Exception as e:
                print(f"Error in update_feature_dropdown: {e}")
                traceback.print_exc()
                return [{'label': 'Feature 0', 'value': 0}], 0
            
        @self.app.callback(
            Output('debug-info', 'children'),
            [Input('transcoder-type', 'value'),
             Input('feature-dropdown', 'value')]
        )
        def update_debug_info(transcoder_type, feature_idx):
            """Show debug information"""
            try:
                debug_info = []
                debug_info.append(f"Transcoder type: {transcoder_type}")
                debug_info.append(f"Feature index: {feature_idx} (type: {type(feature_idx)})")
                
                if hasattr(self.analyzer, 'feature_activations'):
                    debug_info.append(f"Feature activations type: {type(self.analyzer.feature_activations)}")
                    if isinstance(self.analyzer.feature_activations, dict):
                        debug_info.append(f"Available transcoder types: {list(self.analyzer.feature_activations.keys())}")
                        if transcoder_type in self.analyzer.feature_activations:
                            debug_info.append(f"Features in {transcoder_type}: {len(self.analyzer.feature_activations[transcoder_type])} features")
                            if feature_idx in self.analyzer.feature_activations[transcoder_type]:
                                data_sample = self.analyzer.feature_activations[transcoder_type][feature_idx]
                                debug_info.append(f"Data sample type: {type(data_sample)}")
                                if isinstance(data_sample, dict):
                                    debug_info.append(f"Number of sequences: {len(data_sample)}")
                
                return html.Pre('\n'.join(debug_info))
            except Exception as e:
                return html.Pre(f"Debug error: {e}\n{traceback.format_exc()}")
            
        @self.app.callback(
            Output('feature-stats', 'children'),
            [Input('transcoder-type', 'value'),
             Input('feature-dropdown', 'value')]
        )
        def update_feature_stats(transcoder_type, feature_idx):
            """Update feature statistics display"""
            try:
                if feature_idx is None:
                    return "No feature selected"
                
                # Check if analyzer has the get_feature_summary method
                if hasattr(self.analyzer, 'get_feature_summary'):
                    try:
                        stats = self.analyzer.get_feature_summary(transcoder_type, feature_idx)
                        
                        if stats.get('n_sequences', 0) == 0:
                            return "This feature has no activations."
                        
                        stats_div = html.Div([
                            html.H4(f"{transcoder_type.title()} Gate Feature {feature_idx}"),
                            html.P(f"Active in {stats['n_sequences']} sequences"),
                            html.P(f"Total activations: {stats['n_activations']}"),
                            html.P(f"Average activations per sequence: {stats['avg_activations_per_sequence']:.2f}"),
                        ])
                        
                        if stats.get('magnitude_stats'):
                            mag_stats = stats['magnitude_stats']
                            stats_div.children.extend([
                                html.P(f"Activation magnitude - Mean: {mag_stats['mean']:.3f}, "
                                      f"Std: {mag_stats['std']:.3f}, Range: [{mag_stats['min']:.3f}, {mag_stats['max']:.3f}]")
                            ])
                        
                        return stats_div
                    except Exception as e:
                        return html.Div(f"Error getting feature summary: {str(e)}")
                else:
                    # Manual stats calculation
                    if (hasattr(self.analyzer, 'feature_activations') and 
                        isinstance(self.analyzer.feature_activations, dict) and
                        transcoder_type in self.analyzer.feature_activations and
                        feature_idx in self.analyzer.feature_activations[transcoder_type]):
                        
                        data = self.analyzer.feature_activations[transcoder_type][feature_idx]
                        return html.Div([
                            html.H4(f"{transcoder_type.title()} Gate Feature {feature_idx}"),
                            html.P(f"Data available: {len(data) if isinstance(data, dict) else 'Unknown format'}")
                        ])
                    else:
                        return "This feature has no activations."
                        
            except Exception as e:
                print(f"Error in update_feature_stats: {e}")
                traceback.print_exc()
                return html.Div(f"Error updating stats: {str(e)}")
            
        @self.app.callback(
            Output('sequence-display', 'children'),
            [Input('transcoder-type', 'value'),
             Input('feature-dropdown', 'value'),
             Input('sort-option', 'value'),
             Input('category-option', 'value')]
        )
        def update_sequence_display(transcoder_type, feature_idx, sort_option, category_option):
            """Update main sequence display"""
            try:
                print(f"=== DEBUG: update_sequence_display called ===")
                print(f"transcoder_type: {transcoder_type} (type: {type(transcoder_type)})")
                print(f"feature_idx: {feature_idx} (type: {type(feature_idx)})")
                print(f"sort_option: {sort_option} (type: {type(sort_option)})")
                print(f"category_option: {category_option} (type: {type(category_option)})")
                
                if feature_idx is None:
                    return [html.Div("No feature selected")]
                
                # Check if analyzer exists
                if not hasattr(self.analyzer, 'feature_activations'):
                    print("ERROR: analyzer has no feature_activations attribute")
                    return [html.Div("Analyzer missing feature_activations")]
                
                # Check if feature_activations is a dict
                if not isinstance(self.analyzer.feature_activations, dict):
                    print(f"ERROR: feature_activations is not a dict, it's {type(self.analyzer.feature_activations)}")
                    return [html.Div(f"feature_activations is {type(self.analyzer.feature_activations)}, not dict")]
                
                # Check if transcoder_type exists
                if transcoder_type not in self.analyzer.feature_activations:
                    available_types = list(self.analyzer.feature_activations.keys())
                    print(f"ERROR: transcoder_type '{transcoder_type}' not found. Available: {available_types}")
                    return [html.Div(f"Transcoder type '{transcoder_type}' not found. Available: {available_types}")]
                
                # Check if feature_idx exists
                if feature_idx not in self.analyzer.feature_activations[transcoder_type]:
                    available_features = list(self.analyzer.feature_activations[transcoder_type].keys())[:10]  # Show first 10
                    print(f"ERROR: feature_idx {feature_idx} not found. Sample features: {available_features}")
                    return [html.Div(f"Feature {feature_idx} not found")]
                
                feature_data = self.analyzer.feature_activations[transcoder_type][feature_idx]
                print(f"feature_data type: {type(feature_data)}")
                
                if isinstance(feature_data, dict):
                    print(f"feature_data has {len(feature_data)} items")
                    # Show a sample of the data structure
                    sample_keys = list(feature_data.keys())[:3]
                    for key in sample_keys:
                        print(f"Sample key: {key} (type: {type(key)}) -> {type(feature_data[key])}")
                        if isinstance(feature_data[key], dict):
                            print(f"  Dict keys: {list(feature_data[key].keys())}")
                        elif isinstance(feature_data[key], list) and len(feature_data[key]) > 0:
                            print(f"  List length: {len(feature_data[key])}, first item type: {type(feature_data[key][0])}")
                
                if not isinstance(feature_data, dict):
                    return [html.Div(f"Unexpected data format: {type(feature_data)}")]
                
                sequences_data = []
                item_count = 0
                for seq_tuple, activations_list in feature_data.items():
                    try:
                        item_count += 1
                        if item_count <= 3:  # Debug first 3 items
                            print(f"Processing item {item_count}:")
                            print(f"  seq_tuple: {seq_tuple} (type: {type(seq_tuple)})")
                            print(f"  activations_list: (type: {type(activations_list)})")
                        
                        # Handle different data structures
                        if isinstance(activations_list, list):
                            # Combine all activations for this sequence
                            all_positions = []
                            all_magnitudes = []
                            for activation in activations_list:
                                if isinstance(activation, dict):
                                    all_positions.extend(activation.get('positions', []))
                                    all_magnitudes.extend(activation.get('magnitudes', []))
                        elif isinstance(activations_list, dict):
                            # Single activation dict
                            all_positions = activations_list.get('positions', [])
                            all_magnitudes = activations_list.get('magnitudes', [])
                        else:
                            if item_count <= 3:
                                print(f"  Skipping unknown type: {type(activations_list)}")
                            continue
                        
                        # Convert seq_tuple to sequence
                        if isinstance(seq_tuple, tuple):
                            sequence = list(seq_tuple)
                        elif isinstance(seq_tuple, str):
                            sequence = [seq_tuple]  # Single string token
                        else:
                            sequence = [str(seq_tuple)]
                        
                        sequences_data.append({
                            'sequence': sequence,
                            'positions': all_positions,
                            'magnitudes': all_magnitudes
                        })
                        
                        if item_count <= 3:
                            print(f"  Added sequence with {len(sequence)} tokens, {len(all_positions)} positions")
                            
                    except Exception as e:
                        print(f"Error processing sequence {seq_tuple}: {e}")
                        traceback.print_exc()
                        continue
                
                print(f"Total sequences_data: {len(sequences_data)}")
                
                # Sort sequences based on selected option
                def get_sort_key(x, sort_option):
                    if sort_option == 'total':
                        return sum(x.get('magnitudes', []))
                    elif sort_option == 'max':
                        return max(x.get('magnitudes', [0])) if x.get('magnitudes') else 0
                    elif sort_option == 'positions':
                        return len(x.get('positions', []))
                    elif sort_option == 'length':
                        return len(x.get('sequence', []))
                    return 0
                
                try:
                    # First, sort all sequences
                    sequences_data.sort(key=lambda x: get_sort_key(x, sort_option), reverse=True)
                    
                    # Group by sequence length
                    length_groups = {}
                    for seq_data in sequences_data:
                        seq_length = len(seq_data['sequence'])
                        if seq_length not in length_groups:
                            length_groups[seq_length] = []
                        length_groups[seq_length].append(seq_data)
                    
                    print(f"Found {len(length_groups)} different sequence lengths: {sorted(length_groups.keys())}")
                    
                    # Select top 20 best or worst, sampling across different lengths
                    selected_sequences = []
                    
                    if category_option == 'best':
                        # For best: sort lengths by their top sequence score (descending)
                        length_scores = []
                        for length, seqs in length_groups.items():
                            if seqs:
                                top_score = get_sort_key(seqs[0], sort_option)  # Already sorted descending
                                length_scores.append((length, top_score, seqs))
                        
                        # Sort by top score descending
                        length_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        # Take top sequences from each length group, round-robin style
                        sequences_per_length = max(1, 20 // len(length_groups)) if length_groups else 1
                        remaining_slots = 20
                        
                        # First pass: take sequences_per_length from each group
                        for length, score, seqs in length_scores:
                            if remaining_slots <= 0:
                                break
                            take_count = min(sequences_per_length, len(seqs), remaining_slots)
                            selected_sequences.extend(seqs[:take_count])
                            remaining_slots -= take_count
                        
                        # Second pass: fill remaining slots with best remaining sequences
                        if remaining_slots > 0:
                            all_remaining = []
                            for length, score, seqs in length_scores:
                                start_idx = min(sequences_per_length, len(seqs))
                                all_remaining.extend(seqs[start_idx:])
                            
                            all_remaining.sort(key=lambda x: get_sort_key(x, sort_option), reverse=True)
                            selected_sequences.extend(all_remaining[:remaining_slots])
                    
                    else:  # worst
                        # For worst: sort lengths by their worst sequence score (ascending)  
                        length_scores = []
                        for length, seqs in length_groups.items():
                            if seqs:
                                # Sort this group ascending for worst scores
                                seqs.sort(key=lambda x: get_sort_key(x, sort_option), reverse=False)
                                worst_score = get_sort_key(seqs[0], sort_option)
                                length_scores.append((length, worst_score, seqs))
                        
                        # Sort by worst score ascending
                        length_scores.sort(key=lambda x: x[1], reverse=False)
                        
                        # Take worst sequences from each length group
                        sequences_per_length = max(1, 20 // len(length_groups)) if length_groups else 1
                        remaining_slots = 20
                        
                        # First pass: take sequences_per_length from each group
                        for length, score, seqs in length_scores:
                            if remaining_slots <= 0:
                                break
                            take_count = min(sequences_per_length, len(seqs), remaining_slots)
                            selected_sequences.extend(seqs[:take_count])
                            remaining_slots -= take_count
                        
                        # Second pass: fill remaining slots with worst remaining sequences
                        if remaining_slots > 0:
                            all_remaining = []
                            for length, score, seqs in length_scores:
                                start_idx = min(sequences_per_length, len(seqs))
                                all_remaining.extend(seqs[start_idx:])
                            
                            all_remaining.sort(key=lambda x: get_sort_key(x, sort_option), reverse=False)
                            selected_sequences.extend(all_remaining[:remaining_slots])
                    
                    # Final sort for display
                    if category_option == 'best':
                        selected_sequences.sort(key=lambda x: get_sort_key(x, sort_option), reverse=True)
                    else:
                        selected_sequences.sort(key=lambda x: get_sort_key(x, sort_option), reverse=False)
                    
                    print(f"Selected {len(selected_sequences)} sequences for {category_option}")
                    sequences_data = selected_sequences
                    
                except Exception as e:
                    print(f"Error in sequence selection: {e}")
                    traceback.print_exc()
                    # Fallback to simple approach
                    sequences_data.sort(key=lambda x: get_sort_key(x, sort_option), 
                                      reverse=(category_option == 'best'))
                    sequences_data = sequences_data[:20]
                
                print("Calling _create_sequence_display...")
                result = self._create_sequence_display(sequences_data, feature_idx)
                print(f"_create_sequence_display returned {len(result)} items")
                return result
                
            except Exception as e:
                error_msg = f"Error in update_sequence_display: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                return [html.Div(error_msg)]
    
    def run(self, host='0.0.0.0', port=8050, debug=True):
        """Run the Dash app"""
        print(f"Starting interactive visualizer at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

def launch_visualizer(analyzer):
    visualizer = InteractiveFeatureVisualizer(analyzer)
    visualizer.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dict_path", required=True)

    args = parser.parse_args()
    with open(args.feature_dict_path, "rb") as f:
        analysis_dict = pickle.load(f)
    analyzer = CopyFeatureActivationAnalyzer(None, None, None, "cpu")
    analyzer.feature_activations = analysis_dict
    launch_visualizer(analyzer)