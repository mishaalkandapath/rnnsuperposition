from typing import Dict, List, Optional, Set, Tuple
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

class EnhancedInteractiveFeatureVisualizer:
    """Interactive web-based visualizer for RNN transcoder feature activations with inactive sequence support"""
    
    def __init__(self, analyzer, all_sequences=None, 
                 hidden_features=None, update_features=None):
        """
        Args:
            analyzer: FeatureActivationAnalyzer instance with collected data
            all_sequences: List/Set of all sequences that were analyzed (for finding inactive ones)
            total_features: Total number of features (for finding features that never activate)
        """
        self.analyzer = analyzer
        self.all_sequences = set(all_sequences) if all_sequences else None
        self.all_counts = {}
        for sequence in self.all_sequences:
            self.all_counts[len(sequence)] = self.all_counts.get(len(sequence), 0) + 1

        assert sum(self.all_counts.values()) == len(self.all_sequences), f"{sum(self.all_counts.values())} {len(self.all_sequences)}\n{self.all_sequences[0]}"

        self.hidden_features = hidden_features
        self.update_features = update_features
        self._inactive_sequences_cache = {}

        self.app = dash.Dash(__name__)        
        self._setup_layout()
        self._setup_callbacks()
        
    def _get_inactive_sequences_for_feature(self, transcoder_type: str,
                                             feature_idx: int) -> List[Dict]:
        """Get sequences where the specified feature never activated"""
        cache_key = (transcoder_type, feature_idx)
        
        if cache_key in self._inactive_sequences_cache:
            return self._inactive_sequences_cache[cache_key]
        
        if self.all_sequences is None:
            print(f"Warning: all_sequences not provided, cannot find inactive sequences for feature {feature_idx}")
            self._inactive_sequences_cache[cache_key] = []
            return []
        
        try:
            # Get sequences where this feature was active
            active_sequences = set()
            if (hasattr(self.analyzer, 'feature_activations') and 
                isinstance(self.analyzer.feature_activations, dict) and
                transcoder_type in self.analyzer.feature_activations and
                feature_idx in self.analyzer.feature_activations[transcoder_type]):
                
                feature_data = self.analyzer.feature_activations[transcoder_type][feature_idx]
                active_sequences = set(feature_data.keys())
            
            # inactive sequences
            inactive_sequence_tuples = self.all_sequences - active_sequences
            inactive_sequences_data = []
            for sequence in inactive_sequence_tuples:
                inactive_sequences_data.append({
                    'sequence': sequence,
                    'is_active': False})
            
            print(f"Found {len(inactive_sequences_data)} inactive sequences for {transcoder_type} feature {feature_idx}")
            self._inactive_sequences_cache[cache_key] = inactive_sequences_data
            return inactive_sequences_data
            
        except Exception as e:
            print(f"Error getting inactive sequences for feature {feature_idx}: {e}")
            traceback.print_exc()
            self._inactive_sequences_cache[cache_key] = []
            return []
        
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
        # sequences_data must be sorted according to whatever metric
        try:
            if not sequences_data:
                return [html.Div("No sequences found for this feature.", className="no-data")]
            
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
                        print("SHOULD'VE BEEN A DICT? ", seq_data)
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
                                str(token),  
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
                    is_inactive = len(positions) == 0
                    
                    # Different styling for inactive sequences
                    seq_style = {
                        'border': '1px solid #ddd', 
                        'padding': '10px', 
                        'margin': '5px', 
                        'border-radius': '5px'
                    }
                    if is_inactive:
                        seq_style['background-color'] = '#f8f8f8'
                        seq_style['border-color'] = '#ccc'
                    
                    info_text = f"Total: {total_activation:.3f}, Max: {max_activation:.3f}, Positions: {len(positions)}"
                    if is_inactive:
                        info_text += " [INACTIVE]"
                    
                    seq_div = html.Div([
                        html.Div([
                            html.Strong(f"Sequence {seq_idx + 1} (Length {seq_length}): "),
                            html.Span(info_text, style={'color': '#999' if is_inactive else '#666'})
                        ], style={'margin-bottom': '5px', 'font-size': '12px'}),
                        html.Div(token_spans, style={'margin-bottom': '15px'})
                    ], 
                    style=seq_style,
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
            html.H1("Enhanced RNN Transcoder Feature Activation Visualizer", 
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
                        {'label': 'Top 20 Worst (Low/No Activation)', 'value': 'worst'}
                    ],
                    value='best',
                    inline=True
                )
            ], style={'margin-bottom': '20px'}),
            
            # Include inactive sequences toggle
            html.Div([
                dcc.Checklist(
                    id='include-inactive',
                    options=[
                        {'label': 'Include sequences where feature never activates (for "worst" view)', 'value': 'include'}
                    ],
                    value=['include'] if self.all_sequences is not None else [],
                    style={'margin': '10px 0'}
                ),
                html.Div(
                    f"Total sequences available: {len(self.all_sequences) if self.all_sequences else 'Unknown'}" +
                    (f", Total features: {self.hidden_features+self.update_features}"),
                    style={'font-size': '12px', 'color': '#666', 'margin-left': '20px'}
                )
            ], style={'margin-bottom': '20px', 'padding': '10px', 'background-color': '#f9f9f9', 'border-radius': '5px'}),
            
            # Color Legend
            html.Div([
                html.H4("Color Legend:", style={'margin-bottom': '10px'}),
                html.Div([
                    html.Span("Inactive/Never Active", style={'background-color': 'rgba(200, 200, 200, 0.3)', 
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
                
                options = [{'label': f'Feature {f}', 'value': f} for f in features]
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
                                    debug_info.append(f"Number of active sequences: {len(data_sample)}")
                
                # Add info about inactive sequences
                if self.all_sequences:
                    inactive_count = len(self._get_inactive_sequences_for_feature(transcoder_type, feature_idx))
                    debug_info.append(f"Number of inactive sequences: {inactive_count}")
                
                return html.Pre('\n'.join(debug_info))
            except Exception as e:
                return html.Pre(f"Debug error: {e}\n{traceback.format_exc()}")
            
        @self.app.callback(
            Output('feature-stats', 'children'),
            [Input('transcoder-type', 'value'),
             Input('feature-dropdown', 'value'),
             Input('include-inactive', 'value')]
        )
        def update_feature_stats(transcoder_type, feature_idx, include_inactive):
            """Update feature statistics display"""
            try:
                if feature_idx is None:
                    return "No feature selected"
                
                # Get active sequences count
                active_sequences_count = 0
                if (hasattr(self.analyzer, 'feature_activations') and 
                    isinstance(self.analyzer.feature_activations, dict) and
                    transcoder_type in self.analyzer.feature_activations and
                    feature_idx in self.analyzer.feature_activations[transcoder_type]):
                    
                    feature_data = self.analyzer.feature_activations[transcoder_type][feature_idx]
                    active_sequences_count = len(feature_data) if isinstance(feature_data, dict) else 0
                
                # Get inactive sequences count
                inactive_sequences_count = 0
                if self.all_sequences and 'include' in include_inactive:
                    inactive_sequences = self._get_inactive_sequences_for_feature(transcoder_type, feature_idx)
                    inactive_sequences_count = len(inactive_sequences)
                
                total_sequences = active_sequences_count + inactive_sequences_count
                if self.all_sequences:
                    total_available = len(self.all_sequences)
                else:
                    total_available = "Unknown"
                
                # Check if analyzer has the get_feature_summary method
                stats_div = html.Div([
                    html.H4(f"{transcoder_type.title()} Gate Feature {feature_idx}"),
                    html.P(f"Active in {active_sequences_count} sequences"),
                    html.P(f"Inactive in {inactive_sequences_count} sequences"),
                    html.P(f"Total sequences being considered: {total_sequences}"),
                    html.P(f"Total sequences available: {total_available}"),
                ])
                
                if hasattr(self.analyzer, 'get_feature_summary'):
                    # try:
                    stats = self.analyzer.get_feature_summary(transcoder_type, feature_idx)
                    
                    if stats.get('n_sequences', 0) > 0:
                        stats_div.children.extend([
                            html.P(f"Total activations: {stats['n_activations']}"),
                            html.P(f"Average activations per active sequence: {stats['avg_activations_per_sequence']:.2f}"),
                        ])
                        
                        if stats.get('magnitude_stats'):
                            mag_stats = stats['magnitude_stats']
                            stats_div.children.append(
                                html.P(f"Activation magnitude - Mean: {mag_stats['mean']:.3f}, "
                                        f"Std: {mag_stats['std']:.3f}, Range: [{mag_stats['min']:.3f}, {mag_stats['max']:.3f}]")
                            )
                        
                        # Add position distribution plots if available
                        if stats.get('position_distribution'):
                            pos_dist = stats['position_distribution']
                            # Plot 1: Normalized w.r.t feature's own counts
                            total_feature_activations = sum(pos_dist)
                            if total_feature_activations > 0:
                                positions = list(range(len(pos_dist)))
                                position_counts = pos_dist  # Already a list
                                normalized_counts_1 = [count / total_feature_activations for count in position_counts]
                                
                                fig1 = go.Figure()
                                fig1.add_trace(go.Bar(
                                    x=positions,
                                    y=normalized_counts_1,
                                    name='Position Distribution',
                                    marker_color='lightgreen',
                                    text=[f"{val:.3f}<br>({pos_dist[pos]} acts)" for val, pos in zip(normalized_counts_1, positions)],
                                    textposition='auto'
                                ))
                                fig1.update_layout(
                                    title=f"Feature {feature_idx} Position Distribution",
                                    xaxis_title="Position",
                                    yaxis_title="Proportion of Feature's Activations",
                                    height=300,
                                    margin=dict(l=50, r=50, t=60, b=50)
                                )
                                
                                stats_div.children.append(
                                    dcc.Graph(
                                        figure=fig1,
                                        style={'margin': '10px 0'}
                                    )
                                )

                            # Plot 2: Position by Magnitude Distribution 
                            if stats.get('position_mag_distribution'):
                                pos_mag_dist = stats['position_mag_distribution']  # This is a list
                                total_magfeature_activations = sum(pos_mag_dist)
                                if total_magfeature_activations > 0:
                                    mag_positions = list(range(len(pos_mag_dist)))
                                    mag_counts = pos_mag_dist  # Already a list
                                    normalized_mag_counts = [count / total_magfeature_activations for count in mag_counts]
                                    
                                    fig2 = go.Figure()
                                    fig2.add_trace(go.Bar(
                                        x=mag_positions,
                                        y=normalized_mag_counts,
                                        name='Position by Magnitude',
                                        marker_color='orange',
                                        text=[f"{val:.3f}<br>({pos_mag_dist[pos]:.3f} mag)" for val, pos in zip(normalized_mag_counts, mag_positions)],
                                        textposition='auto'
                                    ))
                                    fig2.update_layout(
                                        title=f"Feature {feature_idx} Position by Magnitude Distribution",
                                        xaxis_title="Position",
                                        yaxis_title="Proportion of Feature's Total Magnitude",
                                        height=300,
                                        margin=dict(l=50, r=50, t=60, b=50)
                                    )
                                    
                                    stats_div.children.append(
                                        dcc.Graph(
                                            figure=fig2,
                                            style={'margin': '10px 0'}
                                        )
                                    )
                            
                            # Plot 3: Length Distribution (normalized by total sequences per length)
                            if stats.get('length distribution') and hasattr(self, 'all_counts'):
                                length_dist = stats['length distribution']  # This is a list
                                lengths = list(range(len(length_dist)))
                                normalized_length_counts = []
                                hover_texts = []
                                
                                for length_idx, length in enumerate(lengths):
                                    feature_count = length_dist[length_idx]
                                    total_seqs_of_length = self.all_counts.get(length, 1) if hasattr(self, 'all_counts') else 1
                                    activation_rate = feature_count / total_seqs_of_length if total_seqs_of_length > 0 else 0
                                    normalized_length_counts.append(activation_rate)
                                    hover_texts.append(f"Length {length}<br>Activation Rate: {activation_rate:.3f}<br>({feature_count}/{total_seqs_of_length} sequences)")
                                
                                fig3 = go.Figure()
                                fig3.add_trace(go.Bar(
                                    x=lengths,
                                    y=normalized_length_counts,
                                    name='Length Distribution',
                                    marker_color='lightcoral',
                                    text=[f"{val:.3f}" for val in normalized_length_counts],
                                    textposition='auto',
                                    hovertext=hover_texts,
                                    hoverinfo='text'
                                ))
                                fig3.update_layout(
                                    title=f"Feature {feature_idx} Length Distribution<br>(Normalized by Dataset Length Distribution)",
                                    xaxis_title="Sequence Length",
                                    yaxis_title="Activation Rate (Feature Activations / Total Sequences)",
                                    height=300,
                                    margin=dict(l=50, r=50, t=80, b=50)
                                )
                                
                                stats_div.children.append(
                                    dcc.Graph(
                                        figure=fig3,
                                        style={'margin': '10px 0'}
                                    )
                                )
                                
                    # except Exception as e:
                    #     stats_div.children.append(html.P(f"Error getting detailed stats: {str(e)}", style={'color': 'red'}))
                
                return stats_div
                        
            except Exception as e:
                print(f"Error in update_feature_stats: {e}")
                traceback.print_exc()
                return html.Div(f"Error updating stats: {str(e)}")
            
        @self.app.callback(
            Output('sequence-display', 'children'),
            [Input('transcoder-type', 'value'),
             Input('feature-dropdown', 'value'),
             Input('sort-option', 'value'),
             Input('category-option', 'value'),
             Input('include-inactive', 'value')]
        )
        def update_sequence_display(transcoder_type, feature_idx, sort_option, category_option, 
                               include_inactive, ranking_mode, length_filter):
        """Update main sequence display with ranking options"""
        try:
            print(f"=== DEBUG: update_sequence_display called ===")
            print(f"transcoder_type: {transcoder_type}, feature_idx: {feature_idx}")
            print(f"sort_option: {sort_option}, category_option: {category_option}")
            print(f"include_inactive: {include_inactive}")
            print(f"ranking_mode: {ranking_mode}, length_filter: {length_filter}")  # NEW
            
            if feature_idx is None:
                return [html.Div("No feature selected")]
            
            # Get active sequences data
            active_sequences_data = []
            if (hasattr(self.analyzer, 'feature_activations') and 
                isinstance(self.analyzer.feature_activations, dict) and
                transcoder_type in self.analyzer.feature_activations and
                feature_idx in self.analyzer.feature_activations[transcoder_type]):
                
                feature_data = self.analyzer.feature_activations[transcoder_type][feature_idx]
                
                for seq_tuple, activations_list in feature_data.items():
                    try:
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
                            continue
                        
                        # Convert seq_tuple to sequence
                        if isinstance(seq_tuple, tuple):
                            sequence = list(seq_tuple)
                        elif isinstance(seq_tuple, str):
                            sequence = [seq_tuple]
                        else:
                            sequence = [str(seq_tuple)]
                        
                        active_sequences_data.append({
                            'sequence': sequence,
                            'positions': all_positions,
                            'magnitudes': all_magnitudes,
                            'is_active': True
                        })
                        
                    except Exception as e:
                        print(f"Error processing active sequence {seq_tuple}: {e}")
                        continue
            
            # Get inactive sequences data if requested
            inactive_sequences_data = []
            if 'include' in include_inactive and category_option == 'worst':
                inactive_sequences_data = self._get_inactive_sequences_for_feature(transcoder_type, feature_idx)
            
            # Combine active and inactive sequences
            all_sequences_data = active_sequences_data + inactive_sequences_data
            
            print(f"Active sequences: {len(active_sequences_data)}")
            print(f"Inactive sequences: {len(inactive_sequences_data)}")
            print(f"Total sequences: {len(all_sequences_data)}")
            
            if not all_sequences_data:
                return [html.Div("No sequences found for this feature.")]
            
            # NEW: Apply length filtering if in length-specific mode
            if ranking_mode == 'length_specific' and length_filter is not None:
                all_sequences_data = [
                    seq_data for seq_data in all_sequences_data 
                    if len(seq_data.get('sequence', [])) == length_filter
                ]
                print(f"After length filtering (length={length_filter}): {len(all_sequences_data)} sequences")
                
                if not all_sequences_data:
                    return [html.Div(f"No sequences of length {length_filter} found for this feature.")]
            
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
            
            # Sort and select sequences
            if category_option == 'best':
                # For best: only show active sequences, sorted by highest activation
                sequences_to_show = [seq for seq in all_sequences_data if seq.get('is_active', True)]
                sequences_to_show.sort(key=lambda x: get_sort_key(x, sort_option), reverse=True)
            else:  # worst
                # For worst: show all sequences (active + inactive), sorted by lowest activation
                sequences_to_show = all_sequences_data
                sequences_to_show.sort(key=lambda x: get_sort_key(x, sort_option), reverse=False)
            
            # Take top 20
            sequences_to_show = sequences_to_show[:20]
            
            print(f"Showing {len(sequences_to_show)} sequences")
            
            # Add ranking mode info to the display
            ranking_info = ""
            if ranking_mode == 'length_specific' and length_filter is not None:
                ranking_info = f" (Length {length_filter} only)"
            
            result = [html.Div([
                html.H4(f"Sequences for Feature {feature_idx}{ranking_info}", 
                       style={'margin-bottom': '15px'}),
                *self._create_sequence_display(sequences_to_show, feature_idx)
            ])]
            
            return result
            
        except Exception as e:
            error_msg = f"Error in update_sequence_display: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return [html.Div(error_msg)]
    
    def run(self, host='0.0.0.0', port=8050, debug=True):
        """Run the Dash app"""
        print(f"Starting enhanced interactive visualizer at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

def launch_enhanced_visualizer(analyzer, all_sequences=None, 
                               features_hidden=None, features_update=None):
    """
    Launch the enhanced visualizer with inactive sequence support
    
    Args:
        analyzer: CopyFeatureActivationAnalyzer instance
        all_sequences: Set/List of all sequences that were analyzed (tuples)
        total_features: Total number of features in the model
    """
    visualizer = EnhancedInteractiveFeatureVisualizer(analyzer, all_sequences, 
                                                      features_hidden, features_update)
    visualizer.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dict_path", required=True)
    parser.add_argument("--features_hidden", type=int, required=True)
    parser.add_argument("--features_update", type=int, required=True)

    args = parser.parse_args()
    
    # Load feature activations
    with open(args.feature_dict_path, "rb") as f:
        analysis_dict = pickle.load(f)
    
    # Load all sequences if provided
    all_sequences = []
    for transcoder_type in analysis_dict:
        for feature in analysis_dict[transcoder_type]:
            sequences = analysis_dict[transcoder_type][feature].keys()
            all_sequences.extend(list(sequences))
    
    # Create analyzer
    analyzer = CopyFeatureActivationAnalyzer(None, None, None, "cpu")
    analyzer.feature_activations = analysis_dict
    
    launch_enhanced_visualizer(analyzer, all_sequences, args.features_hidden,
                               args.features_update)