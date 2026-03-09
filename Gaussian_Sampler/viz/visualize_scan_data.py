#!/usr/bin/env python3
"""
Visualization script for acoustic scan data pickle files using Plotly.
Designed for use in Jupyter notebooks with interactive sliders.
"""

import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import argparse
import sys
import pprint

def load_scan_data(pickle_path):
    """Load scan data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract numeric keys (scan indices)
    numeric_keys = sorted([k for k in data.keys() if isinstance(k, (int, np.integer))])
    
    print(f"Loaded {len(numeric_keys)} scans from {pickle_path}")
    print(f"File name: {data.get('fileName', 'N/A')}")
    
    return data, numeric_keys

try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

def plotly_viewer(dataset):
    """Interactive plot viewer that updates automatically as the slider changes."""
    if not IPYWIDGETS_AVAILABLE:
        raise ImportError("ipywidgets is required for interactive plotting. Please install it: pip install ipywidgets")
    
    # Create slider
    i_slider = widgets.IntSlider(
        description='Scan #', 
        min=0, 
        max=len(dataset.numeric_keys)-1, 
        step=1, 
        value=0
    )
    
    # Get initial data
    idx, signal = dataset[i_slider.value]
    time = dataset.data[idx]['time']
    
    # Create FigureWidget (not Figure) so it can be updated interactively
    fig = go.FigureWidget()
    
    # Add initial trace
    fig.add_trace(go.Scatter(
        x=time, y=signal,
        mode='lines', name=dataset.dset_name,
        line=dict(color='blue', width=1.5)
    ))
    
    # Set up layout
    fig.update_layout(
        title=f'Scan {idx} - {dataset.dset_name}',
        xaxis_title='Time (samples or time units)',
        yaxis_title='Voltage',
        hovermode='x unified',
        height=600,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=1.01,
            orientation="v"
        ),
        margin=dict(r=150)  # Add right margin for legend
    )
    
    # Update function that gets called when slider changes
    def update_plot(change):
        # Get new data based on slider value
        idx, signal = dataset[i_slider.value]
        time = dataset.data[idx]['time']
        
        # Convert to lists if needed (FigureWidget sometimes needs lists)
        def to_list(arr):
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return list(arr) if not isinstance(arr, list) else arr
        
        time_list = to_list(time)
        signal_list = to_list(signal)
        
        # Update the figure with new data
        with fig.batch_update():
            fig.data[0].x = time_list
            fig.data[0].y = signal_list
            fig.layout.title.text = f'Scan {idx} - {dataset.dset_name}'
    
    # Connect the slider to the update function
    i_slider.observe(update_plot, names='value')
    
    # Create container with slider and figure
    container = widgets.VBox([i_slider, fig])
    return container
