#!/usr/bin/env python3
"""
Visualization script for acoustic scan data pickle files using Plotly.
Visualizes voltage measurements from transmission and echo scans (forward and reverse).
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

try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


def load_scan_data(pickle_path):
    """Load scan data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract numeric keys (scan indices)
    numeric_keys = sorted([k for k in data.keys() if isinstance(k, (int, np.integer))])
    
    print(f"Loaded {len(numeric_keys)} scans from {pickle_path}")
    print(f"File name: {data.get('fileName', 'N/A')}")
    
    return data, numeric_keys

def plot_single_scan(data, scan_idx, show_metadata=True):
    """Plot a single scan showing all voltage measurements using Plotly."""
    scan_data = data[scan_idx]
    time = scan_data.get('time', np.arange(len(scan_data['voltage_transmission_forward'])))
    
    fig = go.Figure()
    
    # Add all four voltage measurements
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_transmission_forward'],
        mode='lines', name='Transmission Forward',
        line=dict(color='blue', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_echo_forward'],
        mode='lines', name='Echo Forward',
        line=dict(color='green', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_transmission_reverse'],
        mode='lines', name='Transmission Reverse',
        line=dict(color='red', width=1.5, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_echo_reverse'],
        mode='lines', name='Echo Reverse',
        line=dict(color='orange', width=1.5, dash='dot')
    ))
    
    # Add metadata as annotation if requested
    if show_metadata:
        metadata_text = (
            f"Offset Forward: {scan_data.get('voltageOffsetForward', 'N/A'):.4f}<br>"
            f"Gain Forward: {scan_data.get('gainForward', 'N/A'):.4f}<br>"
            f"Offset Reverse: {scan_data.get('voltageOffsetReverse', 'N/A'):.4f}<br>"
            f"Gain Reverse: {scan_data.get('gainReverse', 'N/A'):.4f}"
        )
        fig.add_annotation(
            text=metadata_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 224, 0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title=f'Scan {scan_idx} - All Voltage Measurements',
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
    
    return fig


def plot_comparison(data, scan_idx1, scan_idx2):
    """Compare two scans side by side using Plotly."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Scan {scan_idx1}', f'Scan {scan_idx2}'),
        shared_yaxes=True
    )
    
    for col_idx, scan_idx in enumerate([scan_idx1, scan_idx2], 1):
        scan_data = data[scan_idx]
        time = scan_data.get('time', np.arange(len(scan_data['voltage_transmission_forward'])))
        
        fig.add_trace(go.Scatter(
            x=time, y=scan_data['voltage_transmission_forward'],
            mode='lines', name='Trans Forward', showlegend=(col_idx == 1),
            line=dict(color='blue', width=1.5)
        ), row=1, col=col_idx)
        
        fig.add_trace(go.Scatter(
            x=time, y=scan_data['voltage_echo_forward'],
            mode='lines', name='Echo Forward', showlegend=(col_idx == 1),
            line=dict(color='green', width=1.5)
        ), row=1, col=col_idx)
        
        fig.add_trace(go.Scatter(
            x=time, y=scan_data['voltage_transmission_reverse'],
            mode='lines', name='Trans Reverse', showlegend=(col_idx == 1),
            line=dict(color='red', width=1.5, dash='dot')
        ), row=1, col=col_idx)
        
        fig.add_trace(go.Scatter(
            x=time, y=scan_data['voltage_echo_reverse'],
            mode='lines', name='Echo Reverse', showlegend=(col_idx == 1),
            line=dict(color='orange', width=1.5, dash='dot')
        ), row=1, col=col_idx)
    
    fig.update_layout(
        title=f'Comparison: Scan {scan_idx1} vs Scan {scan_idx2}',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Voltage", row=1, col=1)
    
    return fig


def plot_overview(data, numeric_keys, max_scans=50):
    """Plot overview of multiple scans using Plotly."""
    n_scans = min(len(numeric_keys), max_scans)
    selected_keys = numeric_keys[::max(1, len(numeric_keys)//n_scans)][:n_scans]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Transmission Forward', 'Echo Forward',
                       'Transmission Reverse', 'Echo Reverse'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    labels = ['Transmission Forward', 'Echo Forward', 
              'Transmission Reverse', 'Echo Reverse']
    keys_list = ['voltage_transmission_forward', 'voltage_echo_forward',
                 'voltage_transmission_reverse', 'voltage_echo_reverse']
    
    for idx, (label, key) in enumerate(zip(labels, keys_list)):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        for scan_idx in selected_keys:
            scan_data = data[scan_idx]
            time = scan_data.get('time', np.arange(len(scan_data[key])))
            
            fig.add_trace(go.Scatter(
                x=time, y=scan_data[key],
                mode='lines', name=f'Scan {scan_idx}',
                showlegend=False,
                line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
                hovertemplate=f'Scan {scan_idx}<br>Time: %{{x}}<br>Voltage: %{{y}}<extra></extra>'
            ), row=row, col=col)
    
    fig.update_layout(
        title=f'Overview - {n_scans} Scans Overlay',
        height=800,
        template='plotly_white',
        hovermode='closest'
    )
    
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Time", row=i, col=j)
            fig.update_yaxes(title_text="Voltage", row=i, col=j)
    
    return fig


def interactive_scan_viewer(data, numeric_keys, use_ipywidgets=True):
    """
    Create an interactive viewer with slider to browse through scans using Plotly.
    
    Args:
        data: Dictionary containing scan data
        numeric_keys: List of scan indices
        use_ipywidgets: If True and ipywidgets is available, use ipywidgets for faster loading.
                       If False, use Plotly frames (slower but works outside Jupyter)
    
    Returns:
        If use_ipywidgets=True and ipywidgets available: Returns (widget, fig) tuple
        Otherwise: Returns Plotly figure with frames
    """
    # Use ipywidgets for faster loading in Jupyter
    if use_ipywidgets and IPYWIDGETS_AVAILABLE:
        return interactive_scan_viewer_ipywidgets(data, numeric_keys)
    
    # Fallback to frames-based approach (slower but works everywhere)
    return interactive_scan_viewer_frames(data, numeric_keys)


def interactive_scan_viewer_ipywidgets(data, numeric_keys):
    """Fast interactive viewer using ipywidgets (for Jupyter notebooks)."""
    # Create initial data for first scan
    initial_idx = 0
    scan_data = data[numeric_keys[initial_idx]]
    time = scan_data.get('time', np.arange(len(scan_data['voltage_transmission_forward'])))
    
    # Create Plotly figure
    fig = go.FigureWidget()
    
    # Add traces for all four measurements
    trace1 = fig.add_scatter(
        x=time, y=scan_data['voltage_transmission_forward'],
        mode='lines', name='Transmission Forward',
        line=dict(color='blue', width=2)
    )
    trace2 = fig.add_scatter(
        x=time, y=scan_data['voltage_echo_forward'],
        mode='lines', name='Echo Forward',
        line=dict(color='green', width=2)
    )
    trace3 = fig.add_scatter(
        x=time, y=scan_data['voltage_transmission_reverse'],
        mode='lines', name='Transmission Reverse',
        line=dict(color='red', width=2, dash='dot')
    )
    trace4 = fig.add_scatter(
        x=time, y=scan_data['voltage_echo_reverse'],
        mode='lines', name='Echo Reverse',
        line=dict(color='orange', width=2, dash='dot')
    )
    
    fig.update_layout(
        title='Interactive Scan Viewer - Use slider to browse scans',
        xaxis_title='Time',
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
    
    # Create slider widget
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(numeric_keys) - 1,
        step=1,
        description='Scan:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='600px')
    )
    
    # Update function
    def update_plot(change):
        idx = slider.value
        scan_idx = numeric_keys[idx]
        scan_data = data[scan_idx]
        time = scan_data.get('time', np.arange(len(scan_data['voltage_transmission_forward'])))
        
        # Convert numpy arrays to lists if needed (FigureWidget sometimes needs lists)
        def to_list(arr):
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return list(arr) if not isinstance(arr, list) else arr
        
        time_list = to_list(time)
        vtf_list = to_list(scan_data['voltage_transmission_forward'])
        vef_list = to_list(scan_data['voltage_echo_forward'])
        vtr_list = to_list(scan_data['voltage_transmission_reverse'])
        ver_list = to_list(scan_data['voltage_echo_reverse'])
        
        # Update traces - access via fig.data for reliability
        with fig.batch_update():
            fig.data[0].x = time_list
            fig.data[0].y = vtf_list
            fig.data[1].x = time_list
            fig.data[1].y = vef_list
            fig.data[2].x = time_list
            fig.data[2].y = vtr_list
            fig.data[3].x = time_list
            fig.data[3].y = ver_list
            fig.layout.title.text = f'Scan {scan_idx} - Interactive Viewer'
    
    slider.observe(update_plot, names='value')
    
    # Create container with slider and figure
    container = widgets.VBox([slider, fig])
    return container


def interactive_scan_viewer_frames(data, numeric_keys):
    """Slower frames-based viewer (works outside Jupyter, but loads all data upfront)."""
    # Create initial data for first scan
    initial_idx = 0
    scan_data = data[numeric_keys[initial_idx]]
    time = scan_data.get('time', np.arange(len(scan_data['voltage_transmission_forward'])))
    
    fig = go.Figure()
    
    # Add traces for all four measurements
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_transmission_forward'],
        mode='lines', name='Transmission Forward',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_echo_forward'],
        mode='lines', name='Echo Forward',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_transmission_reverse'],
        mode='lines', name='Transmission Reverse',
        line=dict(color='red', width=2, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=time, y=scan_data['voltage_echo_reverse'],
        mode='lines', name='Echo Reverse',
        line=dict(color='orange', width=2, dash='dot')
    ))
    
    # Create frames for animation (one frame per scan) - this is slow!
    print("Creating frames (this may take a while for large datasets)...")
    frames = []
    for idx, scan_idx in enumerate(numeric_keys):
        scan_data = data[scan_idx]
        time = scan_data.get('time', np.arange(len(scan_data['voltage_transmission_forward'])))
        
        frame = go.Frame(
            data=[
                go.Scatter(x=time, y=scan_data['voltage_transmission_forward']),
                go.Scatter(x=time, y=scan_data['voltage_echo_forward']),
                go.Scatter(x=time, y=scan_data['voltage_transmission_reverse'],
                          line=dict(dash='dot')),
                go.Scatter(x=time, y=scan_data['voltage_echo_reverse'],
                          line=dict(dash='dot'))
            ],
            name=str(idx)
        )
        frames.append(frame)
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(numeric_keys)} scans...")
    
    fig.frames = frames
    
    # Create slider
    steps = []
    for idx, scan_idx in enumerate(numeric_keys):
        step = dict(
            method='animate',
            args=[[str(idx)],
                   dict(frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0))],
            label=f'Scan {scan_idx}'
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Scan Index: "},
        pad={"t": 50},
        steps=steps,
        len=0.9,
        x=0.05,
        xanchor="left",
        y=0,
        yanchor="top"
    )]
    
    fig.update_layout(
        title='Interactive Scan Viewer - Use slider to browse scans',
        xaxis_title='Time',
        yaxis_title='Voltage',
        hovermode='x unified',
        height=600,
        template='plotly_white',
        sliders=sliders,
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=100, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=0))]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate',
                                           transition=dict(duration=0))])],
            x=0.1, xanchor='left',
            y=0, yanchor='top',
            pad=dict(t=50, r=10)
        )]
    )
    
    return fig


def plot_statistics(data, numeric_keys):
    """Plot statistics across all scans using Plotly."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Transmission Forward', 'Echo Forward',
                       'Transmission Reverse', 'Echo Reverse'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    keys_list = ['voltage_transmission_forward', 'voltage_echo_forward',
                 'voltage_transmission_reverse', 'voltage_echo_reverse']
    labels = ['Transmission Forward', 'Echo Forward', 
              'Transmission Reverse', 'Echo Reverse']
    
    scan_indices = np.arange(len(numeric_keys))
    
    for idx, (key, label) in enumerate(zip(keys_list, labels)):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        means = []
        stds = []
        mins = []
        maxs = []
        
        for scan_idx in numeric_keys:
            scan_data = data[scan_idx]
            values = scan_data[key]
            means.append(np.mean(values))
            stds.append(np.std(values))
            mins.append(np.min(values))
            maxs.append(np.max(values))
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Mean line
        fig.add_trace(go.Scatter(
            x=scan_indices, y=means,
            mode='lines', name='Mean', showlegend=(idx == 0),
            line=dict(color='blue', width=2)
        ), row=row, col=col)
        
        # Std deviation fill
        fig.add_trace(go.Scatter(
            x=np.concatenate([scan_indices, scan_indices[::-1]]),
            y=np.concatenate([means + stds, (means - stds)[::-1]]),
            fill='toself', fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std', showlegend=(idx == 0),
            hoverinfo='skip'
        ), row=row, col=col)
        
        # Min line
        fig.add_trace(go.Scatter(
            x=scan_indices, y=mins,
            mode='lines', name='Min', showlegend=(idx == 0),
            line=dict(color='red', width=1, dash='dash')
        ), row=row, col=col)
        
        # Max line
        fig.add_trace(go.Scatter(
            x=scan_indices, y=maxs,
            mode='lines', name='Max', showlegend=(idx == 0),
            line=dict(color='green', width=1, dash='dash')
        ), row=row, col=col)
    
    fig.update_layout(
        title='Statistics Across All Scans',
        height=800,
        template='plotly_white',
        hovermode='x unified'
    )
    
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Scan Index", row=i, col=j)
            fig.update_yaxes(title_text="Voltage", row=i, col=j)
    
    return fig


def create_jupyter_viewer(pickle_path, use_ipywidgets=True):
    """
    Main function to create an interactive Plotly viewer for Jupyter notebooks.
    Returns a widget/figure with slider that can be displayed directly in a notebook.
    
    Args:
        pickle_path: Path to pickle file
        use_ipywidgets: If True (default), use ipywidgets for fast loading.
                       If False, use Plotly frames (slower but works everywhere)
    
    Usage in Jupyter:
        from visualize_scan_data import create_jupyter_viewer
        viewer = create_jupyter_viewer('path/to/file.pickle')
        display(viewer)  # or just: viewer  (in Jupyter)
    """
    data, numeric_keys = load_scan_data(pickle_path)
    return interactive_scan_viewer(data, numeric_keys, use_ipywidgets=use_ipywidgets)


def main():
    parser = argparse.ArgumentParser(description='Visualize acoustic scan data from pickle file using Plotly')
    parser.add_argument('pickle_file', type=str, 
                       help='Path to pickle file containing scan data')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['single', 'comparison', 'overview', 'interactive', 'stats', 'all'],
                       help='Visualization mode')
    parser.add_argument('--scan', type=int, default=0,
                       help='Scan index to visualize (for single mode)')
    parser.add_argument('--scan1', type=int, default=0,
                       help='First scan index (for comparison mode)')
    parser.add_argument('--scan2', type=int, default=1,
                       help='Second scan index (for comparison mode)')
    parser.add_argument('--max-scans', type=int, default=50,
                       help='Maximum number of scans to show in overview')
    parser.add_argument('--output', type=str, default=None,
                       help='Output HTML file path (optional)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        data, numeric_keys = load_scan_data(args.pickle_file)
    except Exception as e:
        print(f"Error loading pickle file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate scan indices
    if args.scan not in numeric_keys:
        print(f"Warning: Scan {args.scan} not found. Using scan {numeric_keys[0]} instead.")
        args.scan = numeric_keys[0]
    if args.scan1 not in numeric_keys:
        args.scan1 = numeric_keys[0]
    if args.scan2 not in numeric_keys:
        args.scan2 = numeric_keys[min(1, len(numeric_keys)-1)]
    
    # Create visualizations based on mode
    if args.mode == 'single' or args.mode == 'all':
        print(f"\nPlotting single scan {args.scan}...")
        fig = plot_single_scan(data, args.scan)
        if args.output:
            fig.write_html(args.output.replace('.html', '_single.html'))
        fig.show()
    
    if args.mode == 'comparison' or args.mode == 'all':
        print(f"\nPlotting comparison: scan {args.scan1} vs {args.scan2}...")
        fig = plot_comparison(data, args.scan1, args.scan2)
        if args.output:
            fig.write_html(args.output.replace('.html', '_comparison.html'))
        fig.show()
    
    if args.mode == 'overview' or args.mode == 'all':
        print(f"\nPlotting overview of scans...")
        fig = plot_overview(data, numeric_keys, max_scans=args.max_scans)
        if args.output:
            fig.write_html(args.output.replace('.html', '_overview.html'))
        fig.show()
    
    if args.mode == 'stats' or args.mode == 'all':
        print(f"\nPlotting statistics across all scans...")
        fig = plot_statistics(data, numeric_keys)
        if args.output:
            fig.write_html(args.output.replace('.html', '_stats.html'))
        fig.show()
    
    if args.mode == 'interactive' or args.mode == 'all':
        print(f"\nCreating interactive viewer with slider...")
        print("Use the slider to browse through scans.")
        # For command line, use frames (ipywidgets won't work)
        fig = interactive_scan_viewer(data, numeric_keys, use_ipywidgets=False)
        if args.output:
            fig.write_html(args.output if args.output.endswith('.html') else args.output + '.html')
        fig.show()


if __name__ == '__main__':
    main()
