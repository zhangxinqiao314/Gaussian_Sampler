from functools import lru_cache
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display


class Poisson_Sampled_PV_viz:
    '''Interactive visualization using Plotly and ipywidgets'''
    
    def __init__(self, dset, sampler=None):
        self.dset = dset
        self.dset_list = dset.h5_keys()
        self.sampler = sampler
        self.colors = ['green', 'orange', 'yellow', 'brown', 'pink', 'gray', 
                       'magenta', 'cyan', 'purple', 'lime', 'teal', 'maroon', 'indigo', 'gold']

        # Create interactive widgets
        self.i_slider = widgets.IntSlider(description='Sampled rate', value=0, min=0, max=len(self.dset_list)-1)
        self.s_slider = widgets.IntSlider(description='spectral', value=0, min=0, max=dset.shape[2]-1)
        
        # x and y are now set by clicking on the plot
        self.x = 25
        self.y = 25

        # Create FigureWidgets for interactive updates
        self.img_fig = go.FigureWidget()
        self.spec_fig = go.FigureWidget()

        if sampler is not None:
            self.batch_checkboxes = widgets.SelectMultiple(
                options=list(range(int(np.ceil(self.sampler.batch_size / self.sampler.num_neighbors)))),
                value=tuple(range(min(10, int(np.ceil(self.sampler.batch_size / self.sampler.num_neighbors))))),
                description='Batches'
            )
            self.batch_inds = next(iter(self.sampler))
            self.new_batch_button = widgets.Button(description='New Batch')
            self.new_batch_button.on_click(self._new_batch)
            # Separate coordinates for batch visualization
            self.batch_x = 25
            self.batch_y = 25

    @lru_cache(maxsize=10)
    def select_datacube(self, i):
        self.dset.dset_name = i
        return self.dset[:][1]

    @lru_cache(maxsize=10)
    def select_zero_datacube(self, i):
        self.dset.dset_name = i
        return self.dset.getitem_zero_dset(slice(0, self.dset.shape[0] * self.dset.shape[1]))[1]

    @lru_cache(maxsize=32)
    def datacube_max(self, i):
        return float(self.select_datacube(i).max())

    ############################################ Plotting functions

    def plot_datacube_img(self, i, s, x, y):
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        data_ = datacube[:, :, s]

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=data_, colorscale='Viridis', zmin=0, zmax=self.datacube_max(i),
            colorbar=dict(title='Intensity')
        ))
        # Add red dot for selected position
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(color='red', size=10), name='Selected'
        ))
        fig.update_layout(
            title=f'Sampled rate: {self.dset_list[i]}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=400, height=400
        )
        return fig

    def plot_datacube_spectrum(self, i, x, y, s):
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        spectrum = datacube[y, x]
        zero_spectrum = zero_datacube[y, x]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=spectrum, mode='lines', name='Noisy', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=zero_spectrum, mode='lines', name='Clean', line=dict(color='green')))
        # Vertical line at spectral position
        fig.add_vline(x=s, line=dict(color='black', width=2))
        fig.update_layout(
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, self.datacube_max(i)]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=400, height=400
        )
        return fig

    def _update_plots(self, change=None):
        i, s = self.i_slider.value, self.s_slider.value
        x, y = self.x, self.y
        
        # Update image
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        data_ = datacube[:, :, s]
        with self.img_fig.batch_update():
            self.img_fig.data[0].z = data_
            self.img_fig.data[0].zmax = self.datacube_max(i)
            self.img_fig.data[1].x = [x]
            self.img_fig.data[1].y = [y]
            self.img_fig.layout.title.text = f'Sampled rate: {self.dset_list[i]}'

        # Update spectrum
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        with self.spec_fig.batch_update():
            self.spec_fig.data[0].y = datacube[y, x]
            self.spec_fig.data[1].y = zero_datacube[y, x]
            self.spec_fig.layout.shapes[0].x0 = s
            self.spec_fig.layout.shapes[0].x1 = s
            self.spec_fig.layout.yaxis.range = [0, self.datacube_max(i)]

    def _handle_click(self, trace, points, selector):
        """Handle click on image plot to set x and y coordinates"""
        if points.xs and points.ys:
            # For heatmap, get the actual data coordinates
            x_clicked = points.xs[0]
            y_clicked = points.ys[0]
            # Round to nearest integer for array indexing
            self.x = int(round(x_clicked))
            self.y = int(round(y_clicked))
            # Clamp to valid range
            self.x = max(0, min(self.x, self.dset.shape[0] - 1))
            self.y = max(0, min(self.y, self.dset.shape[1] - 1))
            # Update plots
            self._update_plots()
    
    def _handle_spectrum_click(self, trace, points, selector):
        """Handle click on spectrum plot to set spectral index"""
        if points.xs:
            # Get the x-coordinate (spectral index)
            s_clicked = points.xs[0]
            # Round to nearest integer and clamp to valid range
            s_new = int(round(s_clicked))
            s_new = max(0, min(s_new, self.dset.shape[2] - 1))
            # Update the slider value, which will trigger _update_plots
            self.s_slider.value = s_new
    
    def _init_figures(self):
        i, s = self.i_slider.value, self.s_slider.value
        x, y = self.x, self.y
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        data_ = datacube[:, :, s]

        # Image figure
        self.img_fig.add_trace(go.Heatmap(
            z=data_, colorscale='Viridis', zmin=0, zmax=self.datacube_max(i),
            colorbar=dict(title='Intensity')
        ))
        self.img_fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(color='red', size=10), name='Selected'
        ))
        # Add click handler to the heatmap
        self.img_fig.data[0].on_click(self._handle_click)
        self.img_fig.update_layout(
            title=f'Sampled rate: {self.dset_list[i]}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=400, height=400
        )

        # Spectrum figure
        self.spec_fig.add_trace(go.Scatter(y=datacube[y, x], mode='lines', name='Noisy', line=dict(color='blue')))
        self.spec_fig.add_trace(go.Scatter(y=zero_datacube[y, x], mode='lines', name='Clean', line=dict(color='green')))
        self.spec_fig.add_vline(x=s, line=dict(color='black', width=2))
        # Add click handler to spectrum traces
        self.spec_fig.data[0].on_click(self._handle_spectrum_click)
        self.spec_fig.data[1].on_click(self._handle_spectrum_click)
        self.spec_fig.update_layout(
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, self.datacube_max(i)]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=400, height=400
        )

    def layout_input(self):
        self._init_figures()
        
        # Connect widgets to update function (only i and s sliders now)
        for slider in [self.i_slider, self.s_slider]:
            slider.observe(self._update_plots, names='value')

        sliders = widgets.VBox([
            widgets.HBox([self.i_slider, self.s_slider]),
            widgets.HTML(value='<i>Click on the image to select x, y coordinates</i>')
        ])
        plots = widgets.HBox([self.img_fig, self.spec_fig])
        return widgets.VBox([sliders, plots])

    ############################################ Batch helpers

    def split_list(self):
        return [self.batch_inds[i:i + self.sampler.num_neighbors] 
                for i in range(0, len(self.batch_inds), self.sampler.num_neighbors)]

    def get_points_idx(self):
        clumps = self.split_list()
        return [[(ind % self.dset.shape[0], ind // self.dset.shape[0]) for ind in clump] 
                for clump in clumps]

    def get_points_data(self, i):
        dset = self.select_datacube(i)
        clumps = self.split_list()
        return [np.asarray([dset[ind] for ind in clump], dtype=np.float32) for clump in clumps]

    def _new_batch(self, b):
        self.batch_inds = next(iter(self.sampler))
        self._update_batch_plots()
    
    def _handle_batch_click(self, trace, points, selector):
        """Handle click on batch image plot to set batch x and y coordinates"""
        if points.xs and points.ys:
            # For heatmap, get the actual data coordinates
            x_clicked = points.xs[0]
            y_clicked = points.ys[0]
            # Round to nearest integer for array indexing
            self.batch_x = int(round(x_clicked))
            self.batch_y = int(round(y_clicked))
            # Clamp to valid range
            self.batch_x = max(0, min(self.batch_x, self.dset.shape[1] - 1))
            self.batch_y = max(0, min(self.batch_y, self.dset.shape[0] - 1))
            # Update batch plots
            self._update_batch_plots()

    ############################################ Batch plotting

    def plot_batch_points(self, i, s, checked):
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        data_ = datacube[:, :, s]
        pts = self.get_points_idx()

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=data_, colorscale='Viridis', zmin=0, zmax=self.datacube_max(i),
            colorbar=dict(title='Intensity')
        ))
        for p, pt in enumerate(pts):
            xs, ys = zip(*pt) if pt else ([], [])
            alpha = 1.0 if p in checked else 0.2
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(color=self.colors[p % len(self.colors)], size=6, opacity=alpha),
                name=f'Batch {p}'
            ))
        fig.update_layout(
            title=f'Sampled rate: {self.dset_list[i]}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=450, height=450
        )
        return fig

    def plot_batch_spectrum(self, i, checked):
        data = self.get_points_data(i)
        fig = go.Figure()
        for d, dat in enumerate(data):
            alpha = 1.0 if d in checked else 0.2
            fig.add_trace(go.Scatter(
                y=dat.mean(axis=0), mode='lines',
                line=dict(color=self.colors[d % len(self.colors)], width=1),
                opacity=alpha, name=f'Batch {d}'
            ))
        fig.update_layout(
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, max(self.dset.maxes)]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=450, height=450
        )
        return fig

    def _update_batch_plots(self, change=None):
        i = self.i_slider.value
        s = self.s_slider.value
        checked = list(self.batch_checkboxes.value)
        
        # Update batch image
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        data_ = datacube[:, :, s]
        pts = self.get_points_idx()
        
        with self.batch_img_fig.batch_update():
            self.batch_img_fig.data[0].z = data_
            self.batch_img_fig.data[0].zmax = self.datacube_max(i)
            for p, pt in enumerate(pts):
                if p + 1 < len(self.batch_img_fig.data) - 1:  # -1 for clicked point marker
                    xs, ys = zip(*pt) if pt else ([], [])
                    self.batch_img_fig.data[p + 1].x = xs
                    self.batch_img_fig.data[p + 1].y = ys
                    self.batch_img_fig.data[p + 1].opacity = 1.0 if p in checked else 0.2
            # Update clicked point marker (last trace)
            if len(self.batch_img_fig.data) > len(pts) + 1:
                self.batch_img_fig.data[-1].x = [self.batch_x]
                self.batch_img_fig.data[-1].y = [self.batch_y]

        # Update batch spectrum
        data = self.get_points_data(i)
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        clicked_zero_spectrum = zero_datacube[self.batch_y, self.batch_x]
        
        with self.batch_spec_fig.batch_update():
            num_batches = len(data)
            for d, dat in enumerate(data):
                if d < num_batches:
                    self.batch_spec_fig.data[d].y = dat.mean(axis=0)
                    self.batch_spec_fig.data[d].opacity = 1.0 if d in checked else 0.2
            # Update clicked point spectrum (clean only, last trace)
            if len(self.batch_spec_fig.data) >= num_batches + 1:
                self.batch_spec_fig.data[num_batches].y = clicked_zero_spectrum
            # Update vertical line position
            if self.batch_spec_fig.layout.shapes:
                self.batch_spec_fig.layout.shapes[0].x0 = s
                self.batch_spec_fig.layout.shapes[0].x1 = s

    def _init_batch_figures(self):
        # Initialize batch coordinates if they don't exist
        if not hasattr(self, 'batch_x'):
            self.batch_x = 25
        if not hasattr(self, 'batch_y'):
            self.batch_y = 25
            
        i = self.i_slider.value
        s = self.s_slider.value
        checked = list(self.batch_checkboxes.value)
        
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        data_ = datacube[:, :, s]
        pts = self.get_points_idx()

        self.batch_img_fig = go.FigureWidget()
        self.batch_img_fig.add_trace(go.Heatmap(
            z=data_, colorscale='Viridis', zmin=0, zmax=self.datacube_max(i),
            colorbar=dict(title='Intensity')
        ))
        for p, pt in enumerate(pts):
            xs, ys = zip(*pt) if pt else ([], [])
            alpha = 1.0 if p in checked else 0.2
            scatter_trace = go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(color=self.colors[p % len(self.colors)], size=6),
                opacity=alpha, name=f'{p}'
            )
            self.batch_img_fig.add_trace(scatter_trace)
            # Add click handler to each batch point scatter trace
            self.batch_img_fig.data[-1].on_click(self._handle_batch_click)
        # Add red marker for clicked point
        self.batch_img_fig.add_trace(go.Scatter(
            x=[self.batch_x], y=[self.batch_y], mode='markers',
            marker=dict(color='red', size=10), name='Select'
        ))
        # Add click handler to the heatmap and red marker
        self.batch_img_fig.data[0].on_click(self._handle_batch_click)
        self.batch_img_fig.data[-1].on_click(self._handle_batch_click)
        self.batch_img_fig.update_layout(
            title=f'Sampled rate: {self.dset_list[i]}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=450, height=450,
            legend=dict[str, str | float | dict[str, int]](orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, entrywidth=0.15, entrywidthmode='fraction', 
                        font=dict(size=10)  )
        )

        # Batch spectrum
        self.batch_spec_fig = go.FigureWidget()
        data = self.get_points_data(i)
        for d, dat in enumerate(data):
            alpha = 1.0 if d in checked else 0.2
            self.batch_spec_fig.add_trace(go.Scatter(
                y=dat.mean(axis=0), mode='lines',
                line=dict(color=self.colors[d % len(self.colors)], width=1),
                opacity=alpha, name=f'{d}'
            ))
        # Add clicked point spectrum (clean only)
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        clicked_zero_spectrum = zero_datacube[self.batch_y, self.batch_x]
        self.batch_spec_fig.add_trace(go.Scatter(
            y=clicked_zero_spectrum, mode='lines', name='Clean', line=dict(color='red')
        ))
        # Vertical line at spectral position
        self.batch_spec_fig.add_vline(x=s, line=dict(color='black', width=2))
        self.batch_spec_fig.update_layout(
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, max(self.dset.maxes.flatten())]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=450, height=450,
            legend=dict[str, str | float | dict[str, int]](orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, entrywidth=0.15, entrywidthmode='fraction', 
                        font=dict(size=10)  )
        )

    def layout_batch(self):
        self._init_batch_figures()
        
        # Connect widgets
        for slider in [self.i_slider, self.s_slider]:
            slider.observe(self._update_batch_plots, names='value')
        self.batch_checkboxes.observe(self._update_batch_plots, names='value')

        sliders = widgets.VBox([
            widgets.HBox([self.i_slider, self.s_slider]),
            widgets.HBox([self.batch_checkboxes, self.new_batch_button])
        ])
        plots = widgets.HBox([self.batch_img_fig, self.batch_spec_fig])
        return widgets.VBox([sliders, plots])

