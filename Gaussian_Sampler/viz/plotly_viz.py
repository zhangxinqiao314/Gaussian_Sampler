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
                description='Batches' )
            self.batch_inds = next(iter(self.sampler))
            self.new_batch_button = widgets.Button(description='New Batch')
            self.new_batch_button.on_click(self._new_batch)

    @lru_cache(maxsize=10)
    def select_datacube(self, i):
        self.dset.dset_index = i
        return self.dset[:][1]

    @lru_cache(maxsize=10)
    def select_zero_datacube(self, i):
        self.dset.dset_index = i
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
            marker=dict(color='red', size=10),
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
            self.spec_fig.layout.title.text = f'Spectrum at ({x}, {y})'

    def _handle_click(self, trace, points, selector):
        """Handle click on image plot to set x and y coordinates (works for both batch and non-batch)"""
        if points.xs and points.ys:
            # For heatmap, get the actual data coordinates
            x_clicked = points.xs[0]
            y_clicked = points.ys[0]
            # Round to nearest integer for array indexing
            x_new = int(round(x_clicked))
            y_new = int(round(y_clicked))
            # Clamp to valid range
            x_new = max(0, min(x_new, self.dset.shape[1] - 1))
            y_new = max(0, min(y_new, self.dset.shape[0] - 1))
            
            # Update coordinates (shared for batch and non-batch)
            self.x = x_new
            self.y = y_new
            
            # Check if this is a batch figure and update accordingly
            is_batch = (hasattr(self, 'batch_img_fig') and trace in self.batch_img_fig.data) or \
                      (hasattr(self, 'batch_fitted_img_fig') and trace in self.batch_fitted_img_fig.data)
            if is_batch:
                self._update_batch_plots()
            else:
                self._update_plots()
    
    def _handle_spectrum_click(self, trace, points, selector):
        """Handle click on spectrum plot to set spectral index (works for both batch and non-batch)"""
        if points.xs:
            # Get the x-coordinate (spectral index)
            s_clicked = points.xs[0]
            # Round to nearest integer and clamp to valid range
            s_new = int(round(s_clicked))
            s_new = max(0, min(s_new, self.dset.shape[2] - 1))
            # Update the slider value, which will trigger appropriate update function
            # (both _update_plots and _update_batch_plots observe s_slider)
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
            colorbar=dict(title='Intensity') ))
        self.img_fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(color='red', size=10) ))
        # Add click handler to the heatmap
        self.img_fig.data[0].on_click(self._handle_click)
        self.img_fig.update_layout(
            title=f'Sampled rate: {self.dset_list[i]}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=400, height=400 )

        # Spectrum figure
        self.spec_fig.add_trace(go.Scatter(y=datacube[y, x], mode='lines', name='Noisy', line=dict(color='blue')))
        self.spec_fig.add_trace(go.Scatter(y=zero_datacube[y, x], mode='lines', name='Clean', line=dict(color='green')))
        self.spec_fig.add_vline(x=s, line=dict(color='black', width=2))
        # Add click handler to spectrum traces
        self.spec_fig.data[0].on_click(self._handle_spectrum_click)
        self.spec_fig.data[1].on_click(self._handle_spectrum_click)
        self.spec_fig.update_layout(
            title=f'Spectrum at ({x}, {y})',
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
                self.batch_img_fig.data[-1].x = [self.x]
                self.batch_img_fig.data[-1].y = [self.y]

        # Update batch spectrum
        data = self.get_points_data(i)
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        clicked_zero_spectrum = zero_datacube[self.y, self.x]
        
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
            self.batch_img_fig.data[-1].on_click(self._handle_click)
        # Add red marker for clicked point
        self.batch_img_fig.add_trace(go.Scatter(
            x=[self.x], y=[self.y], mode='markers',
            marker=dict(color='red', size=10), name='Select'
        ))
        # Add click handler to the heatmap and red marker
        self.batch_img_fig.data[0].on_click(self._handle_click)
        self.batch_img_fig.data[-1].on_click(self._handle_click)
        self.batch_img_fig.update_layout(
            title=f'Sampled rate: {self.dset_list[i]}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=450, height=450,
            showlegend=False
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
        clicked_zero_spectrum = zero_datacube[self.y, self.x]
        self.batch_spec_fig.add_trace(go.Scatter(
            y=clicked_zero_spectrum, mode='lines', name='Clean', line=dict(color='red')
        ))
        # Vertical line at spectral position
        self.batch_spec_fig.add_vline(x=s, line=dict(color='black', width=2))
        # Add click handlers to spectrum traces
        for trace_idx in range(len(self.batch_spec_fig.data)):
            self.batch_spec_fig.data[trace_idx].on_click(self._handle_spectrum_click)
        self.batch_spec_fig.update_layout(
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, max(self.dset.maxes.flatten())]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=450, height=450,
            showlegend=False
        )

    def layout_batch(self):
        self._init_batch_figures()
        
        # Connect widgets
        for slider in [self.i_slider, self.s_slider]:
            slider.observe(self._update_batch_plots, names='value')
        self.batch_checkboxes.observe(self._update_batch_plots, names='value')

        # Create legend HTML
        num_batches = len(self.batch_checkboxes.options)
        legend_items = [f'<div style="display: inline-flex; align-items: center; margin: 2px 5px;"><div style="width: 12px; height: 12px; background-color: {self.colors[i % len(self.colors)]}; margin-right: 5px; border-radius: 50%; border: 1px solid black;"></div><span>{i}</span></div>' 
                        for i in range(min(num_batches, len(self.colors)))]
        legend_html = widgets.HTML(value='<div style="border: 1px solid black; padding: 10px; background-color: white;"><b>Batch Colors</b><br><div style="display: flex; flex-wrap: wrap;">' + ''.join(legend_items) + '</div></div>')
        
        sliders = widgets.VBox([
            widgets.HBox([self.i_slider, self.s_slider]),
            widgets.HBox([self.batch_checkboxes, self.new_batch_button, legend_html])
        ])
        plots = widgets.HBox([self.batch_img_fig, self.batch_spec_fig])
        return widgets.VBox([sliders, plots])



class Poisson_Sampled_PV_viz_embeddings(Poisson_Sampled_PV_viz):
    '''Interactive visualization for embeddings using Plotly and ipywidgets'''
    
    def __init__(self, model, emb, dset, **kwargs):
        super().__init__(dset, model._dataloader_sampler)
        self.parameters_list = ['Amplitude', 'Mean', 'FWHM', 'nu']
        self.model = model
        self.emb = emb
        
        # Add fit channel slider
        self.f_slider = widgets.IntSlider(description='Fit channel', value=0, min=0, max=self.model.num_fits-1)
        # Set initial dataset index to match embedding noise level
        self.i_slider.value = self.dset.h5_keys().index(self.dset.dset_name)
        
        # Create FigureWidgets for final (fitted) image and spectrum
        self.fitted_img_fig = go.FigureWidget()
        self.fitted_spec_fig = go.FigureWidget()
        self.param_fig_list = [go.FigureWidget() for _ in range(self.model.num_params)]
        
    @lru_cache(maxsize=10)
    def select_fits_params(self, which=slice(None)):
        '''
        returns fits, params, shape (10000, s), (10000, 4)
        '''
        return self.emb[:][which] # (10000, s), (10000, 4)

    def fits_max(self):
        """Get maximum value from fits for scaling"""
        fits = self.select_fits_params(which=0)
        return float(fits.max())

    def params_max(self, par, f):
        """Get maximum value from parameters for scaling"""
        params = self.select_fits_params(which=1).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        return float(params[:, :, f, par].max())

    ############################################ Fits plotting functions

    def plot_fits_sum_img(self, s):
        fits = self.select_fits_params(which=0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        data_ = np.flipud(fits[..., s].sum(axis=-1).T)
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=data_, colorscale='Viridis', zmin=0, zmax=self.fits_max(),
            colorbar=dict(title='Intensity')
        ))
        fig.update_layout(
            title='Fitted Intensity',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=400, height=400
        )
        return fig

    def plot_fits_img(self, s, f):
        fits = self.select_fits_params(which=0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        data_ = np.flipud(fits[..., f, s].T)
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=data_, colorscale='Viridis', zmin=0, zmax=self.fits_max(),
            colorbar=dict(title='Intensity')
        ))
        fig.update_layout(
            title=f'Fit {f}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=400, height=400
        )
        return fig

    def plot_fit_sum_spectrum(self, x, y):
        fits = self.select_fits_params(0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        spectrum = fits[x, y].sum(axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=spectrum, mode='lines', name='Fitted', line=dict(color='red')))
        fig.update_layout(
            title='Fitted Spectrum',
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, self.fits_max()]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=400, height=400
        )
        return fig

    def plot_fits_spectrum(self, x, y, f):
        fits = self.select_fits_params(0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        spectrum = fits[x, y, f]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=spectrum, mode='lines', name=f'Fit {f}', 
                                 line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=f'Fitted Spectrum {f}',
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, self.fits_max()]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=400, height=400
        )
        return fig

    def plot_params_img(self, par, f):
        params = self.select_fits_params(1).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        data_ = np.flipud(params[:, :, f, par].T)
        
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=data_, colorscale='Viridis', zmin=0, zmax=self.params_max(par, f),
            colorbar=dict(title='Value')
        ))
        fig.update_layout(
            title=f'{self.parameters_list[par]}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=200, height=200
        )
        return fig

    ############################################ Update functions

    def _update_fits_params(self):
        """Update Final Image and Final Spectrum plots."""
        i, s = self.i_slider.value, self.s_slider.value
        f = self.f_slider.value
        x, y = self.x, self.y
        
        fits, params = self.select_fits_params()
        fits = fits.reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        params = params.reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        
        # Update fitted image
        data_fit = np.flipud(fits[..., f, s].T)
        with self.fitted_img_fig.batch_update():
            self.fitted_img_fig.data[0].z = data_fit
            self.fitted_img_fig.data[0].zmax = self.fits_max()
            self.fitted_img_fig.data[1].x = [x]
            self.fitted_img_fig.data[1].y = [y]
            self.fitted_img_fig.layout.title.text = f'Final Image - Fit {f}'
        
        # Update fitted spectrum
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        spectrum_fit = fits[x, y, f]
        spectrum_sum = fits[x, y].sum(axis=0)  # Sum of all channels
        zero_spectrum = zero_datacube[y, x]  # Zero noise line
        with self.fitted_spec_fig.batch_update():
            self.fitted_spec_fig.data[0].y = spectrum_fit
            self.fitted_spec_fig.data[1].y = spectrum_sum
            self.fitted_spec_fig.data[2].y = zero_spectrum
            # Update vertical line
            self.fitted_spec_fig.layout.shapes[0].x0 = s
            self.fitted_spec_fig.layout.shapes[0].x1 = s
            self.fitted_spec_fig.layout.title.text = f'Fitted Spectrum - Fit {f}'

        # Update parameter images
        for par in range(self.model.num_params):
            with self.param_fig_list[par].batch_update():
                self.param_fig_list[par].data[0].z = params[:, :, f, par]
                self.param_fig_list[par].data[0].zmax = self.params_max(par, f)
                self.param_fig_list[par].data[1].x = [x]
                self.param_fig_list[par].data[1].y = [y]

    def _update_fits_plots(self, change=None):
        """Update all fits plots by calling individual update functions."""
        self._update_plots()
        self._update_fits_params()

    def _handle_fits_click(self, trace, points, selector):
        """Handle click on fits image plot to set x and y coordinates"""
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
            self._update_fits_plots()

    def _init_fits_params_figures(self):
        i, s = self.i_slider.value, self.s_slider.value
        f = self.f_slider.value
        x, y = self.x, self.y
        
        # Initialize parent figures (original image and spectrum) only if empty
        if not hasattr(self, 'img_fig') or len(self.img_fig.data) == 0:
            self._init_figures()
        
        # Update parent figure titles and legend settings
        self.img_fig.update_layout(
            title=f'Original Image - {self.dset_list[i]}',
            showlegend=False
        )
        
        fits, params = self.select_fits_params()
        fits = fits.reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        params = params.reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        
        # Only initialize fitted figures if they're empty (like parent class does for batch figures)
        if len(self.fitted_img_fig.data) == 0:
            # Fitted image
            data_fit = np.flipud(fits[..., f, s].T)
            self.fitted_img_fig.add_trace(go.Heatmap(
                z=data_fit, colorscale='Viridis', zmin=0, zmax=self.fits_max(),
                colorbar=dict(title='Intensity')
            ))
            self.fitted_img_fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(color='red', size=10), 
            ))
            self.fitted_img_fig.data[0].on_click(self._handle_fits_click)
            self.fitted_img_fig.update_layout(
                title=f'Final Image - Fit {f}',
                xaxis_title='X Position', yaxis_title='Y Position',
                width=400, height=400,
                showlegend=False
            )
            
            # Final (fitted) spectrum
            zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
            spectrum_fit = fits[x, y, f]
            spectrum_sum = fits[x, y].sum(axis=0)  # Sum of all channels
            zero_spectrum = zero_datacube[y, x]  # Zero noise line
            self.fitted_spec_fig.add_trace(go.Scatter(y=spectrum_fit, mode='lines', name=f'Fit {f}', 
                                                    line=dict(color='red', dash='dash')))
            self.fitted_spec_fig.add_trace(go.Scatter(y=spectrum_sum, mode='lines', name='Sum', 
                                                    line=dict(color='orange', dash='dot')))
            self.fitted_spec_fig.add_trace(go.Scatter(y=zero_spectrum, mode='lines', name='Clean', 
                                                    line=dict(color='green')))
            self.fitted_spec_fig.add_vline(x=s, line=dict(color='black', width=2))
            self.fitted_spec_fig.update_layout(
                title=f'Final Spectrum - Fit {f}',
                xaxis_title='Spectrum Value', yaxis_title='Intensity',
                yaxis=dict(range=[0, self.fits_max()]),
                xaxis=dict(range=[0, self.dset.spec_len]),
                width=400, height=400,
                showlegend=False
            )

            # Parameter images
            for par in range(self.model.num_params):
                data_param = np.flipud(params[:, :, f, par].T)
                self.param_fig_list[par].add_trace(go.Heatmap(
                    z=data_param, colorscale='Viridis', zmin=0, zmax=self.params_max(par, f),
                    colorbar=dict(title='Value')
                ))
                self.param_fig_list[par].add_trace(go.Scatter(
                    x=[x], y=[y], mode='markers',
                    marker=dict(color='red', size=10),
                ))
                self.param_fig_list[par].data[0].on_click(self._handle_fits_click)
                self.param_fig_list[par].update_layout(
                    title=f'{self.parameters_list[par]}',
                    xaxis_title='X Position', yaxis_title='Y Position',
                    width=350, height=350
                )

    def layout_fits_params(self):
        """Layout for viewing training results of a single point at a time."""
        self._init_fits_params_figures()
        
        # Enable legends for spectrum plots
        self.spec_fig.update_layout(showlegend=True)
        self.fitted_spec_fig.update_layout(showlegend=True)
  
        # Connect widgets to update function (updates both original and fitted plots)
        for slider in [self.i_slider, self.s_slider, self.f_slider]:
            slider.observe(self._update_fits_plots, names='value')
        
        # Connect click handlers for images
        def handle_img_click(trace, points, selector):
            if points.xs and points.ys:
                x_clicked = points.xs[0]
                y_clicked = points.ys[0]
                self.x = int(round(x_clicked))
                self.y = int(round(y_clicked))
                self.x = max(0, min(self.x, self.dset.shape[0] - 1))
                self.y = max(0, min(self.y, self.dset.shape[1] - 1))
                self._update_fits_plots()
        
        self.img_fig.data[0].on_click(handle_img_click)
        self.fitted_img_fig.data[0].on_click(handle_img_click)
        
        # Connect click handlers for spectra
        def handle_spec_click(trace, points, selector):
            if points.xs:
                s_clicked = points.xs[0]
                s_new = int(round(s_clicked))
                s_new = max(0, min(s_new, self.dset.shape[2] - 1))
                self.s_slider.value = s_new
        
        self.spec_fig.data[0].on_click(handle_spec_click)
        self.spec_fig.data[1].on_click(handle_spec_click)
        self.fitted_spec_fig.data[0].on_click(handle_spec_click)
        if len(self.fitted_spec_fig.data) > 1:
            self.fitted_spec_fig.data[1].on_click(handle_spec_click)
        if len(self.fitted_spec_fig.data) > 2:
            self.fitted_spec_fig.data[2].on_click(handle_spec_click)
            
        sliders = widgets.VBox([
            widgets.HBox([self.i_slider, self.s_slider, self.f_slider]),
            widgets.HTML(value='<i>Click on the image to select x, y coordinates</i>')
        ])
        # Left column: images (top: original, bottom: final)
        # Right column: spectra (top: initial, bottom: final)
        left_col = widgets.VBox([self.img_fig, self.fitted_img_fig])
        right_col = widgets.VBox([self.spec_fig, self.fitted_spec_fig])
        main_row = widgets.HBox([left_col, right_col])
        param_plots = widgets.HBox(self.param_fig_list)
        
        return widgets.VBox([sliders, main_row, param_plots])

    ############################################ Batch fits plotting

    def plot_batch_fits(self, i, checked):
        data = self.get_points_data(i)
        fig = go.Figure()
        for d, dat in enumerate(data):
            alpha = 1.0 if d in checked else 0.1
            fig.add_trace(go.Scatter(
                y=dat.mean(axis=0), mode='lines',
                line=dict(color=self.colors[d % len(self.colors)], width=1),
                opacity=alpha, name=f'Batch {d}'
            ))
        fig.update_layout(
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, max(self.dset.maxes.flatten())]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=450, height=450
        )
        return fig

    def layout_batch_fits(self):
        # Initialize batch figures from parent
        if not hasattr(self, 'batch_img_fig') or len(self.batch_img_fig.data) == 0:
            self._init_batch_figures()
        
        # Initialize fits figures
        self._init_fits_figures()
        
        # Create batch final image (fitted image with batch points)
        i = self.i_slider.value
        s = self.s_slider.value
        f = self.f_slider.value
        checked = list(self.batch_checkboxes.value) if hasattr(self, 'batch_checkboxes') else []
        
        fits = self.select_fits_params(which=0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        data_fit = np.flipud(fits[..., f, s].T)
        pts = self.get_points_idx()
        
        self.batch_fitted_img_fig = go.FigureWidget()
        self.batch_fitted_img_fig.add_trace(go.Heatmap(
            z=data_fit, colorscale='Viridis', zmin=0, zmax=self.fits_max(),
            colorbar=dict(title='Intensity')
        ))
        for p, pt in enumerate(pts):
            xs, ys = zip(*pt) if pt else ([], [])
            alpha = 1.0 if p in checked else 0.2
            self.batch_fitted_img_fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(color=self.colors[p % len(self.colors)], size=6),
                opacity=alpha, name=f'{p}'
            ))
        self.batch_fitted_img_fig.add_trace(go.Scatter(
            x=[self.x], y=[self.y], mode='markers',
            marker=dict(color='red', size=10), name='Select'
        ))
        self.batch_fitted_img_fig.data[0].on_click(self._handle_click)
        self.batch_fitted_img_fig.update_layout(
            title=f'Final Image - Fit {f}',
            xaxis_title='X Position', yaxis_title='Y Position',
            width=450, height=450,
            showlegend=False
        )
        
        # Create batch final spectrum (fitted spectrum with batch fits)
        self.batch_fitted_spec_fig = go.FigureWidget()
        data = self.get_points_data(i)
        fits_reshaped = fits.reshape(-1, self.model.num_fits, self.dset.shape[-1])
        clumps = self.split_list()
        for d, dat in enumerate(data):
            alpha = 1.0 if d in checked else 0.1
            batch_inds = clumps[d]
            batch_fits = fits_reshaped[batch_inds].mean(axis=0)[f]
            self.batch_fitted_spec_fig.add_trace(go.Scatter(
                y=batch_fits, mode='lines',
                line=dict(color=self.colors[d % len(self.colors)], width=1),
                opacity=alpha, name=f'{d}'
            ))
        clicked_fit_spectrum = fits[self.y, self.x, f]
        self.batch_fitted_spec_fig.add_trace(go.Scatter(
            y=clicked_fit_spectrum, mode='lines', line=dict(color='red')
        ))
        self.batch_fitted_spec_fig.add_vline(x=s, line=dict(color='black', width=2))
        # Add click handlers to spectrum traces
        for trace_idx in range(len(self.batch_fitted_spec_fig.data)):
            self.batch_fitted_spec_fig.data[trace_idx].on_click(self._handle_spectrum_click)
        self.batch_fitted_spec_fig.update_layout(
            title=f'Final Spectrum - Fit {f}',
            xaxis_title='Spectrum Value', yaxis_title='Intensity',
            yaxis=dict(range=[0, self.fits_max()]),
            xaxis=dict(range=[0, self.dset.spec_len]),
            width=450, height=450,
            showlegend=False
        )
        
        # Update parent batch figures titles
        self.batch_img_fig.update_layout(title=f'Original Image - {self.dset_list[i]}', showlegend=False)
        self.batch_spec_fig.update_layout(title='Initial Spectrum', showlegend=False)
        
        # Connect widgets
        for slider in [self.i_slider, self.s_slider, self.f_slider]:
            slider.observe(self._update_batch_fits_plots, names='value')
        if hasattr(self, 'batch_checkboxes'):
            self.batch_checkboxes.observe(self._update_batch_fits_plots, names='value')
        
        # Create legend HTML
        num_batches = len(self.batch_checkboxes.options) if hasattr(self, 'batch_checkboxes') else 0
        legend_items = [f'<div style="display: inline-flex; align-items: center; margin: 2px 5px;"><div style="width: 12px; height: 12px; background-color: {self.colors[i % len(self.colors)]}; margin-right: 5px; border-radius: 50%; border: 1px solid black;"></div><span>{i}</span></div>' 
                        for i in range(min(num_batches, len(self.colors)))]
        legend_html = widgets.HTML(value='<div style="border: 1px solid black; padding: 10px; background-color: white;"><b>Batch Colors</b><br><div style="display: flex; flex-wrap: wrap;">' + ''.join(legend_items) + '</div></div>')
        
        sliders = widgets.VBox([
            widgets.HBox([self.i_slider, self.s_slider, self.f_slider]),
            widgets.HBox([self.batch_checkboxes, self.new_batch_button, legend_html]) if hasattr(self, 'batch_checkboxes') else widgets.HBox([legend_html])
        ])
        
        # Left column: images (top: original, bottom: final)
        # Right column: spectra (top: initial, bottom: final)
        left_col = widgets.VBox([self.batch_img_fig, self.batch_fitted_img_fig])
        right_col = widgets.VBox([self.batch_spec_fig, self.batch_fitted_spec_fig])
        main_row = widgets.HBox([left_col, right_col])
        
        return widgets.VBox([sliders, main_row])

    def _update_batch_fits_plots(self, change=None):
        i = self.i_slider.value
        s = self.s_slider.value
        f = self.f_slider.value
        checked = list(self.batch_checkboxes.value) if hasattr(self, 'batch_checkboxes') else []
        
        # Update original batch image (from parent)
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        data_ = datacube[:, :, s]
        pts = self.get_points_idx()
        with self.batch_img_fig.batch_update():
            self.batch_img_fig.data[0].z = data_
            self.batch_img_fig.data[0].zmax = self.datacube_max(i)
            for p, pt in enumerate(pts):
                if p + 1 < len(self.batch_img_fig.data) - 1:
                    xs, ys = zip(*pt) if pt else ([], [])
                    self.batch_img_fig.data[p + 1].x = xs
                    self.batch_img_fig.data[p + 1].y = ys
                    self.batch_img_fig.data[p + 1].opacity = 1.0 if p in checked else 0.2
            if len(self.batch_img_fig.data) > len(pts) + 1:
                self.batch_img_fig.data[-1].x = [self.x]
                self.batch_img_fig.data[-1].y = [self.y]
            self.batch_img_fig.layout.title.text = f'Original Image - {self.dset_list[i]}'
        
        # Update original batch spectrum (from parent)
        data = self.get_points_data(i)
        zero_datacube = self.select_zero_datacube(i).reshape(self.dset.shape)
        clicked_zero_spectrum = zero_datacube[self.y, self.x]
        with self.batch_spec_fig.batch_update():
            num_batches = len(data)
            for d, dat in enumerate(data):
                if d < num_batches:
                    self.batch_spec_fig.data[d].y = dat.mean(axis=0)
                    self.batch_spec_fig.data[d].opacity = 1.0 if d in checked else 0.2
            if len(self.batch_spec_fig.data) >= num_batches + 1:
                self.batch_spec_fig.data[num_batches].y = clicked_zero_spectrum
            if self.batch_spec_fig.layout.shapes:
                self.batch_spec_fig.layout.shapes[0].x0 = s
                self.batch_spec_fig.layout.shapes[0].x1 = s
        
        # Update final batch image
        fits = self.select_fits_params(which=0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits, -1)
        data_fit = np.flipud(fits[..., f, s].T)
        with self.batch_fitted_img_fig.batch_update():
            self.batch_fitted_img_fig.data[0].z = data_fit
            self.batch_fitted_img_fig.data[0].zmax = self.fits_max()
            for p, pt in enumerate(pts):
                if p + 1 < len(self.batch_fitted_img_fig.data) - 1:
                    xs, ys = zip(*pt) if pt else ([], [])
                    self.batch_fitted_img_fig.data[p + 1].x = xs
                    self.batch_fitted_img_fig.data[p + 1].y = ys
                    self.batch_fitted_img_fig.data[p + 1].opacity = 1.0 if p in checked else 0.2
            if len(self.batch_fitted_img_fig.data) > len(pts) + 1:
                self.batch_fitted_img_fig.data[-1].x = [self.x]
                self.batch_fitted_img_fig.data[-1].y = [self.y]
            self.batch_fitted_img_fig.layout.title.text = f'Final Image - Fit {f}'
        
        # Update final batch spectrum
        fits_reshaped = fits.reshape(-1, self.model.num_fits, self.dset.shape[-1])
        clumps = self.split_list()
        with self.batch_fitted_spec_fig.batch_update():
            for d, dat in enumerate(data):
                if d < len(clumps):
                    batch_inds = clumps[d]
                    batch_fits = fits_reshaped[batch_inds].mean(axis=0)[f]
                    if d < len(self.batch_fitted_spec_fig.data) - 2:
                        self.batch_fitted_spec_fig.data[d].y = batch_fits
                        self.batch_fitted_spec_fig.data[d].opacity = 1.0 if d in checked else 0.1
            clicked_fit_spectrum = fits[self.y, self.x, f]
            if len(self.batch_fitted_spec_fig.data) >= len(data) + 1:
                self.batch_fitted_spec_fig.data[len(data)].y = clicked_fit_spectrum
            if self.batch_fitted_spec_fig.layout.shapes:
                self.batch_fitted_spec_fig.layout.shapes[0].x0 = s
                self.batch_fitted_spec_fig.layout.shapes[0].x1 = s
            self.batch_fitted_spec_fig.layout.title.text = f'Final Spectrum - Fit {f}'