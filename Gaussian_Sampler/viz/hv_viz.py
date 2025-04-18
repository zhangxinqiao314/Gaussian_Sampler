from functools import lru_cache
from tabnanny import check
import holoviews as hv
import panel as pn
import numpy as np
import holoviews.streams as streams

def test():
    print('Hello World') 
  
class Fake_PV_viz:
    '''testing'''
    def __init__(self, dset, sampler=None):
        self.dset = dset
        self.dset_list = dset.h5_keys()
        self.sampler = sampler
        self.colors = ['green','orange','yellow','brown','pink','gray','magenta','cyan','purple','lime','teal','maroon','indigo','gold']

        if sampler is not None: 
            self.sampler = sampler
            self.batch_checkboxes = pn.widgets.CheckBoxGroup(name='Batch Checkboxes', 
                options=list(range(int(np.ceil(self.sampler.batch_size/self.sampler.num_neighbors)))),
                value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                inline=True)
            self.batch_inds = next(iter(self.sampler))
            # Create dynamic maps for batch plots
            self.batch_inds_dmap = hv.DynamicMap(pn.bind(self.plot_batch_points, 
                                                        checked=self.batch_checkboxes))
                                                        #  trigger=self.button_stream.param.button))
            self.batch_spec_dmap = hv.DynamicMap(pn.bind(self.plot_batch_spectrum, i=self.i_slider,
                                                        checked=self.batch_checkboxes))
                                                        #  trigger=self.button_stream.param.button) )
            
        
        # Create interactive widgets
        self.i_slider = pn.widgets.IntSlider(name='Noise std', value=0, start=0, end=len(self.dset_list )-1)
        self.x_slider = pn.widgets.IntSlider(name='x', value=25, start=0, end=dset.shape[0]-1)
        self.y_slider = pn.widgets.IntSlider(name='y', value=25, start=0, end=dset.shape[1]-1)
        self.s_slider = pn.widgets.IntSlider(name='spectral value', value=0, start=0, end=dset.shape[2]-1)
        
        
        self.button_stream = streams.Stream.define('ButtonStream', button=False)()
        self.button = pn.widgets.Button(name='New Batch', button_type='primary')
        self.button.on_click(lambda event: self.button_stream.event(button=True))

        def trigger(click): self.batch_inds = next(iter(self.sampler))
        pn.bind(trigger, self.button_stream.param.button)
        
        # Dynamic maps for the red dot and vertical line
        self.dot_dmap = hv.DynamicMap(pn.bind(self.show_dot, 
                                              x=self.x_slider, y=self.y_slider))
        self.vline_dmap = hv.DynamicMap(pn.bind(self.show_vline, 
                                                s=self.s_slider))

        # Create dynamic maps for image and spectrum plots
        self.img_dmap = hv.DynamicMap(pn.bind(self.plot_datacube_img, 
                                              i=self.i_slider, s=self.s_slider))
        self.img_scaled_dmap = hv.DynamicMap(pn.bind(self.plot_datacube_img_scaled, 
                                                     i=self.i_slider, s=self.s_slider))
        self.spec_dmap = hv.DynamicMap(pn.bind(self.plot_datacube_spectrum, 
                                               i=self.i_slider, x=self.x_slider, y=self.y_slider))
        self.spec_scaled_dmap = hv.DynamicMap(pn.bind(self.plot_datacube_spectrum, 
                                                      i=self.i_slider, x=self.x_slider, y=self.y_slider))
        self.zero_spec_dmap = hv.DynamicMap(pn.bind(self.plot_datacube_spectrum, 
                                                    i=0, x=self.x_slider, y=self.y_slider))
        
           
    @lru_cache(maxsize=10)
    def select_datacube(self,i):
        self.dset.noise_ = i # self.dset.h5_keys()[i]
        # self.dset.scale = False
        return self.dset[:][1] # 100, 100, 500
    
    ############################################ input data helpers
    
    def show_dot(self, x, y): return hv.Scatter([(x, y)]).opts( color='red', size=5, marker='o',
                                        axiswise=True, shared_axes=False)

    def show_vline(self, s): return hv.VLine(int(s)).opts(
            color='black', line_width=2,
            axiswise=True, shared_axes=False)

    ############################################ input data plotting
    
    def plot_datacube_img(self, i, s):
        datacube = self.select_datacube(i)[:].reshape(self.dset.shape[0],self.dset.shape[1],self.dset.shape[2])
        data_ = np.flipud(datacube[:, :, s].T)
        return hv.Image(
                    data_, bounds=(0,0,datacube.shape[0],datacube.shape[1]),
                    kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                    vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(
                            cmap='viridis', colorbar=True, clim=(0, datacube.max()),
                            width=350, height=300, title=f'Noise:{self.dset_list[i]}')
    
    
    def plot_datacube_img_scaled(self, i, s): # TODO: fix
        datacube = self.select_datacube(i).reshape(self.dset.shape[0],self.dset.shape[1],self.dset.shape[2])
        data_ = np.flipud(datacube[:, s].reshape(self.shape[0],self.shape[1]).T)
        return hv.Image(data_, bounds=(0,0,self.dset.shape[0],self.dset.shape[1]),
                        kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(
                            cmap='viridis', colorbar=True, clim=(0, datacube.max()),
                            width=350, height=300, title='Datacube Intensity')

    def plot_datacube_spectrum(self, i, x, y):
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        
        return hv.Curve(datacube[x, y],
                        kdims=[hv.Dimension('spectrum', label='Spectrum Value')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(width=350, height=300,
                                ylim=(0, datacube.max()), xlim=(0, self.dset.spec_len),
                                axiswise=True, shared_axes=False)
        
    def layout_input(self):
        return pn.Column(
        pn.Row(self.i_slider, self.s_slider),
        pn.Row(self.x_slider, self.y_slider),
        (self.img_dmap*self.dot_dmap + \
         self.spec_dmap*self.vline_dmap*self.zero_spec_dmap).opts(shared_axes=True, axiswise=True),
                        )

    ############################################Batch helpers
    
    def split_list(self,): # TODO: trigger if dset changes
        return [self.batch_inds[i:i + self.sampler.num_neighbors] for i in range(0, len(self.batch_inds), self.sampler.num_neighbors)]
    
    def get_points_idx(self): 
        clumps = self.split_list()
        return [[ (int(ind / self.dset.shape[0]),ind % self.dset.shape[0]
                    ) for ind in clump
                ] for clump in clumps ]
    
    def get_points_data(self,i):
        dset = self.select_datacube(i)
        clumps = self.split_list()
        return [ np.asarray([dset[ind] for ind in clump],dtype=np.float32
                         ) for clump in clumps ]
        
    ############################################Batch plotting
    def plot_batch_points(self, checked):
        pts = self.get_points_idx()                                             
        scatter_list = []
        for p, pt in enumerate(pts):
            if p in checked:
                point = hv.Scatter(pt).opts(color=self.colors[p], size=3, marker='o', alpha=1,
                                            axiswise=True, shared_axes=False)
            else:
                point = hv.Scatter(pt).opts(color=self.colors[p], size=3, marker='o', alpha=0.1,
                                            axiswise=True, shared_axes=False)
            scatter_list.append(point)
        return hv.Overlay(scatter_list).opts(shared_axes=True, axiswise=True)
       
    def plot_batch_spectrum(self, checked, i):
        data = self.get_points_data(i)
        curves_list = []
        for d,dat in enumerate(data):
            if d in checked: curve = hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
                                            color=self.colors[d], alpha=1,
                                            ylim=(0, self.dset.maxes.max()), xlim=(0, self.dset.spec_len),
                                            axiswise=True, shared_axes=False, line_width=1)
            else: curve = hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
                                            color=self.colors[d], alpha=0.1,
                                            ylim=(0, self.dset.maxes.max()), xlim=(0, self.dset.spec_len),
                                            axiswise=True, shared_axes=False, line_width=1)
            curves_list.append( curve )
            
        return hv.Overlay(curves_list).opts(shared_axes=True, axiswise=True) 
    
    def layout_batch(self):
        dmap = pn.Column(#pn.Row(button),
                pn.Row(self.i_slider, self.s_slider),
                pn.Row(self.x_slider, self.y_slider),
                pn.Row(self.batch_checkboxes),
                (self.img_dmap*self.dot_dmap*self.batch_inds_dmap + 
                    self.spec_dmap*self.zero_spec_dmap*self.batch_spec_dmap*self.vline_dmap).opts(shared_axes=True,axiswise=True)
            )

        return dmap
    
    
class Fake_PV_viz_embeddings(Fake_PV_viz):
    def __init__(self, model, emb, dset, **kwargs):
        super().__init__(dset, model._dataloader_sampler)
        self.parameters_list = ['Amplitude', 'Center', 'Width', 'Nu']
        self.model = model
        self.emb = emb
        
        
        # Create dynamic maps for embedding and fits
        self.f_slider = pn.widgets.IntSlider(name='Fit channel', value=0, start=0, end=self.model.num_fits-1)
        self.i_slider.value = self.dset.h5_keys().index(self.emb._noise)
        
        self.fit_img_sum_dmap = hv.DynamicMap(pn.bind(self.plot_fits_sum_img, s=self.s_slider))
        
        self.fit_img_dmap = hv.DynamicMap(pn.bind(self.plot_fits_img, s=self.s_slider, f=self.f_slider))
        
        self.fit_spec_sum_dmap = hv.DynamicMap(pn.bind(self.plot_fit_sum_spectrum, 
                                                   x=self.x_slider, y=self.y_slider))
        
        self.fit_spec_dmap = hv.DynamicMap(pn.bind(self.plot_fits_spectrum, 
                                                   x=self.x_slider, y=self.y_slider, f=self.f_slider))
        
        self.param_dmap_list = []
        for par in range(self.model.num_params):
            self.param_dmap_list.append( hv.DynamicMap(pn.bind(self.plot_params_img, 
                                                      par=par, f=self.f_slider)
                                              )*self.dot_dmap )
   
    @lru_cache(maxsize=10)
    def select_fits_params(self, which=slice(None)):
        '''
        returns fits, params, shape (10000, s), (10000, 4)
        '''
        return self.emb[:][which] # (10000, s), (10000, 4)


    #######################################################
    def plot_fits_sum_img(self, s):
        fits = self.select_fits_params(which=0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits,-1)
        data_ = np.flipud(fits[..., s].T)
        
        return hv.Image(data_.sum(axis=-1), bounds=(0, 0, data_.shape[0], data_.shape[1]),
                        kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(cmap='viridis', colorbar=True, clim=(0, fits.max()),
                            width=350, height=300, title='Fitted Intensity')
                        
    def plot_fits_img(self, s, f):
        fits = self.select_fits_params(which=0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits,-1)
        data_ = np.flipud(fits[..., f, s].T)
        
        return hv.Image(data_, bounds=(0, 0, data_.shape[0], data_.shape[1]),
                        kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(cmap='viridis', colorbar=True, clim=(0, fits.max()),
                            width=350, height=300, title=f'Fit {f}')

    #######################################################
    def plot_fit_sum_spectrum(self, x, y):
        fits = self.select_fits_params(0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits,-1)
        return hv.Curve(fits[x, y].sum(axis=0),
                        kdims=[hv.Dimension('spectrum', label='Spectrum Value')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(width=350, height=300, color='red',
                            ylim=(0, fits.max()), xlim=(0, self.dset.spec_len),
                            axiswise=True, shared_axes=False, title=f'Fitted Spectrum')
                        
    def plot_fits_spectrum(self, x, y, f):
        fits = self.select_fits_params(0).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits)
        return hv.Curve(fits[x, y, f],
                        kdims=[hv.Dimension('spectrum', label='Spectrum Value')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(width=350, height=300, color='red', line_style='dashed',
                               ylim=(0,fits.max()), xlim=(0,self.dset.spec_len),
                            axiswise=True, shared_axes=False, title=f'Fitted Spectrum {f}')

    #######################################################
    def plot_params_img(self, par, f):
        params = self.select_fits_params(1).reshape(self.dset.shape[0], self.dset.shape[1], self.model.num_fits,-1)
        data_ = np.flipud(params[:, :, f, par].T)
        
        return hv.Image(data_, bounds=(0, 0, data_.shape[0], data_.shape[1]),
                        kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(cmap='viridis', colorbar=True, clim=(0, data_.max()),
                            width=350, height=300, title=f'{self.parameters_list[par]}')

    #######################################################
    def layout_fits_params(self): # TODO: fix
    
        return pn.Column(
            pn.Row(self.f_slider, self.s_slider),
            pn.Row(self.x_slider, self.y_slider),
            
            (self.img_dmap*self.dot_dmap + \
             self.fit_img_dmap*self.dot_dmap + \
             self.spec_dmap*self.vline_dmap*self.zero_spec_dmap*self.fit_spec_sum_dmap
             ).opts(shared_axes=True, axiswise=True),
            
            hv.Layout(self.param_dmap_list).opts(shared_axes=True, axiswise=True) )
    #######################################################
    
    def plot_batch_fits(self, checked): # TODO: fix
        data = self.get_points_data(i)
        curves_list = []
        for d,dat in enumerate(data):
            if d in checked: curve = hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
                                            color=self.colors[d], alpha=1,
                                            ylim=(0, self.dset.maxes.max()), xlim=(0, self.dset.spec_len),
                                            axiswise=True, shared_axes=False, line_width=1)
            else: curve = hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
                                            color=self.colors[d], alpha=0.1,
                                            ylim=(0, self.dset.maxes.max()), xlim=(0, self.dset.spec_len),
                                            axiswise=True, shared_axes=False, line_width=1)
            curves_list.append( curve )
            
        return hv.Overlay(curves_list).opts(shared_axes=True, axiswise=True) 
    
    def layout_batch_fits(self): # TODO: fix
        dmap = pn.Column(#pn.Row(button),
                pn.Row(self.i_slider, self.s_slider),
                pn.Row(self.x_slider, self.y_slider),
                pn.Row(self.batch_checkboxes),
                (self.img_dmap*self.dot_dmap*self.batch_inds_dmap + 
                    self.spec_dmap*self.zero_spec_dmap*self.batch_spec_dmap*self.vline_dmap).opts(shared_axes=True,axiswise=True)
            )

        return dmap
    
    