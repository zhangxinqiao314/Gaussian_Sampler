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
        # self.dset_list = [  '00.000_noise',
        #                     '00.011_noise',
        #                     '00.032_noise',
        #                     '00.058_noise',
        #                     '00.089_noise',
        #                     '00.125_noise',
        #                     '00.164_noise',
        #                     '00.207_noise',
        #                     '00.253_noise',
        #                     '00.302_noise',
        #                     '00.354_noise',
        #                     '00.408_noise',
        #                     '00.465_noise',
        #                     '00.524_noise',
        #                     '00.586_noise',
        #                     '00.650_noise',
        #                     '00.716_noise',
        #                     '00.784_noise',
        #                     '00.854_noise',
        #                     '00.926_noise']
        self.parameters_list = ['Amplitude', 'Center', 'Width', 'Nu']
        self.samplers = ['scipy','gaussian']
        self.colors = ['green','orange','yellow','brown','pink','gray','magenta','cyan','purple','lime','teal','maroon','indigo','gold']

        if sampler is not None: self.sampler = sampler
        
        # Create interactive widgets
        self.i_slider = pn.widgets.IntSlider(name='Noise std', value=0, start=0, end=len(self.dset_list )-1)
        self.x_slider = pn.widgets.IntSlider(name='x', value=25, start=0, end=dset.shape[0]-1)
        self.y_slider = pn.widgets.IntSlider(name='y', value=25, start=0, end=dset.shape[1]-1)
        self.s_slider = pn.widgets.IntSlider(name='spectral value', value=0, start=0, end=dset.shape[2]-1)
        self.batch_checkboxes = pn.widgets.CheckBoxGroup(name='Batch Checkboxes', 
            options=list(range(int(np.ceil(self.sampler.batch_size/self.sampler.num_neighbors)))),
            value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            inline=True)
        
        self.sampler_selector = pn.widgets.Select(name='Sampler', options=self.samplers)
        
        
        self.button_stream = streams.Stream.define('ButtonStream', button=False)()
        self.button = pn.widgets.Button(name='New Batch', button_type='primary')
        self.button.on_click(lambda event: self.button_stream.event(button=True))
        self.batch_inds = next(iter(self.sampler))
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
        
        # Create dynamic maps for batch plots
        self.batch_inds_dmap = hv.DynamicMap(pn.bind(self.plot_batch_points, 
                                                     checked=self.batch_checkboxes))
                                                    #  trigger=self.button_stream.param.button))
        self.batch_spec_dmap = hv.DynamicMap(pn.bind(self.plot_batch_spectrum, 
                                                     checked=self.batch_checkboxes))
                                                    #  trigger=self.button_stream.param.button) )
        
        # Create dynamic maps for embedding and fits
        self.fit_img_dmap = hv.DynamicMap(pn.bind(self.plot_fits_img, 
                                                  i=self.i_slider, s=self.s_slider, sampler=self.sampler_selector))
        self.fit_img_scaled_dmap = hv.DynamicMap(pn.bind(self.plot_fits_img, 
                                                         i=self.i_slider, s=self.s_slider, sampler=self.sampler_selector))
        
        self.fit_spec_dmap = hv.DynamicMap(pn.bind(self.plot_fit_spectrum, 
                                                   i=self.i_slider, x=self.x_slider, y=self.y_slider, sampler=self.sampler_selector))
        self.fit_spec_dmap = hv.DynamicMap(pn.bind(self.plot_fit_spectrum, 
                                                   i=self.i_slider, x=self.x_slider, y=self.y_slider, sampler=self.sampler_selector))
        
        self.embedding_dmaps = [hv.DynamicMap(pn.bind(self.plot_embedding_img, 
                                                      i=self.i_slider, par=par, sampler=self.sampler_selector)
                                              )*self.dot_dmap for par in range(4)]
        
    @lru_cache(maxsize=10)
    def select_datacube(self,i):
        self.dset.noise_ = self.dset.h5_keys()[i]
        self.dset.scale = False
        return self.dset[:][1] # 100, 100, 500
        
    @lru_cache(maxsize=10)
    def select_datacube_scaled(self,i):
        self.dset.noise_ = self.dset.h5_keys()[i]
        self.dset.scale = True
        return self.dset[:][1] # 100, 100, 500
            
    # @lru_cache(maxsize=10)
    # def select_embedding(i,sampler): # TODO: fix
    #     if sampler == 'scipy':
    #         with h5py.File(f'{datapath}/{sampler}/pv_scipy_fits.h5','a') as f:
    #             key = [k for k in list(f.keys()) if 'emb_' in k][0]
    #             return f[key][:].reshape(100,100,-1)
    #     with h5py.File(f'{datapath}/{sampler}/{dset_list[i]}/embeddings_1D.h5','a') as f:
    #         key = [k for k in list(f.keys()) if 'embedding_' in k][0]
    #         return f[key][:].reshape(100,100,-1) # 10000,1,4
    # @lru_cache(maxsize=10)
    # def select_fits(i,sampler,noise):
    #     # noise_ = i**(1.5)
    #     with h5py.File(f'{datapath}/{sampler}/{dset_list[i]}/embeddings_1D.h5','a') as f:
    #         key = [k for k in list(f.keys()) if 'fits_' in k][0]
    #         return f[key][:].reshape(100,100,-1) # 10000,1,500
    
    ############################################
    
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
    
    def plot_datacube_img_scaled(self, i, s):
        datacube = self.select_datacube(i).reshape(self.dset.shape[0],self.dset.shape[1],self.dset.shape[2])
        data_ = np.flipud(datacube[:, s].reshape(self.shape[0],self.shape[1]).T)
        return hv.Image(data_, bounds=(0,0,self.dset.shape[0],self.dset.shape[1]),
                        kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(
                            cmap='viridis', colorbar=True, clim=(0, datacube.max()),
                            width=350, height=300, title='Datacube Intensity')

    def plot_embedding_img(self, i, par, sampler):
        datacube = self.select_embedding(i,sampler)
        data_ = np.flipud(datacube[:, :, par].T)
        
        return hv.Image(data_, bounds=(0, 0, data_.shape[0], data_.shape[1]),
                        kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(cmap='viridis', colorbar=True, clim=(0, data_.max()),
                            width=350, height=300, title=f'{self.parameters_list[par]}')

    def plot_fits_img(self, i, s, sampler):
        datacube = self.select_fits(i,sampler)
        data_ = np.flipud(datacube[:, :, s].T)
        
        return hv.Image(data_, bounds=(0, 0, data_.shape[0], data_.shape[1]),
                        kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(cmap='viridis', colorbar=True, clim=(0, datacube.max()),
                            width=350, height=300, title='Fitted Intensity')


    def plot_datacube_spectrum(self, i, x, y):
        datacube = self.select_datacube(i).reshape(self.dset.shape)
        
        return hv.Curve(datacube[x, y],
                        kdims=[hv.Dimension('spectrum', label='Spectrum Value')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(width=350, height=300,
                                ylim=(0, datacube.max()), xlim=(0, 500),
                                axiswise=True, shared_axes=False)
    
    def plot_fit_spectrum(self, i, x, y, sampler, noise):
        datacube = self.select_fits(i, sampler, noise)
        return hv.Curve(datacube[x, y],
                        kdims=[hv.Dimension('spectrum', label='Spectrum Value')],
                        vdims=[hv.Dimension('intensity', label='Intensity')],
                        ).opts(width=350, height=300,
                            ylim=(0, datacube.max()), xlim=(0, 500),
                                        axiswise=True, shared_axes=False)

    def show_dot(self, x, y): return hv.Scatter([(x, y)]).opts( color='red', size=5, marker='o',
                                        axiswise=True, shared_axes=False)

    def show_vline(self, s): return hv.VLine(int(s)).opts(
            color='black', line_width=2,
            axiswise=True, shared_axes=False)
        
    def layout_input(self):
        return pn.Column(
        pn.Row(self.i_slider, self.s_slider),
        pn.Row(self.x_slider, self.y_slider),
        (self.img_dmap*self.dot_dmap + \
         self.spec_dmap*self.vline_dmap*self.zero_spec_dmap).opts(shared_axes=True, axiswise=True),
                        )


    def split_list(self,): # trigger if dset changes
        return [self.batch_inds[i:i + self.sampler.num_neighbors] for i in range(0, len(self.batch_inds), self.sampler.num_neighbors)]
    def get_points_idx(self): 
        clumps = self.split_list()
        return [[ (int(ind / self.dset.shape[0]),ind % self.dset.shape[0]
                    ) for ind in clump
                ] for clump in clumps ]
    def get_points_data(self):
        dset = self.select_datacube(self.i_slider.value)
        clumps = self.split_list()
        return [ np.asarray([dset[ind] for ind in clump],dtype=np.float32
                         ) for clump in clumps ]
        
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
    
    def plot_batch_spectrum(self, checked):
        data = self.get_points_data()
        curves_list = []
        for d,dat in enumerate(data):
            if d in checked: curve = hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
                                            color=self.colors[d], alpha=1,
                                            ylim=(0, self.dset.maxes.max()), xlim=(0, 500),
                                            axiswise=True, shared_axes=False, line_width=1)
            else: curve = hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
                                            color=self.colors[d], alpha=0.1,
                                            ylim=(0, self.dset.maxes.max()), xlim=(0, 500),
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
    