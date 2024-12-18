from functools import lru_cache
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
        self.colors = ['green','orange','yellow','brown','pink','gray', 'white', 'magenta', 'cyan','purple']

        if sampler is not None: self.sampler = sampler
        
        # Create interactive widgets
        self.i_slider = pn.widgets.IntSlider(name='Noise std', value=0, start=0, end=len(self.dset_list )-1)
        self.x_slider = pn.widgets.IntSlider(name='x', value=25, start=0, end=dset.shape[0]-1)
        self.y_slider = pn.widgets.IntSlider(name='y', value=25, start=0, end=dset.shape[1]-1)
        self.s_slider = pn.widgets.IntSlider(name='spectral value', value=0, start=0, end=dset.shape[2]-1)
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
                                                     i=self.i_slider, trigger=self.button_stream.param.button))
        self.batch_spec_dmap = hv.DynamicMap(pn.bind(self.plot_batch_spectrum, 
                                                     i=self.i_slider, trigger=self.button_stream.param.button) )
        
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
        # data_ = np.flipud(datacube[:, :, s].T)/
        return hv.Image(
                    datacube[:,:,s], bounds=(0,0,datacube.shape[0],datacube.shape[1]),
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
        
    def plot_batch_points(self,i,trigger):
        pts = self.get_points_idx()                                             
        scatter_list = []
        for p,pt in enumerate(pts):
            scatter_list.append( hv.Scatter(pt).opts( color=self.colors[p], size=3, marker='o',
                                                    axiswise=True, shared_axes=False))
        return hv.Overlay(scatter_list).opts(shared_axes=True, axiswise=True)
    
    def plot_batch_spectrum(self,i,trigger):
        data = self.get_points_data()
        curves_list = []
        for d,dat in enumerate(data):
            curves_list.append(hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
                                                            color=self.colors[d],
                                            ylim=(0, self.dset.maxes.max()), xlim=(0, 500),
                                            axiswise=True, shared_axes=False, 
                                            line_width=1, line_dash='dashed'))
        return hv.Overlay(curves_list).opts(shared_axes=True, axiswise=True) 
    
    def layout_batch(self):
        dmap = pn.Column(#pn.Row(button),
                pn.Row(self.i_slider, self.s_slider),
                pn.Row(self.x_slider, self.y_slider),
                (self.img_dmap*self.dot_dmap*self.batch_inds_dmap + 
                    self.spec_dmap*self.zero_spec_dmap*self.batch_spec_dmap*self.vline_dmap).opts(shared_axes=True,axiswise=True)
            )

        return dmap
    
    
class Py4DSTEM_hv_viz():
    def __init__(self, dataset, model, embedding, generated,):
        self.dataset = dataset
        self.model = model
        self.embedding = embedding
        self.generated = generated
        self.shape = dataset.raw_data.shape
        
        self.avg_diffraction = np.mean(self.dataset.log_data, axis=(0,1))
        self.avg_sample = np.mean(self.dataset.log_data, axis=(2,3))
        
        # Create interactive widgets
        self.emb_slider = pn.widgets.IntSlider(name='Embedding Channel', value=0, 
                                               start=0, end=self.embedding.shape[-1])
        self.x_slider = pn.widgets.IntSlider(name='x', value=25, 
                                             start=0, end=self.shape[0])
        self.y_slider = pn.widgets.IntSlider(name='y', value=25, 
                                             start=0, end=self.shape[1])
        self.a_slider = pn.widgets.IntSlider(name='a', value=0, 
                                             start=0, end=self.shape[2])
        self.b_slider = pn.widgets.IntSlider(name='b', value=0, 
                                             start=0, end=self.shape[3])
        self.gen_slider = pn.widgets.IntSlider(name='Generated Spectrum', value=0, 
                                               start=0, end=generated.shape[0])
        
        # Dynamic maps for the red dot and vertical line
        self.xydot_dmap = hv.DynamicMap(pn.bind(self.xy_dot, x=self.x_slider, y=self.y_slider))
        self.abdot_dmap = hv.DynamicMap(pn.bind(self.ab_dot, x=self.a_slider, y=self.b_slider))

        # self. 
        
    # def xy_dot(self, x, y): 
    #     return hv.Scatter([(x, y)]).opts( color='red', size=5, marker='o',
    #                                         axiswise=True, shared_axes=False)
    # def ab_dot(self, x, y): 
    #     return hv.Scatter([(x, y)]).opts( color='yellow', size=5, marker='o',
    #                                         axiswise=True, shared_axes=False)

    # @lru_cache(maxsize=200)
    # def select_datacube(self,i):
    #     return 
        
    # @lru_cache(maxsize=200)
    # def select_embedding(i,sampler,noise): # TODO: fix
    #     if sampler == 'scipy':
    #         with h5py.File(f'{datapath}/{sampler}/{noise}/pv_scipy_fits.h5','a') as f:
    #             key = [k for k in list(f.keys()) if 'emb_' in k][0]
    #             return f[key][:].reshape(100,100,-1)
    #     with h5py.File(f'{datapath}/{sampler}/{noise}/{dset_list[i]}/embeddings_1D.h5','a') as f:
    #         key = [k for k in list(f.keys()) if 'embedding_' in k][0]
    #         return f[key][:].reshape(100,100,-1) # 10000,1,4
    
    # @lru_cache(maxsize=200)
    # def select_fits(i,sampler,noise):
    #     # noise_ = i**(1.5)
    #     with h5py.File(f'{datapath}/{sampler}/{noise}/{dset_list[i]}/embeddings_1D.h5','a') as f:
    #         key = [k for k in list(f.keys()) if 'fits_' in k][0]
    #         return f[key][:].reshape(100,100,-1) # 10000,1,500
    
    
    # def plot_datacube_img(i, s, noise):
    #     datacube = select_datacube(i, noise)
    #     # data_ = np.flipud(datacube[:, :, s])
    #     data_ = np.flipud(datacube[:, :, s].T)
        
    #     return hv.Image(data_, bounds=(0, 0, data_.shape[0], data_.shape[1]),
    #                     kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
    #                     vdims=[hv.Dimension('intensity', label='Intensity')],
    #                     ).opts(cmap='viridis', colorbar=True, clim=(0, datacube.max()),
    #                         width=350, height=300, title='Datacube Intensity')

    # def plot_embedding_img(i, par, sampler, noise):
    #     datacube = select_embedding(i,sampler,noise)
    #     # data_ = np.flipud(datacube[:, :, s])
    #     data_ = np.flipud(datacube[:, :, par].T)
        
    #     return hv.Image(data_, bounds=(0, 0, data_.shape[0], data_.shape[1]),
    #                     kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
    #                     vdims=[hv.Dimension('intensity', label='Intensity')],
    #                     ).opts(cmap='viridis', colorbar=True, clim=(0, data_.max()),
    #                         width=350, height=300, title=f'{parameters_list[par]}')

    # def plot_fits_img(i, s, sampler, noise):
    #     datacube = select_fits(i,sampler,noise)
    #     # data_ = np.flipud(datacube[:, :, s])
    #     data_ = np.flipud(datacube[:, :, s].T)
        
    #     return hv.Image(data_, bounds=(0, 0, data_.shape[0], data_.shape[1]),
    #                     kdims=[hv.Dimension('x', label='X Position'), hv.Dimension('y', label='Y Position')],
    #                     vdims=[hv.Dimension('intensity', label='Intensity')],
    #                     ).opts(cmap='viridis', colorbar=True, clim=(0, datacube.max()),
    #                         width=350, height=300, title='Fitted Intensity')


    # def plot_datacube_spectrum(i, x, y, noise):
    #     datacube = select_datacube(i, noise)
    #     return hv.Curve(datacube[x, y],
    #                     kdims=[hv.Dimension('spectrum', label='Spectrum Value')],
    #                     vdims=[hv.Dimension('intensity', label='Intensity')],
    #                     ).opts(width=350, height=300,
    #                         ylim=(0, datacube.max()), xlim=(0, 500),
    #                                     axiswise=True, shared_axes=False)
    # def plot_fit_spectrum(i, x, y, sampler, noise):
    #     datacube = select_fits(i, sampler, noise)
    #     return hv.Curve(datacube[x, y],
    #                     kdims=[hv.Dimension('spectrum', label='Spectrum Value')],
    #                     vdims=[hv.Dimension('intensity', label='Intensity')],
    #                     ).opts(width=350, height=300,
    #                         ylim=(0, datacube.max()), xlim=(0, 500),
    #                                     axiswise=True, shared_axes=False)

        

    # # Create dynamic maps for image and spectrum plots
    # img_dmap = hv.DynamicMap(pn.bind(plot_datacube_img, i=i_slider, s=s_slider, noise=noise_selector))
    # fit_img_dmap = hv.DynamicMap(pn.bind(plot_fits_img, i=i_slider, s=s_slider, noise=noise_selector, sampler=sampler_selector))

    # spec_dmap = hv.DynamicMap(pn.bind(plot_datacube_spectrum, i=i_slider, x=x_slider, y=y_slider, noise=noise_selector))
    # zero_spec_dmap = hv.DynamicMap(pn.bind(plot_datacube_spectrum, i=0, x=x_slider, y=y_slider, noise=noise_selector))
    # fit_spec_dmap = hv.DynamicMap(pn.bind(plot_fit_spectrum, i=i_slider, x=x_slider, y=y_slider, noise=noise_selector, sampler=sampler_selector))

    # embedding_dmaps = [hv.DynamicMap(pn.bind(plot_embedding_img, i=i_slider, par=par, noise=noise_selector, sampler=sampler_selector))*dot_dmap for par in range(4)]

    # dataset = Indexing_Dataset(dset.reshape(-1, dset.shape[-1]))
    # sampler = Gaussian_Sampler(dataset, batch_size=100, original_shape=(x_,y_), gaussian_std=5, num_neighbors=10)
    # batch_inds = next(iter(sampler))

    # def retrieve_batch(dset):
    #     global sampler
    #     dataset = Indexing_Dataset(dset.reshape(-1, dset.shape[-1]))
    #     sampler = Gaussian_Sampler(dataset, batch_size=100, original_shape=(x_,y_), gaussian_std=5, num_neighbors=10)
    #     return sampler
    # sampler = Gaussian_Sampler(Indexing_Dataset(dset.reshape(-1, dset.shape[-1])),
    #                             batch_size=100, original_shape=(x_,y_), 
    #                             gaussian_std=5, num_neighbors=10)
    # batch_inds = next(iter(sampler))
    # from cv2 import line


    # def split_list(dset,trigger): 
    #     global batch_inds,sampler
    #     if trigger:
    #         sampler = Gaussian_Sampler(Indexing_Dataset(dset.reshape(-1, dset.shape[-1])),
    #                                 batch_size=100, original_shape=(x_,y_), gaussian_std=5, num_neighbors=10)
    #         batch_inds = next(iter(sampler))
    #     return [batch_inds[i:i + sampler.num_neighbors] for i in range(0, len(batch_inds), sampler.num_neighbors)]

    # colors = ['green','orange','yellow','brown','pink','gray', 'white', 'magenta', 'cyan','purple']

    # def plot_batch(i,noise,trigger=False):
    #     dset = select_datacube(i,noise)
    #     clumps = split_list(dset,trigger)
    #     pts = [ [(int(ind / x_), ind % x_) for ind in clump] for clump in clumps ]
    #     scatter_list = []
    #     for p,pt in enumerate(pts):
    #         scatter_list.append( hv.Scatter(pt).opts( color=colors[p], size=3, marker='o',
    #                                                 axiswise=True, shared_axes=False))
    #     return hv.Overlay(scatter_list).opts(shared_axes=True, axiswise=True)
    
    # def plot_mean_spectrum(i,noise,trigger=False):
    #     dset = select_datacube(i,noise)
    #     clumps = split_list(dset,trigger)
    #     data = [ np.array([dset[int(ind / x_), ind % x_] for ind in clump]) for clump in clumps ]
    #     curves_list = []
    #     for d,dat in enumerate(data):
    #         curves_list.append(hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
    #                                                         color=colors[d],
    #                                         ylim=(0, dset.max()), xlim=(0, 500),
    #                                         axiswise=True, shared_axes=False, 
    #                                         line_width=1, line_dash='dashed'))
    #     return hv.Overlay(curves_list).opts(shared_axes=True, axiswise=True)


    # class ButtonStream(streams.Stream):
    #     button = param.Boolean(default=False)

    # button_stream = ButtonStream()

    # def trigger_button(event):
    #     button_stream.button = not button_stream.button

    # button = pn.widgets.Button(name='Trigger', button_type='primary')
    # button.on_click(trigger_button)

    # pn.Row(button)

    # batch_inds_dmap = hv.DynamicMap(pn.bind(plot_batch, i=i_slider, noise=noise_selector, trigger=button_stream.param.button))
    # batch_spec_dmap = hv.DynamicMap(pn.bind(plot_mean_spectrum, i=i_slider, noise=noise_selector, trigger=button_stream.param.button) )   