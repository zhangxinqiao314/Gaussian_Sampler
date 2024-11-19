from cProfile import label
from turtle import color
from urllib import response
from m3_learning.be import dataset
from m3_learning.util.file_IO import make_folder
from m3_learning.viz.layout import layout_fig
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from m3_learning.viz.layout import layout_fig, imagemap, labelfigs, find_nearest, add_scalebar
from os.path import join as pjoin
from m3_learning.viz.nn import embeddings as embeddings_
from m3_learning.viz.nn import affines as affines_
import glob
import os
import h5py
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib.image as mpimg
import panel as pn
  
from m3_learning.nn.STEM_AE_multimodal.Dataset import STEM_EELS_Dataset, EELS_Embedding_Dataset
from m3_learning.nn.STEM_AE_multimodal.STEM_AE import FitterAutoencoder_1D
import functools
import holoviews as hv
from holoviews import opts
from holoviews.streams import Stream, Tap
import panel as pn
hv.extension('bokeh')
pn.extension('bokeh')

import time

# # Profiling decorator
# def profile(func):
#     return wrapper

from holoviews.operation.datashader import rasterize
import holoviews as hv
import datashader as ds
import datashader.transfer_functions as tf
from holoviews.operation.datashader import shade, datashade, rasterize
from holoviews.streams import Tap
hv.extension('bokeh')  # or 'matplotlib' if you prefer

class Viz_affine_AE_hv():
    def __init__(self,dset,model,embedding,channel_type='all',active_threshold=0.1):
        '''
        '''
        self.dset = dset
        self.model = model
        self.embedding = embedding
        self.active_threshold = active_threshold
        
        self.affine_labels = [ 'x scaling', 'y scaling', 'x translation', 'y translation', 'rotation', 'shear']

        self.channels = {'all': list(range(model.num_fits)),
                         'non-zeros': list(embedding.get_active_channels(self.active_threshold))}
         
        # Define Sliders
        self.p_select = pn.widgets.Select(name='Particle', options=self.particle_dict)  # Replace with actual options
        self.e_select = pn.widgets.Select(name='EELS Channel', options=[k for k in range(dset.eels_chs)])  # Replace with actual options
        self.c_select = pn.widgets.DiscreteSlider(name='Emb Channel', value=self.channels[channel_type][0], options=self.channels[channel_type])  # Replace with actual options
        self.s_select = pn.widgets.DiscreteSlider(name='Spectral Value', options=[s for s in range(dset.spec_len)])  # Replace with actual options
        
        # Define Stream tools
        # multiply to image you wanna tap on
        self.max_ =  max([s[0] for s in self.dset.meta['shape_list']])
        self.blank_img = hv.Image( np.zeros((self.max_,self.max_))
                                ).opts(tools=['tap'], axiswise=True, shared_axes=False)
        # self.blank_spec = hv.Curve( np.zeros(dset.spec_len)
        #                         ).opts(tools=['tap'], axiswise=True, shared_axes=False)
        
        # use as input on dynamic maps
        self.point_stream = Tap( source=self.blank_img, x=0, y=0 )
        # self.spectrum_stream = Tap(source=self.blank_spec, x=0) 
        
        # multiply to dynamic maps you wanna display on
        self.dot_overlay = hv.DynamicMap(pn.bind(self._show_dot, p=self.p_select), streams=[self.point_stream]
                                    ).opts(axiswise=True, shared_axes=False)
        self.vline_overlay = hv.DynamicMap(pn.bind(self._show_vline, x=self.s_select),
                                            ).opts(xlim=(0, self.dset.spec_len), ylim=(0, 1),
                                                    axiswise=True, shared_axes=False)
            
        self.coef1_scale = np.linspace(0,model.loss_params['coef_1'],model.num_fits)  
        self.xticks = [ [(i, np.round(label,2)) for i, label in zip(np.arange(dset.spec_len), 
                                                        dset.meta['eels_axis_labels'][0]) ],
                        [(i, np.round(label,2)) for i, label in zip(np.arange(dset.spec_len), 
                                                        dset.meta['eels_axis_labels'][1]) ] ]
        self.x_labels = [ {i:v for i,v in enumerate(dset.meta['eels_axis_labels'][0])}, 
                          {i:v for i,v in enumerate(dset.meta['eels_axis_labels'][0])} ]
        self.p_cache = {}
        self.e_cache = {}
    
    @profile  
    def _show_dot(self, p, x, y): return hv.Scatter([(int(x), int(y))]).opts(xlim=(0, self.dset.meta['shape_list'][p][0]), 
                                                ylim=(0, self.dset.meta['shape_list'][p][1]), 
                                                color='red', size=5, marker='o',
                                                axiswise=True, shared_axes=False)
    @profile
    def _show_vline(self, x): return hv.VLine(int(x)).opts(xlim=(0, self.dset.spec_len), ylim=(0, 1), 
                                        color='black', line_width=2,
                                        axiswise=True, shared_axes=False)

   # update data and manage cached particles ################ 
    @profile  
    def _data(self, p): 
        if p in self.p_cache: return self.p_cache[p]
        else:
            _, eels = self.dset[self.dset.meta['particle_inds'][p]:self.dset.meta['particle_inds'][p+1]]
            self.p_cache[p] = eels
            return eels
    @profile  
    def data(self, p, e, **kwargs): 
        return self._data(p)[:,e]
    @profile 
    def _emb(self, p): 
        if p in self.e_cache: return self.e_cache[p]
        else:
            emb = self.embedding[self.dset.meta['particle_inds'][p]:self.dset.meta['particle_inds'][p+1]]
            self.e_cache[p] = emb
            return emb
        # (14400,2,96,6), (14400,2,96,969) 
    @profile     
    def emb(self, p, e, **kwargs): 
        emb,fits =  self._emb(p)
        return emb[:,e], fits[:,e] 
        # (14400,96,6), (14400,96,969) 
    @profile         
    def emb_channel(self,p,e,c,**kwargs):
        emb,fits = self._emb(p)
        return emb[:,e,c], fits[:,e,c]
        # return emb[:,c],fits[:,c] 
        # (14400,6), (14400,969) 
    
    def Re_e_Im_e_(self,p,e,c,x,y):
        idx = np.ravel_multi_index((int(x), int(y)),self.dset.meta['shape_list'][p])
        emb,fits = self.emb_channel(p,e,c)
        fits = fits[idx]
        # Define the frequency (omega) range for which you want to compute epsilon_1
        x_ = np.arange(fits.shape[-1])
        # Define the integrand for the Kramers-Kronig relation
        integrand = []
        for x in x_: 
            x_new = x_
            x_new[x]=0  # avoid pole
            integrand.append(2/np.pi * np.trapz(fits* x /(x_new**2 - x**2))) # (fits are the eps 2 values)
        return np.array(integrand),fits
    
    
         
    # Image objects #########################################
    @profile  
    def mean_image(self,p,e, **kwargs): 
        return hv.Image(self.dset.get_mean_image(p,e), bounds=((0,0,)+self.dset.meta['shape_list'][p])
                        ).opts(width=350, height=300, 
                               cmap='viridis', colorbar=True,
                               axiswise=True, shared_axes=False,
                               title = f"Mean image {self.dset.meta['particle_list'][p]}")            
    @profile 
    def input_image(self,p,e, x, **kwargs): 
        return hv.Image( self.data(p,e)[:, int(x)].reshape(self.dset.meta['shape_list'][p]),
                        bounds=((0,0,self.max_,self.max_))
                        ).opts(axiswise=True, shared_axes=False,
                               title = f"Input at {self.xticks[e][int(x)][1]:.1f} eV" 
                               )
    @profile 
    def mean_emb_image(self,p,e,x=0,**kwargs):
        _,fits = self.emb(p,e) # (14400,96,969)
        return hv.Image(fits[:,:,int(x)].sum(1).reshape(self.dset.meta['shape_list'][p]), 
                        bounds=(0,0)+self.dset.meta['shape_list'][p]
                        ).opts(width=350, height=300, 
                               cmap='viridis', colorbar=True, 
                               clim=(fits.min(), fits.max()),
                               tools=['tap'], 
                               axiswise=True, shared_axes=False, 
                               title=f'Fitted at {self.xticks[e][int(x)][1]:.1f} eV')    
    @profile
    def emb_particles(self,p,e,c,x=0,**kwargs):
        _,fits = self.emb_channel(p,e,c) # (14400,96,969)
        return hv.Image(fits[:,int(x)].reshape(self.dset.meta['shape_list'][p]), 
                        bounds=(0,0)+self.dset.meta['shape_list'][p]
                        ).opts(width=350, height=300, 
                               cmap='viridis', colorbar=True, 
                               clim=(fits.min(), fits.max()),
                               tools=['tap'], 
                               axiswise=True, shared_axes=False, 
                               title=f'Fitted at {self.xticks[e][int(x)][1]:.1f} eV, Channel {c}, coef1: {self.coef1_scale[c]:.2e}')
     
    
    # Spectrum objects ######################################
    @profile          
    def mean_spectrum(self,p,e, **kwargs):
        mean_spectrum = self.data(p,e).mean(axis=0)
        return hv.Curve(mean_spectrum
                        ).opts(tools=['tap'], 
                               color='blue', line_width=1,
                               axiswise=True, shared_axes=False, 
                               xlabel='Loss (eV)', ylabel='Intensity',
                               xticks=self.xticks[e][::240],
                               title='Mean Spectrum')
    @profile     
    def input_spectrum(self,p,e, x, y, **kwargs): 
        return hv.Curve( self.data(p,e).reshape(self.dset.meta['shape_list'][p]+(-1,))[int(x), int(y), :]
                        ).opts(axiswise=True, shared_axes=False, 
                               xlabel='Loss (eV)', ylabel='Intensity',
                               xticks=self.xticks[e][::240],
                               color='blue', line_width=1,
                               title = f"Spec at: ({int(x)},{int(y)})" )
    @profile 
    def mean_emb_fits(self,p,e,x=0,y=0):
        _,fits = self.emb(p,e) # (14400,96,969)
        idx = np.ravel_multi_index((int(x),int(y)),(self.dset.meta['shape_list'][p]))
        fits = hv.Curve( fits[idx].sum(axis=0)
                        ).opts(axiswise=True, shared_axes=False, 
                               xlabel='Loss (eV)', ylabel='Intensity',
                               xticks=self.xticks[e][::240],
                               color='red', line_width=1.5,
                               )
        return fits
    @profile
    def emb_fits(self,p,e,c,x=0,y=0):
        idx = np.ravel_multi_index((int(x),int(y)),(self.dset.meta['shape_list'][p]))
        _,fits = self.emb_channel(p,e,c) # (14400,96,969) 
        fits = fits[idx]
        fits = hv.Curve(fits
                        ).opts(axiswise=True, shared_axes=False, 
                               xlabel='Loss (eV)', ylabel='Intensity',
                               xticks=self.xticks[e][::240],
                               color='red', line_dash='dashed',
                               tools=['tap'],
                               title=f'Raw and fitted Spectra ({int(x)},{int(y)})')

        return fits
    
    def dielectric_spectrum(self,p,e,c,x,y):
        Re,Im = self.Re_e_Im_e_(p,e,c,x,y)
        eps_1 = Re/(Re**2+Im**2)
        eps_2 = Im/(Re**2+Im**2)
        
        epsilon_1 = hv.Curve(eps_1
                ).opts(axiswise=True, shared_axes=False, 
                        xlabel='Loss (eV)', ylabel='Intensity',
                        xticks=self.xticks[e][::240],
                        color='orange', line_dash='solid',                         
                        title=f'Dielectric Response ({int(x)},{int(y)})')
        epsilon_2 = hv.Curve(eps_2
                ).opts(axiswise=True, shared_axes=False, 
                        xlabel='Loss (eV)', ylabel='Intensity',
                        xticks=self.xticks[e][::240],
                        color='black', line_dash='solid',
                        
                        )
        Re = hv.Curve(Re
                ).opts(axiswise=True, shared_axes=False, 
                        xlabel='Loss (eV)', ylabel='Intensity',
                        xticks=self.xticks[e][::240],
                        color='orange', line_dash='solid',
                        )
        Im = hv.Curve(Im
                ).opts(axiswise=True, shared_axes=False, 
                        xlabel='Loss (eV)', ylabel='Intensity',
                        xticks=self.xticks[e][::240],
                        color='Black', line_dash='solid',
                        )

        return epsilon_1*epsilon_2 + Re*Im

    # Parameter objects #####################################
    @profile 
    def mean_emb_parameters(self,p,e,x,y,par,**kwargs):
        emb,_ = self.emb(p,e) # (14400,96,969)
        if par==0 or par==3: mean_par = emb[:,:,par].sum(1)
        else: mean_par = emb[:,:,par].mean(1)
        idx = np.ravel_multi_index((int(x), int(y)),self.dset.meta['shape_list'][p])
        return hv.Image(mean_par.reshape(self.dset.meta['shape_list'][p]), 
                        bounds=(0,0)+self.dset.meta['shape_list'][p]
                        ).opts(colorbar=True,
                               clim=(mean_par.min(),mean_par.max()),
                               axiswise=True, shared_axes=False, 
                               title=f'{self.parameter_labels[par]}: {mean_par[idx]:.3e}' ) 
    @profile                      
    def emb_parameters(self,p,e,c,x,y,par,**kwargs):
        idx = np.ravel_multi_index((int(x), int(y)),self.dset.meta['shape_list'][p])
        emb,_ = self.emb_channel(p,e,c) # (14400,969)
        emb = emb[:,par]
        return hv.Image(emb.reshape(self.dset.meta['shape_list'][p]), 
                        bounds=(0,0)+self.dset.meta['shape_list'][p]
                        ).opts(colorbar=True,
                               clim=(emb.min(),emb.max()),
                               axiswise=True, shared_axes=False, 
                               title=f'{self.parameter_labels[par]}: {emb[idx]:.3e}' ) 
   
    
    # Image dmaps ###########################################
    @profile          
    def mean_input_image_dmap(self):
        return hv.DynamicMap(pn.bind(self.mean_image, p=self.p_select, e=self.e_select)
                                ).opts(width=350, height=300, cmap='viridis', colorbar=True, 
                                        axiswise=True, shared_axes=False)
    @profile 
    def input_image_dmap(self):
        return hv.DynamicMap(pn.bind(self.input_image, p=self.p_select, e=self.e_select, x=self.s_select),
                            ).opts(width=350, height=300, 
                                   cmap='viridis', colorbar=True,
                                   axiswise=True, shared_axes=False)
    @profile 
    def mean_emb_dmap(self):
        return hv.DynamicMap(pn.bind(self.mean_emb_image, p=self.p_select, e=self.e_select), 
                                 streams=[self.spectrum_stream]
                                ).opts(width=350, height=300, cmap='viridis', colorbar=True,
                                        axiswise=True, shared_axes=False, tools=['tap'])
    @profile
    def emb_dmap(self): 
        return hv.DynamicMap(pn.bind(self.emb_particles, 
                                     p=self.p_select, e=self.e_select, c=self.c_select, x=self.s_select), 
                            ).opts(width=350, height=300, cmap='viridis', colorbar=True, colorbar_opts={'width': 10},
                                    axiswise=True, shared_axes=False, tools=['tap'])
     
    
    # Spectrum dmaps ########################################
    @profile      
    def mean_input_spectrum_dmap(self):
        return hv.DynamicMap(pn.bind(self.mean_spectrum, p=self.p_select, e=self.e_select)
                                ).opts(
                                       width=350, height=300,
                                       axiswise=True, shared_axes=False)    
    @profile 
    def input_spectrum_dmap(self):
        return hv.DynamicMap(pn.bind(self.input_spectrum, p=self.p_select, e=self.e_select), 
                             streams=[self.point_stream]
                            ).opts(
                                   width=350, height=300, 
                                   axiswise=True, shared_axes=False)
    @profile 
    def mean_fits_dmap(self):
        return hv.DynamicMap(pn.bind(self.mean_emb_fits, p=self.p_select, e=self.e_select), 
                             streams=[self.point_stream]
                            ).opts(
                                   width=350, height=300, 
                                   axiswise=True, shared_axes=False, 
                                   tools=['tap'])     
    @profile
    def fits_dmap(self):
        return hv.DynamicMap(pn.bind(self.emb_fits, 
                                     p=self.p_select, e=self.e_select, c=self.c_select), 
                             streams=[self.point_stream]
                            ).opts(
                                   axiswise=True, shared_axes=False, 
                                   width=350, height=300, 
                                   tools=['tap'])
        
    def dielectric_dmap(self):
        return hv.DynamicMap(pn.bind(self.dielectric_spectrum, 
                                     p=self.p_select, e=self.e_select, c=self.c_select), 
                             streams=[self.point_stream]
                            ).opts(
                                   axiswise=True, shared_axes=False, 
                                   width=350, height=300,
                                #     legend_position='top_left'
                                   )
    
    
    # Parameter dmaps #######################################
    @profile 
    def mean_parameter_dmap(self,par):
        return hv.DynamicMap( pn.bind(self.mean_emb_parameters, p=self.p_select, e=self.e_select, par=par), 
                                    streams=[self.point_stream]
                                    ).opts(width=250, height=225, 
                                           cmap='viridis', colorbar_opts={'width': 5},
                                           axiswise=True, shared_axes=False )
    @profile
    def parameter_dmaps(self, par):
        return hv.DynamicMap( pn.bind(self.emb_parameters, 
                                    p=self.p_select, e=self.e_select, c=self.c_select, par=par), 
                                streams=[self.point_stream]
                            ).opts(width=250, height=225, cmap='viridis', colorbar_opts={'width': 5},
                                axiswise=True, shared_axes=False )
       
    
    
    
    # Servables #############################################
    @profile          
    def visualize_input_mean(self):
        return pn.Column( pn.Row(self.p_select, self.e_select,),
                self.mean_input_image_dmap() + self.mean_input_spectrum_dmap()).servable()
    @profile         
    def visualize_input_at(self):
        processed_panel = pn.Column( 
            pn.Row(self.p_select, self.e_select,),
            
            ( (self.mean_input_image_dmap() *self.blank_img *self.dot_overlay ).opts(axiswise=True) + \
                (self.mean_input_spectrum_dmap() *self.blank_spec *self.vline_overlay ).opts(axiswise=True) ),
            
            ( (self.input_image_dmap() *self.blank_img *self.dot_overlay ).opts(axiswise=True) + \
                (self.input_spectrum_dmap() *self.blank_spec *self.vline_overlay ).opts(axiswise=True) ) )
        return processed_panel
    # @profile
    # def visualize_embedding_mean(self,view_channels='active'): 
    #     '''
    #     {'active', 'sparse'}
    #     '''        
    #     input_dmap = (self.input_image_dmap() *self.blank_img *self.dot_overlay ).opts(axiswise=True)
    #     emb_dmap = (self.mean_emb_dmap() *self.blank_img *self.dot_overlay).opts(axiswise=True)
    #     specs_dmap = (self.input_spectrum_dmap()*self.mean_fits_dmap() *self.blank_spec *self.vline_overlay).opts(axiswise=True)
        
    #     mean_parameters_panel = pn.Column(
    #         pn.Row(self.p_select, self.e_select),
            
    #         (input_dmap + emb_dmap + specs_dmap),
            
    #         hv.Layout( [self.mean_parameter_dmap(par)*self.blank_img *self.dot_overlay \
    #             for par in list(range(self.model.num_params))] ).cols(4) 
    #         )

    #     return mean_parameters_panel  
    @profile
    def visualize_embedding_mean(self,view_channels='active'): 
        '''
        {'active', 'sparse'}
        '''
        input_dmap = (self.input_image_dmap() *self.blank_img *self.dot_overlay ).opts(axiswise=True)
        emb_dmap = (self.mean_emb_dmap() *self.blank_img *self.dot_overlay).opts(axiswise=True)
        specs_dmap = (self.input_spectrum_dmap()*self.mean_fits_dmap() *self.blank_spec *self.vline_overlay).opts(axiswise=True)
        
        mean_parameters_panel = pn.Column(
            pn.Row(self.p_select, self.e_select),
            
            (input_dmap + emb_dmap + specs_dmap),
            
            hv.Layout( [self.mean_parameter_dmap(par)*self.blank_img *self.dot_overlay \
                for par in list(range(self.model.num_params))] ).cols(4) 
            )

        return mean_parameters_panel  
    @profile                                             
    def visualize_embedding(self,view_channels='active'):
    # visualize_embedding took 0.0571 seconds
        '''
        {'active', 'sparse'}
        '''
        input_dmap = (self.input_image_dmap() *self.blank_img *self.dot_overlay ).opts(axiswise=True)
        emb_dmap = (self.emb_dmap() *self.blank_img *self.dot_overlay).opts(axiswise=True)
        input_specs_dmap = (self.input_spectrum_dmap()*self.mean_fits_dmap()*self.fits_dmap() *self.vline_overlay).opts(framewise=True,axiswise=True)
        # input_specs_dmap = (self.input_spectrum_dmap()*self.blank_spec *self.vline_overlay).opts(axiswise=True)
        # input_specs_dmap = (self.mean_fits_dmap() *self.blank_spec *self.vline_overlay).opts(axiswise=True)
        # input_specs_dmap = (self.fits_dmap() *self.blank_spec *self.vline_overlay).opts(axiswise=True)

        parameter_dmaps_by_channel = pn.Column( 
            pn.Row(self.p_select, self.e_select, self.s_select),
            pn.Row(self.c_select),
                
            (input_dmap + emb_dmap + input_specs_dmap),
            
            hv.Layout( [self.parameter_dmaps(par) *self.blank_img *self.dot_overlay \
                for par in list(range(self.model.num_params))] ).cols(4) )
        
        # parameter_dmaps_by_channel = ( 
        #     pn.Row(self.p_select, self.e_select, self.c_select) +\
        # combined_layout = (        
        #     (input_dmap + emb_dmap + input_specs_dmap) +\
            
        #     hv.Layout( [self.parameter_dmaps(par) *self.blank_img *self.dot_overlay \
        #         for par in list(range(self.model.num_params))] ).cols(4) )

        return parameter_dmaps_by_channel
     
        #  # Add the Panel selectors as a separate row using pn.Row
        # panel_controls = pn.Row(self.p_select, self.e_select, self.c_select)

        # # Return the combined layout with selectors as a Panel Column
        # return pn.Column(panel_controls, combined_layout)
     
     
    def visualize_dielectric(self):
        input_dmap = (self.input_image_dmap() *self.blank_img *self.dot_overlay ).opts(axiswise=True)
        emb_dmap = (self.emb_dmap() *self.blank_img *self.dot_overlay).opts(axiswise=True)
        # e1e2,ReIm = list(self.dielectric_dmap().layout())
        # dielectric_dmap = (e1e2*self.blank_spec *self.vline_overlay).opts(axiswise=True,)
        # response_dmap = (ReIm*self.blank_spec *self.vline_overlay).opts(axiswise=True,)
        dielectric_dmaps = self.dielectric_dmap()
        
        dielectric_dmaps_by_channel = pn.Column( 
            pn.Row(self.p_select, self.e_select, self.c_select),
                
            (input_dmap + emb_dmap ),
            
            # (dielectric_dmap + response_dmap)
            dielectric_dmaps
        )
        return dielectric_dmaps_by_channel
  

# class Viz_Gaussian_Sampler_hv():
#     def __init__(self, dset, sampler):
#         self.dset = dset
#         self.sampler = sampler
#         self._batch_inds = next(iter(self.sampler))
        
#         self.colors = ['green','orange','yellow','brown','pink','gray', 'white', 'magenta', 'cyan','purple']
        
#         # self.cache = {}
        
#     @property
#     def batch_inds(self): return next(iter(self.sampler))

#     def _retrieve_batch(self,dset):
#         return [dset[ind] for ind in self._batch_inds]

#    # update data and manage cached particles ################ 
#     # @profile  
#     def _data(self, p): 
#         if p in self.p_cache: return self.p_cache[p]
#         else:
#             _, eels = self.dset[self.dset.meta['particle_inds'][p]:self.dset.meta['particle_inds'][p+1]]
#             self.p_cache[p] = eels
#             return eels
#     # @profile  
#     def data(self, p, e, **kwargs): 
#         return self._data(p)[:,e]
    
    
#     def plot_batch(self,i,noise,trigger=False):
#         dset = self.data()
#         clumps = self.sampler.split_list(self.batch_inds)
#         p_inds,shps = zip(([self.sampler._which_particle_shape(inds[0]) for inds in clumps]))
#         pts = [ [(int(ind / shps[i][0]), ind % shps[i][0]) for ind in clump] for i, clump in enumerate(clumps) ]
#         scatter_list = []
#         for p,pt in enumerate(pts):
#             scatter_list.append( hv.Scatter(pt).opts( color=colors[p], size=3, marker='o',
#                                                     axiswise=True, shared_axes=False))
#         return hv.Overlay(scatter_list).opts(shared_axes=True, axiswise=True)
    
#     def plot_mean_spectrum(i,noise,trigger=False):
#         dset = select_datacube(i,noise)
#         clumps = split_list(dset,trigger)
#         data = [ np.array([dset[int(ind / x_), ind % x_] for ind in clump]) for clump in clumps ]
#         curves_list = []
#         for d,dat in enumerate(data):
#             curves_list.append(hv.Curve(dat.mean(axis=0)).opts(width=350, height=300,
#                                                             color=colors[d],
#                                             ylim=(0, dset.max()), xlim=(0, 500),
#                                             axiswise=True, shared_axes=False, 
#                                             line_width=1, line_dash='dashed'))
#         return hv.Overlay(curves_list).opts(shared_axes=True, axiswise=True)
    
    
#     class ButtonStream(streams.Stream):
#         button = param.Boolean(default=False)

#     button_stream = ButtonStream()

#     def trigger_button(event):
#         button_stream.button = not button_stream.button

#     button = pn.widgets.Button(name='Trigger', button_type='primary')
#     button.on_click(trigger_button)

#     pn.Row(button)

#     batch_inds_dmap = hv.DynamicMap(pn.bind(plot_batch, i=i_slider, noise=noise_selector, trigger=button_stream.param.button))
#     batch_spec_dmap = hv.DynamicMap(pn.bind(plot_mean_spectrum, i=i_slider, noise=noise_selector, trigger=button_stream.param.button) )
#     # Layout with widgets and plots
#     dmap = pn.Column(pn.Row(button),
#         pn.Row(i_slider, s_slider, noise_selector),
#         pn.Row(x_slider, y_slider, sampler_selector),
#         (img_dmap*dot_dmap*batch_inds_dmap + 
#             spec_dmap*batch_spec_dmap*vline_dmap).opts(shared_axes=True,axiswise=True)
#     )

#     dmap

      
      
def imshow_tensor(x):
    import matplotlib.pyplot as plt
    plt.imshow(x.detach().cpu().numpy());
    plt.colorbar()
    plt.show()