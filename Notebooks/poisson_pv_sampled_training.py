######################################################## Todo: set seed for dset generation and training
# Inputs
########################################################
device = input('Enter the device to use (cuda:0, cuda:1, cpu): ')

########################################################
# Utility/imports
########################################################
import numpy as np
import sys
sys.path.append('..') # path to the src directory
sys.path.append('/home/xinqiao/new_mount/gaussian_sampler/M3Learning-Util/src')
sys.path.append('/home/xinqiao/new_mount/gaussian_sampler/AutoPhysLearn/src')
import wandb

########################################################
# Load and write data
########################################################
from Gaussian_Sampler.models.pv_fitter import pseudovoigt_1D_fitters_new
fitter = pseudovoigt_1D_fitters_new(limits = [1,1,750])
from Gaussian_Sampler.data.datasets import Poisson_Sampled_PV_Dataset
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
datapath = '/home/xinqiao/new_mount/gaussian_sampler/'

dset = Poisson_Sampled_PV_Dataset(shape=(100,100,750),
                       save_folder=datapath+'toy_dataset/max_norm/',
                       pv_fitter=fitter,
                       num_classes=5,
                       num_curves=3,
                       scaler=Normalizer(norm='max'),
                    #    overwrite=True,
                       norm_calculation=lambda x: np.linalg.norm(x, axis=-1, ord=np.inf),
                       )

########################################################
# Train models for gaussian sampler
########################################################
from Gaussian_Sampler.models.pv_fitter import Fitter_AE, pseudovoigt_1D_fitters, pseudovoigt_1D_fitters_new
from autophyslearn.spectroscopic.nn import block_factory, Conv_Block, FC_Block
from autophyslearn.spectroscopic.nn import Multiscale1DFitter
from Gaussian_Sampler.data.custom_sampler import Gaussian_Sampler
num_fits = 16 # number of curves to sum up
num_params = 4 # number of parameters to fit

# for std in [1,3,5]:
#     for nn in [5,10,20]:
#         if (std == 1) and (nn == 20 or nn == 10): # skip these combinations
#             print(f'Skipping gaussian_std {std} and num_neighbors {nn}')
#             continue
#         print(f'Training model for gaussian_std {std} and num_neighbors {nn}')
#         for i in range(10):
#             dset.dset_index=i
#             config_ = {'sampling_type': 'gaussian',
#                         'name': dset.dset_name,
#                         'gaussian_std': std, 
#                         'num_neighbors': nn, }
            
#             name = f'{i:02d}_{config_["sampling_type"]}_std:{config_["gaussian_std"]}_nn:{config_["num_neighbors"]}'
#             model = Fitter_AE(function=pseudovoigt_1D_fitters_new,
#                             dset=dset,
#                             num_params=num_params,
#                             num_fits=num_fits,
#                             checkpoints_label=name[3:],
#                             input_channels = 1,
#                             learning_rate=5e-6,
#                             device=device,
#                             encoder = Multiscale1DFitter,
#                             encoder_params = {
#                                 "model_block_dict": { # factory wrapper for blocks
#                                         "hidden_x1": block_factory(Conv_Block)(output_channels_list=[128,64,32], 
#                                                                                 kernel_size_list=[3,3,3], 
#                                                                                 pool_list=[128,64], 
#                                                                                 max_pool=False),
#                                         "hidden_xfc": block_factory(FC_Block)(output_size_list=[64,32]),
#                                         "hidden_x2": block_factory(Conv_Block)(output_channels_list=[32,16,8], 
#                                                                                 kernel_size_list=[3,3,3], 
#                                                                                 pool_list=[64,32], 
#                                                                                 max_pool=True),
#                                         "hidden_embedding": block_factory(FC_Block)(output_size_list=[8*num_fits,num_params*num_fits], last=True),
#                                     },
#                                     # TEST: LIMITS,
#                                     "skip_connections": {'hidden_xfc': 'hidden_embedding'},
#                                     "function_kwargs": {'limits': [1,dset.shape[-1],dset.shape[-1]] }
#                                 },
#                                 sampler = Gaussian_Sampler,
#                                 sampler_params = {'dset': dset, 
#                                                 'batch_size': 100, 
#                                                 'gaussian_std': std, 
#                                                 'orig_shape': dset.shape[0:-1], 
#                                                 'num_neighbors': nn },
#                             )

#             print(f'Training model for noise level {i}')
#             wandb.init(project='poisson_pv_sampled_training', 
#                     group='sampling:10^(-(i/10)_bkg_noise:0.1',
#                     name=f'{i:02d}_{config_["sampling_type"]}_std:{config_["gaussian_std"]}_nn:{config_["num_neighbors"]}', 
#                         config=config_) # later change config for regularization
#             model.train(epochs=51,save_every=50, log_wandb=True)
#             wandb.finish()

########################################################
# Train models for random sampler
########################################################
for i in range(6,10):
    dset.dset_index=i
    config_ = {'sampling_type': 'random',
                'name': dset.dset_name}
    
    model = Fitter_AE(function=pseudovoigt_1D_fitters_new,
                    dset=dset,
                    num_params=num_params,
                    num_fits=num_fits,
                    checkpoints_label=f'{config_["sampling_type"]}_std:NaN_nn:NaN',
                    input_channels = 1,
                    learning_rate=5e-6,
                    device=device,
                    encoder = Multiscale1DFitter,
                    encoder_params = {
                        "model_block_dict": { # factory wrapper for blocks
                                "hidden_x1": block_factory(Conv_Block)(output_channels_list=[128,64,32],
                                                                        kernel_size_list=[3,3,3], 
                                                                        pool_list=[128,64], 
                                                                        max_pool=False),
                                "hidden_xfc": block_factory(FC_Block)(output_size_list=[64,32]),
                                "hidden_x2": block_factory(Conv_Block)(output_channels_list=[32,16,8], 
                                                                        kernel_size_list=[3,3,3], 
                                                                        pool_list=[64,32], 
                                                                        max_pool=True),
                                "hidden_embedding": block_factory(FC_Block)(output_size_list=[8*num_fits,num_params*num_fits], last=True),
                            },
                            # TEST: LIMITS,
                            "skip_connections": {'hidden_xfc': 'hidden_embedding'},
                            "function_kwargs": {'limits': [1,dset.shape[-1],dset.shape[-1]] }
                        }
                    )
    

    print(f'Training model for noise level {i}')
    wandb.init(project='poisson_pv_sampled_training', 
               group='sampling:10^(-(i/10)_bkg_noise:0.1',
               name=f'{i:02d}_{config_["sampling_type"]}_std:NaN_nn:NaN', 
                config=config_) # later change config for regularization
    model.train(epochs=51,save_every=50, log_wandb=True)
    wandb.finish()