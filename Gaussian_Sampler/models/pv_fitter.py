from random import shuffle
import sys
sys.path.append('/home/m3learning/Northwestern/M3Learning-Util/src')
from STEM_EELS_Curve_Fitting.Data import Sampler
from m3util.utils.IO import make_folder

from AutoPhysLearn.src.autophyslearn.spectroscopic.nn import Multiscale1DFitter, Conv_Block, FC_Block, block_factory

from ..data.custom_sampler import Gaussian_Sampler, custom_collate_fn

import torch
from torch import nn
from torch.utils.data import DataLoader

from datetime import date

class Fitter_AE:
    def __init__(self,
                 function, 
                 dset,
                 num_params,
                 num_fits,
                 limits,
                 device='cuda:0',
                 flatten_from = 1,
                 encoder = Multiscale1DFitter,
                 encoder_params = { "model_block_dict": { 
                    "hidden_x1": block_factory(Conv_Block)(output_channels_list=[8,6,4], 
                                                           kernel_size_list=[7,7,5], 
                                                           pool_list=[64], 
                                                           max_pool=False),
                    "hidden_xfc": block_factory(FC_Block)(output_size_list=[64,32,20]),
                    "hidden_x2": block_factory(Conv_Block)(output_channels_list=[4,4,4,4,4,4], 
                                                           kernel_size_list=[5,5,5,5,5,5], 
                                                           pool_list=[16,8,4], 
                                                           max_pool=True),
                    "hidden_embedding": block_factory(FC_Block)(output_size_list=[16,8,4])
                },
                "skip_connections": ["hidden_xfc", "hidden_embedding"] }
            ):
        self.dset = dset
        self.num_fits = num_fits
        self.limits = limits
        self.device = device
        self.flatten_from = flatten_from
            
        self.encoder = encoder(function = function,
                                x_data = dset,
                                input_channels = dset.shape[1],
                                num_params = num_params,
                                device=device,**encoder_params)
        
        self._dataloader = None
        self._dataloader_sampler = None
        
        
    @property
    def dataloader_sampler(self):
        return self._dataloader_sampler
    def configure_dataloader_sampler(self, **kwargs):
        '''Set the sampler for the dataloader'''
        batch_size = kwargs.get('batch_size', 32)
        orig_shape = kwargs.get('orig_shape', self.dset.shape)
        gaussian_std=kwargs.get('gaussian_std', 5)
        num_neighbors=kwargs.get('num_neighbors', 10)
        sampler = kwargs.get('Sampler', None)
        
        # builds the dataloader
        if sampler is None: 
            self._dataloader_sampler = None
        else:
            self._dataloader_sampler = Gaussian_Sampler(self.dset, orig_shape, batch_size, gaussian_std, num_neighbors)
    
    @property
    def dataloader(self):
        return self._dataloader
    def configure_dataloader(self, **kwargs):
        '''Set the dataloader for the fitter

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            sampler (Sampler, optional): Defines the strategy to draw samples from the dataset. Defaults to None.
            collate_fn (callable, optional): Merges a list of samples to form a mini-batch. Defaults to None.
            shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to False.
        '''
        batch_size = kwargs.get('batch_size', 32)
        sampler = kwargs.get('sampler', None)
        collate_fn = kwargs.get('collate_fn', None)
        shuffle = kwargs.get('shuffle', False)
        
        # builds the dataloader
        if sampler is None: 
            self._dataloader = DataLoader(self.dset, batch_size=batch_size, shuffle=shuffle)
        else:
            self._dataloader = DataLoader(self.dset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, shuffle=shuffle)

    
    def train(self, seed=42):
        today = date.today()
        save_date=today.strftime('(%Y-%m-%d)')
        make_folder(self.folder)

        # set seed
        torch.manual_seed(seed)


        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_

        if self.wandb_project is not None:  
            wandb_init['project'] = self.wandb_project
            wandb.init(**wandb_init) # figure out config later
            
        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            fill_embeddings = False
            # if save_emb_every is not None and epoch % save_emb_every == 0: # tell loss function to give embedding
            #     print(f'Epoch: {epoch:03d}/{N_EPOCHS:03d}, getting embedding')
            #     print('.............................')
            #     fill_embeddings = self.get_embedding(data, train=True)


            loss_dict = self.loss_function( self.DataLoader_,
                                            coef1=coef_1,
                                            coef2=coef_2,
                                            coef3=coef_3,
                                            coef4=coef_4,
                                            coef5=coef_5,
                                            ln_parm=ln_parm,
                                            fill_embeddings=fill_embeddings,
                                            minibatch_logging_rate=minibatch_logging_rate,
                                            binning=binning,
                                            weight_by_distance=weight_by_distance, )
            # divide by batches inplace
            loss_dict.update( (k,v/len(self.DataLoader_)) for k,v in loss_dict.items())
            
            print(
                f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {loss_dict["train_loss"]:.4f}')
            print('.............................')

          #  schedular.step()
            lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
            self.checkpoint = self.folder + f'/{save_date}_' +\
                f'epoch:{epoch:04d}_l1coef:{coef_1:.4f}'+'_lr:'+lr_ +\
                f'_trainloss:{loss_dict["train_loss"]:.4f}.pkl'
            if epoch % save_model_every == 0:
                self.save_checkpoint(epoch,
                                    loss_dict=loss_dict,
                                    coef_1=coef_1, 
                                    coef_2=coef_2,
                                    coef_3=coef_3,
                                    coef_4=coef_4,
                                    ln_parm=ln_parm)

            if save_emb_every is not None and epoch % save_emb_every == 0: # tell loss function to give embedding
                h = self.embedding.file
                check = self.checkpoint.split('/')[-1][:-4]
                h[f'embedding_{check}'] = h[f'embedding_temp']
                h[f'scaleshear_{check}'] = h[f'scaleshear_temp']
                h[f'rotation_{check}'] = h[f'rotation_temp'] 
                h[f'translation_{check}'] = h[f'translation_temp']
                self.embedding = h[f'embedding_{check}']
                self.scale_shear = h[f'scaleshear_{check}']           
                self.rotation = h[f'rotation_{check}']         
                self.translation = h[f'translation_{check}']
                del h[f'embedding_temp']         
                del h[f'scaleshear_temp']          
                del h[f'rotation_temp']          
                del h[f'translation_temp']
                        
        if scheduler is not None:
            scheduler.step()

    def save_checkpoint(self,epoch,loss_dict,**kwargs):
        checkpoint = {
            "Fitter": self.Fitter.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch,
            'loss_dict': loss_dict,
            'loss_params': kwargs,
        }
        torch.save(checkpoint, self.checkpoint)

    # TODO: calculate norms on max intensity
    def loss_function(self,
                      train_iterator,
                      coef1=0,
                      coef2=0,
                      coef3=0,
                      coef4=0,
                      coef5=0,
                      ln_parm=1,
                      beta=None,
                      fill_embeddings=False,
                      minibatch_logging_rate=None,
                      binning=False,
                      weight_by_distance=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef1 (float, optional): Ln hyperparameter. Defaults to 0.
            coef2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef3 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """
        # set the train mode
        self.Fitter.train()

        # loss of the epoch
        loss_dict = {'weighted_ln_loss': 0,
                    #  'contras_loss': 0,
                    #  'maxi_loss': 0,
                     'mse_loss': 0,
                     'train_loss': 0,
                     'sparse_max_loss': 0,
                     'l2_batchwise_loss': 0,
                     }
        weighted_ln_ = Weighted_LN_loss(coef=coef1,channels=self.num_fits).to(self.device)
        con_l = ContrastiveLoss(coef2).to(self.device)
        maxi_ = DivergenceLoss(train_iterator.batch_size, coef3).to(self.device)
        sparse_max = Sparse_Max_Loss(min_threshold=self.learning_rate,
                                        channels=self.num_fits, 
                                        coef=coef4).to(self.device)
        
        for i,(idx,x) in enumerate( tqdm( train_iterator, leave=True, total=len(train_iterator) ) ):
            # tic = time.time()
            idx = idx.to(self.device).squeeze()
            x = x.to(self.device, dtype=torch.float)

            # update the gradients to zero
            self.optimizer.zero_grad()

            if beta is None: embedding, predicted_x = self.Fitter(x)
            else: embedding, sd, mn, predicted_x = self.Fitter(x)
            
            # TODO: weight by euclidean distance to 1st point in the bin
            # i, idx, x are lists. Each list is the gaussian samples from a batch
            if binning:
                x = list(torch.split(x, self.dataloader_sampler.num_neighbors)) # Split the batch into groups based on the number of neighbors
                predicted_x = list(torch.split(predicted_x, self.dataloader_sampler.num_neighbors))
                
                if weight_by_distance:
                    idx = torch.split(idx, self.dataloader_sampler.num_neighbors) # Split the indices into groups based on the number of neighbors
                    weight_list = []
                    for i_, sample_group in enumerate(idx):
                        p_ind, shp = self.dataloader_sampler._which_particle_shape(sample_group[0]) # Determine the particle index and shape for the current sample group
                        coords = [(int((ind - p_ind) % shp[1]), int((ind - p_ind) / shp[0])) for ind in sample_group] # Calculate the coordinates relative to the first point in the group
                        weights = torch.tensor([1]+[1 / (1 + ((coords[0][0]-coord[0])**2 + (coords[0][1]-coord[1])**2)**0.5) for coord in coords[1:]], # Calculate weights based on the Euclidean distance to the first point
                                                device = self.device) 
                        x[i_] = x[i_]*weights.unsqueeze(-1).unsqueeze(-1)
                        predicted_x[i_] = predicted_x[i_]*weights.unsqueeze(-1).unsqueeze(-1)  
                        weight_list.append(weights)
                    weight_sums = torch.stack([weight_.sum(dim=0) for weight_ in weight_list]).unsqueeze(-1).unsqueeze(-1) # Sum the weights in each group and stack them into a new batch
                    x = torch.stack([x_.sum(dim=0) for x_ in x])/weight_sums # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.sum(dim=0) for x_ in predicted_x])/weight_sums
                else:        
                    x = torch.stack([x_.mean(dim=0) for x_ in x]) # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.mean(dim=0) for x_ in predicted_x])
                    # print('')

            if coef1 > 0: 
                reg_loss_1 = weighted_ln_(embedding[:,:,0])
                loss_dict['weighted_ln_loss']+=reg_loss_1
            else: reg_loss_1 = 0

            if coef2 > 0: 
                contras_loss = con_l(embedding[:,:,0])
                loss_dict['contras_loss']+=contras_loss
            else: contras_loss = 0
                
            if coef3 > 0: 
                maxi_loss = maxi_(embedding[:,:,0])
                loss_dict['maxi_loss']+=maxi_loss
            else: maxi_loss = 0
            
            if coef4 > 0: # sparse_max_loss
                sparse_max_loss = sparse_max(embedding[:,:,0])
                loss_dict['sparse_max_loss']+=sparse_max_loss
            else: sparse_max_loss = 0
            
            if coef5 > 0: # set so the variation in x < fwhm, but the smaller the better.
                l2_loss = coef5*( (embedding[:,:,1]/embedding[:,:,2]).max(dim=0).values - \
                                  (embedding[:,:,1]/embedding[:,:,2]).min(dim=0).values ).mean()
                loss_dict['l2_batchwise_loss'] += l2_loss
                
            else: l2_loss = 0
            
            loss = F.mse_loss(x, predicted_x, reduction='mean');
            
            loss_dict['mse_loss'] += loss.item()
            
            loss = loss + reg_loss_1 + contras_loss - maxi_loss + l2_loss
            loss_dict['train_loss'] += loss.item()

            # backward pass
            loss.backward()

            # update the weights
            self.optimizer.step()
            
            # fill embedding if the correct epoch
            if fill_embeddings:
                sorted, indices = torch.sort(idx)
                sorted = sorted.detach().numpy()
                self.embedding[sorted] = embedding[indices].cpu().detach().numpy()
                # print('\tt', abs(tic-toc)) # 2.7684452533721924
            if minibatch_logging_rate is not None: 
                if i%minibatch_logging_rate==0: wandb.log({k: v/(i+1) for k,v in loss_dict.items()})

        return loss_dict
