from m3_learning.nn.STEM_AE import STEM_AE
import numpy as np
import torch
from torch.utils.data import Sampler
from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from m3_learning.util.file_IO import make_folder

from datetime import datetime

def affine_transform(x, scale, shear, rotation, translation, mask_parameter):
    """
    Function to apply affine transformation to the input tensor x

    Args:
    x (torch.Tensor): input tensor
    scale (torch.Tensor): scale transformation
    shear (torch.Tensor): shear transformation
    rotation (torch.Tensor): rotation transformation
    translation (torch.Tensor): translation transformation
    mask_parameter (torch.Tensor): mask parameter

    Returns:
    torch.Tensor: transformed tensor
    """
    # get the shape of the input tensor
    shape = x.shape
    device = x.device
    
    # get the grid
    grid = F.affine_grid(torch.eye(2, 3, device=device).unsqueeze(0).repeat(x.shape[0], 1, 1), x.size(), align_corners=False)

    if rotation is not None:
        grid = F.affine_grid(rotation, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
    # apply the transformations
    if scale is not None:
        grid = F.affine_grid(scale, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

    if shear is not None:
        grid = F.affine_grid(shear, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

    if translation is not None:
        grid = F.affine_grid(translation, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

    if mask_parameter is not None:
        x = x * mask_parameter

    return x

class Affine_Transform(nn.Module):
    def __init__(self,
                 device,
                 scale = True,
                 shear = True,
                 rotation = True,
                 translation = True,
                 Symmetric = True,
                 mask_intensity = True,
                 scale_limit = 0.05,
                 shear_limit = 0.1,
                 rotation_limit = 0.1,
                 trans_limit = 0.15,
                 adj_mask_para=0
                 ):
        '''
        '''
        super(Affine_Transform,self).__init__()
        self.scale = scale
        self.shear = shear
        self.rotation = rotation
        self.translation = translation
        self.Symmetric = Symmetric
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
        self.rotation_limit = rotation_limit
        self.trans_limit = trans_limit
        self.adj_mask_para = adj_mask_para
        self.mask_intensity = mask_intensity
        self.device = device
        self.count = 0

    def forward(self,out,rotate_value = None):

        if self.scale:
            scale_1 = self.scale_limit*nn.Tanh()(out[:,self.count])+1
            scale_2 = self.scale_limit*nn.Tanh()(out[:,self.count+1])+1
            self.count +=2
        else:
            scale_1 = None
            scale_2 = None
            
        if self.rotation:
            if rotate_value!=None:
                # use large mask no need to limit to too small range
                rotate = rotate_value.reshape(out[:,self.count].shape) + self.rotation_limit*nn.Tanh()(out[:,self.count])
                self.count+=1
            else:
                rotate = nn.ReLU()(out[:,self.count])
                self.count+=1
        else:
            rotate = None

        if self.shear:
            if self.Symmetric:
                shear_1 = self.shear_limit*nn.Tanh()(out[:,self.count])
                shear_2 = shear_1
                self.count+=1
            else:
                shear_1 = self.shear_limit*nn.Tanh()(out[:,self.count])
                shear_2 = self.shear_limit*nn.Tanh()(out[:,self.count+1])
                self.count+=2
        else:
            shear_1 = None
            shear_2 = None
        # usually the 4d-stem has symetric shear value, we make xy=yx, that's the reason we don't need shear2

        if self.translation:
            trans_1 = self.trans_limit*nn.Tanh()(out[:,self.count])
            trans_2 = self.trans_limit*nn.Tanh()(out[:,self.count+1])
            self.count +=2
        else:
            trans_1 = None
            trans_2 = None
  
        if self.mask_intensity:
            mask_parameter = self.adj_mask_para*nn.Tanh()(out[:,self.embedding_size:self.embedding_size+1])+1
        else:
            # this project doesn't need mask parameter to adjust value intensity in mask region, so we make it 1 here.
            mask_parameter = None
        self.count = 0

        a_4 = torch.ones([out.shape[0]]).to(self.device)
        a_5 = torch.zeros([out.shape[0]]).to(self.device)
        
        if self.rotate is not None:
            a_1 = torch.cos(rotate)
            a_2 = torch.sin(rotate)
            b1 = torch.stack((a_1,a_2), dim=1).squeeze()
            b2 = torch.stack((-a_2,a_1), dim=1).squeeze()
            b3 = torch.stack((a_5,a_5), dim=1).squeeze()
            rotation = torch.stack((b1, b2, b3), dim=2)
        else: rotation = None
            
        if self.scale:
            # separate scale and shear
            s1 = torch.stack((scale_1, a_5), dim=1).squeeze()
            s2 = torch.stack((a_5, scale_2), dim=1).squeeze()
            s3 = torch.stack((a_5, a_5), dim=1).squeeze()
            scale = torch.stack((s1, s2, s3), dim=2)
        else: scale = None
        
        if self.shear:
            sh1 = torch.stack((a_4, shear_1), dim=1).squeeze()
            sh2 = torch.stack((shear_2, a_4), dim=1).squeeze()
            sh3 = torch.stack((a_5, a_5), dim=1).squeeze()
            shear = torch.stack((sh1, sh2, sh3), dim=2)
        else: shear = None

        # Add the rotation after the shear and strain
        if self.translation:
            d1 = torch.stack((a_4,a_5), dim=1).squeeze()
            d2 = torch.stack((a_5,a_4), dim=1).squeeze()
            d3 = torch.stack((trans_1,trans_2), dim=1).squeeze()
            translation = torch.stack((d1, d2, d3), dim=2)
        else: translation = None

        return scale, shear, rotation, translation, mask_parameter


class Affine_AE_2D_module(nn.Module):
    def __init__(self,
                 device,
                 affine_encoder,
                 affine_module,
                 encoder,
                 decoder,
                 ): # TODO: do we eventually wnat to mask?

        super(Affine_AE_2D_module, self).__init__()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        self.device = device
        self.affine_encoder = affine_encoder
        self.affine_module = affine_module
        self.encoder = encoder
        self.decoder = decoder        

    def forward(self, x):
        emb_affine = self.affine_encoder(x)
        scale, shear, rotation, translation, mask_parameter = self.affine_model(emb_affine)
        x = affine_transform(x, scale, shear, rotation, translation, mask_parameter)
        
        emb = self.encoder(x)
        x = self.decoder(x)
        
        # TODO: make everything into **kwargs later
        return x, emb, scale, shear, rotation, translation, mask_parameter


class Affine_AE_2D(STEM_AE.ConvAutoencoder):
    def __init__(self, 
                 device,
                 sampler, 
                 sampler_kwargs, 
                 collate_fn, 
                 affine_encoder = STEM_AE.Encoder,
                 affine_encoder_kwargs = { 'original_step_size': [100,100], 
                                           'pooling_list': [], 
                                           'embedding_size': 6, 
                                           'conv_size': 128,
                                           'kernel_size': 3,},
                 affine_module = Affine_Transform,
                 affine_kwargs = {  "scale": True,
                                    "shear": True,
                                    "rotation": True,
                                    "translation": True,
                                    "Symmetric": True,
                                    "mask_intensity": True,
                                    "scale_limit": 0.05,
                                    "shear_limit": 0.1,
                                    "rotation_limit": 0.1,
                                    "trans_limit": 0.15,
                                    "adj_mask_para": 0  },
                 encoder = STEM_AE.Encoder,
                 encoder_kwargs = { 'original_step_size': [100,100], 
                                    'pooling_list': [], 
                                    'embedding_size': 32, 
                                    'conv_size': 128,
                                    'kernel_size': 3},
                 decoder = STEM_AE.Decoder,
                 decoder_kwargs = { 'original_step_size': [5,5], 
                                    'upsampling_list': [], 
                                    'embedding_size': 32, 
                                    'conv_size': 128,
                                    'kernel_size': 3},
                 autoencoder = Affine_AE_2D_module,
                 *args, **kwargs):
        super(Affine_AE_2D, self).__init__(device=device,*args, **kwargs)
        
        self.sampler = sampler(**sampler_kwargs)
        self.collate_fn = collate_fn
        
        self.device = device
        
        for key, value in affine_encoder_kwargs.items():
            setattr(self, 'affine_encoder_'+key, value)
        self.affine_encoder = affine_encoder(**affine_encoder_kwargs)
        
        affine_kwargs['device'] = device
        for key, value in affine_kwargs.items():
            setattr(self, 'affine_block_'+key, value)
        self.affine_module = affine_module(**affine_kwargs)
        
        for key, value in encoder_kwargs.items():
            setattr(self, 'encoder_'+key, value)
        self.encoder = encoder(**encoder_kwargs)
        
        for key, value in decoder_kwargs.items():
            setattr(self, 'decoder_'+key, value)
        self.decoder = decoder(**decoder_kwargs)
        
        self.autoencoder = autoencoder(device = device,
                                       affine_encoder = self.affine_encoder,
                                       affine_module = self.affine_module,
                                       encoder = self.encoder,
                                       decoder = self.decoder)
        
    def Train(self,
              data,
              max_learning_rate=1e-4,
              coef_1=0,
              coef_2=0,
              coef_3=0,
              seed=12,
              epochs=100,
              with_scheduler=True,
              ln_parm=1,
              epoch_=None,
              folder_path='./',
              batch_size=32,
              best_train_loss=None,
              dataloader_init = None):
        """function that trains the model

        Args:
            data (torch.tensor): data to train the model
            max_learning_rate (float, optional): sets the max learning rate for the learning rate cycler. Defaults to 1e-4.
            coef_1 (float, optional): hyperparameter for ln loss. Defaults to 0.
            coef_2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef_3 (float, optional): hyperparameter for divergency loss. Defaults to 0.
            seed (int, optional): sets the random seed. Defaults to 12.
            epochs (int, optional): number of epochs to train. Defaults to 100.
            with_scheduler (bool, optional): sets if you should use the learning rate cycler. Defaults to True.
            ln_parm (int, optional): order of the Ln regularization. Defaults to 1.
            epoch_ (int, optional): current epoch for continuing training. Defaults to None.
            folder_path (str, optional): path where to save the weights. Defaults to './'.
            batch_size (int, optional): sets the batch size for training. Defaults to 32.
            best_train_loss (float, optional): current loss value to determine if you should save the value. Defaults to None.
        """

        make_folder(folder_path)

        # set seed
        torch.manual_seed(seed)
        
        # builds the dataloader
        if dataloader_init is None:  
            self.DataLoader_ = DataLoader(data.reshape(-1, data.shape[-2], data.shape[-1]),
                                          batch_size=batch_size, 
                                          shuffle=True)
        else: self.DataLoader_ = DataLoader(data, **dataloader_init)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, step_size_up=15, cycle_momentum=False)
        else: scheduler = None

        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_+1

        # training loop
        for epoch in range(self.start_epoch, epochs):
            train = self.loss_function(self.DataLoader_, 
                                       coef_1, 
                                       coef_2, 
                                       coef_3, 
                                       ln_parm)
            train_loss = train
            train_loss /= len(self.DataLoader_)
            print( f'Epoch: {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f}')
            print('.............................')
            if with_scheduler: scheduler.step()            
            self.save_model(epoch, folder_path, train_loss, )
    
    def save_model(self, epoch, folder_path, train_loss,
                   **kwargs):
        datetime_ = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint = {"net": self.autoencoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch,
                    'affine_encoder': self.affine_encoder.state_dict(),
                    'affine_module' : self.affine_module.state_dict(),
                    "encoder": self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'loss': train_loss,
                    'training_params': kwargs,
                    }
        file_path = folder_path + f'/({datetime_})_epoch:{epoch:04d}_'+\
                    f'trainloss:{train_loss:.4f}.pkl'
        
        torch.save(checkpoint, file_path)
                    
    def loss_function(self,
                      train_iterator,
                      coef=0,
                      coef1=0,
                      coef2=0,
                      ln_parm=1,
                      binning=False,
                      weight_by_distance=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef (float, optional): Ln hyperparameter. Defaults to 0.
            coef1 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef2 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """

        # set the train mode
        self.autoencoder.train()

        # loss of the epoch
        train_loss = 0
        con_l = ContrastiveLoss(coef1).to(self.device)

        for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):

            x = x.to(self.device, dtype=torch.float)
            maxi_ = DivergenceLoss(x.shape[0], coef2).to(self.device)
            
            
            # update the gradients to zero
            self.optimizer.zero_grad()

            embedding = self.encoder(x)
            reg_loss_1 = coef * torch.norm(embedding, ln_parm).to(self.device)/x.shape[0]
            if reg_loss_1 == 0: reg_loss_1 = 0.5
            predicted_x = self.decoder(embedding)
            contras_loss = con_l(embedding)
            maxi_loss = maxi_(embedding)
            
            # i, idx, x are lists. Each list is the gaussian samples from a batch
            if binning:
                # x = list(torch.split(x, self.dataloader_sampler.num_neighbors)) # Split the batch into groups based on the number of neighbors
                # predicted_x = list(torch.split(predicted_x, self.dataloader_sampler.num_neighbors))
                
                x = self.sampler._split_list(x) # Split the batch into groups based on the number of neighbors
                predicted_x = self.sampler._split_list(predicted_x)
                
                if weight_by_distance:
                    idx = self.sampler._split_list(idx, self.dataloader_sampler.num_neighbors) # Split the indices into groups based on the number of neighbors
                    weight_list = []
                    for i_, sample_group in enumerate(idx):
                        coords = [(ind % self.shape[0], int(ind /self.shape[0])) for ind in sample_group] # Calculate the coordinates relative to the first point in the group
                        weights = torch.tensor([1] +[1 /(1 +((coords[0][0]-coord[0])**2 +(coords[0][1]-coord[1])**2)**0.5) for coord in coords[1:]], # Calculate weights based on the Euclidean distance to the first point
                                                device = self.device) 
                        x[i_] = x[i_] *weights.unsqueeze(-1).unsqueeze(-1)
                        predicted_x[i_] = predicted_x[i_]*weights.unsqueeze(-1).unsqueeze(-1)  
                        weight_list.append(weights)
                    weight_sums = torch.stack([weight_.sum(dim=0) for weight_ in weight_list]).unsqueeze(-1).unsqueeze(-1) # Sum the weights in each group and stack them into a new batch
                    x = torch.stack([x_.sum(dim=0) for x_ in x])/weight_sums # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.sum(dim=0) for x_ in predicted_x])/weight_sums
                else:        
                    x = torch.stack([x_.mean(dim=0) for x_ in x]) # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.mean(dim=0) for x_ in predicted_x])
                    # print('')

            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction='mean')
            
            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            # backward pass
            train_loss += loss.item()
            loss.backward()
            # update the weights
            self.optimizer.step()

        return train_loss
                                            


class Averaging_Loss_AE(STEM_AE.ConvAutoencoder):
    def __init__(self, sampler, sampler_kwargs, collate_fn, *args, **kwargs):
        super(Averaging_Loss_AE, self).__init__(*args, **kwargs)
        self.sampler = sampler(**sampler_kwargs)
       
    def Train(self,
              data,
              max_learning_rate=1e-4,
              coef_1=0,
              coef_2=0,
              coef_3=0,
              seed=12,
              epochs=100,
              with_scheduler=True,
              ln_parm=1,
              epoch_=None,
              folder_path='./',
              batch_size=32,
              best_train_loss=None,
              dataloader_init = None):
        """function that trains the model

        Args:
            data (torch.tensor): data to train the model
            max_learning_rate (float, optional): sets the max learning rate for the learning rate cycler. Defaults to 1e-4.
            coef_1 (float, optional): hyperparameter for ln loss. Defaults to 0.
            coef_2 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef_3 (float, optional): hyperparameter for divergency loss. Defaults to 0.
            seed (int, optional): sets the random seed. Defaults to 12.
            epochs (int, optional): number of epochs to train. Defaults to 100.
            with_scheduler (bool, optional): sets if you should use the learning rate cycler. Defaults to True.
            ln_parm (int, optional): order of the Ln regularization. Defaults to 1.
            epoch_ (int, optional): current epoch for continuing training. Defaults to None.
            folder_path (str, optional): path where to save the weights. Defaults to './'.
            batch_size (int, optional): sets the batch size for training. Defaults to 32.
            best_train_loss (float, optional): current loss value to determine if you should save the value. Defaults to None.
        """

        make_folder(folder_path)

        # set seed
        torch.manual_seed(seed)
        
        # builds the dataloader
        if dataloader_init is None:
            self.DataLoader_ = DataLoader(
                data.reshape(-1, 256, 256), batch_size=batch_size, shuffle=True)
        else:
            self.DataLoader_ = DataLoader(data, **dataloader_init)

        # option to use the learning rate scheduler
        if with_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=self.learning_rate, max_lr=max_learning_rate, step_size_up=15, cycle_momentum=False)
        else:
            scheduler = None

        # set the number of epochs
        N_EPOCHS = epochs

        # initializes the best train loss
        if best_train_loss == None:
            best_train_loss = float('inf')

        # initialize the epoch counter
        if epoch_ is None:
            self.start_epoch = 0
        else:
            self.start_epoch = epoch_+1

        # training loop
        for epoch in range(self.start_epoch, N_EPOCHS):
            train = self.loss_function(
                self.DataLoader_, coef_1, coef_2, coef_3, ln_parm)
            train_loss = train
            train_loss /= len(self.DataLoader_)
            print(
                f'Epoch: {epoch:03d}/{N_EPOCHS:03d} | Train Loss: {train_loss:.4f}')
            print('.............................')

          #  schedular.step()
            if best_train_loss > train_loss:
                best_train_loss = train_loss
                checkpoint = {
                    "net": self.autoencoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch,
                    "encoder": self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                }
                if epoch >= 0:
                    lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
                    file_path = folder_path + '/Weight_' +\
                        f'epoch:{epoch:04d}_l1coef:{coef_1:.4f}'+'_lr:'+lr_ +\
                        f'_trainloss:{train_loss:.4f}.pkl'
                    torch.save(checkpoint, file_path)

            if scheduler is not None:
                scheduler.step()
        
    def loss_function(self,
                      train_iterator,
                      coef=0,
                      coef1=0,
                      coef2=0,
                      ln_parm=1,
                      binning=False,
                      weight_by_distance=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef (float, optional): Ln hyperparameter. Defaults to 0.
            coef1 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef2 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """

        # set the train mode
        self.autoencoder.train()

        # loss of the epoch
        train_loss = 0
        con_l = ContrastiveLoss(coef1).to(self.device)

        for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):

            x = x.to(self.device, dtype=torch.float)
            maxi_ = DivergenceLoss(x.shape[0], coef2).to(self.device)
            
            
            # update the gradients to zero
            self.optimizer.zero_grad()

            embedding = self.encoder(x)
            reg_loss_1 = coef * torch.norm(embedding, ln_parm).to(self.device)/x.shape[0]
            if reg_loss_1 == 0: reg_loss_1 = 0.5
            predicted_x = self.decoder(embedding)
            contras_loss = con_l(embedding)
            maxi_loss = maxi_(embedding)
            
            # i, idx, x are lists. Each list is the gaussian samples from a batch
            if binning:
                # x = list(torch.split(x, self.dataloader_sampler.num_neighbors)) # Split the batch into groups based on the number of neighbors
                # predicted_x = list(torch.split(predicted_x, self.dataloader_sampler.num_neighbors))
                
                x = self.sampler._split_list(x) # Split the batch into groups based on the number of neighbors
                predicted_x = self.sampler._split_list(predicted_x)
                
                if weight_by_distance:
                    idx = self.sampler._split_list(idx, self.dataloader_sampler.num_neighbors) # Split the indices into groups based on the number of neighbors
                    weight_list = []
                    for i_, sample_group in enumerate(idx):
                        coords = [(ind % self.shape[0], int(ind /self.shape[0])) for ind in sample_group] # Calculate the coordinates relative to the first point in the group
                        weights = torch.tensor([1] +[1 /(1 +((coords[0][0]-coord[0])**2 +(coords[0][1]-coord[1])**2)**0.5) for coord in coords[1:]], # Calculate weights based on the Euclidean distance to the first point
                                                device = self.device) 
                        x[i_] = x[i_] *weights.unsqueeze(-1).unsqueeze(-1)
                        predicted_x[i_] = predicted_x[i_]*weights.unsqueeze(-1).unsqueeze(-1)  
                        weight_list.append(weights)
                    weight_sums = torch.stack([weight_.sum(dim=0) for weight_ in weight_list]).unsqueeze(-1).unsqueeze(-1) # Sum the weights in each group and stack them into a new batch
                    x = torch.stack([x_.sum(dim=0) for x_ in x])/weight_sums # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.sum(dim=0) for x_ in predicted_x])/weight_sums
                else:        
                    x = torch.stack([x_.mean(dim=0) for x_ in x]) # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.mean(dim=0) for x_ in predicted_x])
                    # print('')

            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction='mean')
            
            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            # backward pass
            train_loss += loss.item()
            loss.backward()
            # update the weights
            self.optimizer.step()

        return train_loss