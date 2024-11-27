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
import torch.optim as optim
import h5py
from torch.autograd import Variable
from datetime import datetime
import os
from m3_learning.viz.layout import find_nearest


def apply_affine_transform(x, scale, shear, rotation, translation, mask_parameter):
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
    
    # get the grid
    # grid = F.affine_grid(torch.eye(2, 3, device=device).unsqueeze(0).repeat(x.shape[0], 1, 1), x.size(), align_corners=False)

    if translation is not None:
        grid = F.affine_grid(translation, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
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


    if mask_parameter is not None:
        x = x * mask_parameter

    return x


def apply_inv_affine_transform(x, scale, shear, rotation, translation, mask_parameter):
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
    
    # compute inverse affine matrix 
    identity = ( torch.tensor([0, 0, 1], dtype=torch.float).reshape(1, 1, 3)\
                .repeat(shape[0], 1, 1).to(device) )
    
    # get the grid
    # grid = F.affine_grid(torch.eye(2, 3, device=device).unsqueeze(0).repeat(x.shape[0], 1, 1), x.size(), align_corners=False)

    # apply the transformations
    if shear is not None:
        inver_shear = torch.linalg.inv( 
                        torch.cat((shear, identity), axis=1).to(device)
                    )[:, 0:2].to(device)
        grid = F.affine_grid(inver_shear, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
    if scale is not None:
        inver_scale = torch.linalg.inv( 
                         torch.cat((scale, identity), axis=1).to(device)
                                        )[:, 0:2].to(device)
        grid = F.affine_grid(inver_scale, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

    if rotation is not None:
        inver_rotation = torch.linalg.inv( 
                            torch.cat((translation, identity), axis=1).to(device)
                      )[:, 0:2].to(device)
        grid = F.affine_grid(inver_rotation, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
    if translation is not None:
        inver_translation = torch.linalg.inv( 
                            torch.cat((translation, identity), axis=1).to(device)
                        )[:, 0:2].to(device)
        grid = F.affine_grid(inver_translation, x.size(), align_corners=False)
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
        
        if self.rotation is not None:
            a_1 = torch.cos(rotate)
            a_2 = torch.sin(rotate)
            b1 = torch.stack((a_1,a_2), dim=1)#.squeeze()
            b2 = torch.stack((-a_2,a_1), dim=1)#.squeeze()
            b3 = torch.stack((a_5,a_5), dim=1)#.squeeze()
            rotation = torch.stack((b1, b2, b3), dim=2)
        else: rotation = None
            
        if self.scale:
            # separate scale and shear
            s1 = torch.stack((scale_1, a_5), dim=1)#.squeeze()
            s2 = torch.stack((a_5, scale_2), dim=1)#.squeeze()
            s3 = torch.stack((a_5, a_5), dim=1)#.squeeze()
            scale = torch.stack((s1, s2, s3), dim=2)
        else: scale = None
        
        if self.shear:
            sh1 = torch.stack((a_4, shear_1), dim=1)#.squeeze()
            sh2 = torch.stack((shear_2, a_4), dim=1)#.squeeze()
            sh3 = torch.stack((a_5, a_5), dim=1)#.squeeze()
            shear = torch.stack((sh1, sh2, sh3), dim=2)
        else: shear = None

        # Add the rotation after the shear and strain
        if self.translation:
            d1 = torch.stack((a_4,a_5), dim=1)#.squeeze()
            d2 = torch.stack((a_5,a_4), dim=1)#.squeeze()
            d3 = torch.stack((trans_1,trans_2), dim=1)#.squeeze()
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
        
        self.device = device
        self.affine_encoder = affine_encoder
        self.affine_module = affine_module
        self.encoder = encoder
        self.decoder = decoder        

    def _encoder(self, x):
        if x.dim() < 4: x = x.unsqueeze(1)
        emb_affine = self.affine_encoder(x)
        scale, shear, rotation, translation, mask_parameter = self.affine_module(emb_affine)
        x = apply_affine_transform(x, scale, shear, rotation, translation, mask_parameter)
        return self.encoder(x), scale, shear, rotation, translation, mask_parameter
        
    def _decoder(self, emb, scale, shear, rotation, translation, mask_parameter):
        x = self.decoder(emb)
        if x.dim() < 4: x = x.unsqueeze(1)
        return apply_inv_affine_transform(x, scale, shear, rotation, translation, mask_parameter)
        
    def forward(self, x):
        emb, scale, shear, rotation, translation, mask_parameter = self._encoder(x)
        x = self._decoder(emb, scale, shear, rotation, translation, mask_parameter)
        
        # TODO: make everything into **kwargs later
        return x, emb, translation, rotation, scale, shear, mask_parameter


class Affine_AE_2D():
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
                 
                 learning_rate = 1e-4,
                 *args, **kwargs):
        super(Affine_AE_2D, self).__init__()
        
        self.device = device
        self.learning_rate = learning_rate
        
        if sampler==None: self.sampler = None
        else: self.sampler = sampler(**sampler_kwargs)
        self.collate_fn = collate_fn
        
        for key, value in affine_encoder_kwargs.items():
            setattr(self, 'affine_encoder_'+key, value)
        self.affine_encoder = affine_encoder(**affine_encoder_kwargs).to(self.device)
        
        affine_kwargs['device'] = device
        for key, value in affine_kwargs.items():
            setattr(self, 'affine_block_'+key, value)
        self.affine_module = affine_module(**affine_kwargs).to(self.device)
        
        for key, value in encoder_kwargs.items():
            setattr(self, 'encoder_'+key, value)
        self.encoder = encoder(**encoder_kwargs).to(self.device)
        
        for key, value in decoder_kwargs.items():
            setattr(self, 'decoder_'+key, value)
        self.decoder = decoder(**decoder_kwargs).to(self.device)
        
        self.autoencoder = autoencoder(device = device,
                                       affine_encoder = self.affine_encoder,
                                       affine_module = self.affine_module,
                                       encoder = self.encoder,
                                       decoder = self.decoder).to(self.device)
        
        # sets the optimizers
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        
    @property
    def checkpoint(self):
        return self.checkpoint_
    @checkpoint.setter
    def checkpoint(self, value):
        self.checkpoint_ = value
        self.folder_path, filename = os.path.split(self.checkpoint)
        self.emb_h5_path = self.folder_path+'/_embedding.h5'
        self.gen_h5_path = self.folder_path+'/_generated.h5'
        self.check = filename[:-4]
        
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
              batch_size=None,
              best_train_loss=None,
              binning=False,
              **kwargs):
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
        self.DataLoader_ = DataLoader(data.reshape(-1, data.shape[-2], data.shape[-1]),
                                      sampler=self.sampler,
                                      collate_fn=self.collate_fn,
                                      batch_size=batch_size,)

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
                                       ln_parm,
                                       binning=binning)
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
                    f'trainloss:{train_loss:.4e}.pkl'
        
        torch.save(checkpoint, file_path)
        self.checkpoint = file_path
  
    def load_weights(self, path_checkpoint,return_checkpoint=False):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
            return_checkpoint (bool, Optional): whether to return the checkpoint loaded. Default False
        
        Returns:
            checkpoint (Optional)
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint)
        self.autoencoder.load_state_dict(checkpoint['net'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        
        if return_checkpoint: return checkpoint
                  
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
            
            sh = x.shape
            # update the gradients to zero
            self.optimizer.zero_grad()

            predicted_x, embedding, translation, rotation, scale, shear, mask_parameter = self.autoencoder(x)
            if coef>0: reg_loss_1 = coef * torch.norm(embedding, ln_parm).to(self.device)/x.shape[0]
            else: reg_loss_1 = 0
            if coef1>0: contras_loss = con_l(embedding)
            else: contras_loss = 0
            if coef2>0: maxi_loss = maxi_(embedding)
            else: maxi_loss = 0
            
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
            loss = F.mse_loss(x.reshape(-1,sh[-2],sh[-1]), 
                              predicted_x.reshape(-1,sh[-2],sh[-1]), 
                              reduction='mean')
            
            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            # backward pass
            train_loss += loss.item()
            loss.backward()
            # update the weights
            self.optimizer.step()

        return train_loss
                                            
    def get_embeddings(self, data, batch_size=32, train=False, check=None):
        # builds the dataloader
        dataloader = DataLoader(data, batch_size, shuffle=False)

        try:
            try: h = h5py.File(self.emb_h5_path,'r+')
            except: 
                h = h5py.File(self.emb_h5_path,'w')
                print(f'creating {self.emb_h5_path} file')

            if check==None: check = self.checkpoint.split('/')[-1][:-4]
            try: 
                embedding_ = h[f'embedding_{check}']
                scale_ = h[f'scale_{check}']
                shear_ = h[f'shear_{check}']
                rotation_ = h[f'rotation_{check}']                    
                translation_ = h[f'translation_{check}']
            except:
                embedding_ = h.create_dataset(f'embedding_{check}', data = np.zeros([data.shape[0], self.encoder_embedding_size]))
                scale_shear_ = h.create_dataset(f'scale_{check}', data = np.zeros([data.shape[0],6]))
                scale_shear_ = h.create_dataset(f'shear_{check}', data = np.zeros([data.shape[0],6]))
                rotation_ = h.create_dataset(f'rotation_{check}', data = np.zeros([data.shape[0],6]))
                translation_ = h.create_dataset(f'translation_{check}', data = np.zeros([data.shape[0],6]))
                print('creating new embedding and affine h5 datasets')

        except Exception as error:
            print(error) 
            assert self.train,"No h5_dataset embedding dataset created"
            print('Warning: not saving to h5')
             
        h.close()
                
        if train: 
            print('Created empty h5 embedding datasets to fill during training')
            return 1 # do not calculate. 
            # return true to indicate this is filled during training

        else:
            for i, x in enumerate(tqdm(dataloader, leave=True, total=len(dataloader))):
                with torch.no_grad():
                    with h5py.File(self.emb_h5_path,'r+') as h:       
                        value = x
                        test_value = Variable(value.to(self.device))
                        test_value = test_value.float()
                        ( predicted_x, 
                          embedding, 
                          translation, 
                          rotation, 
                          scale, 
                          shear, 
                          mask_parameter ) = self.autoencoder(test_value)                      
                        h[f'embedding_{check}'][i*batch_size:(i+1)*batch_size, :] = embedding.cpu().detach().numpy()
                        h[f'scale_{check}'][i*batch_size:(i+1)*batch_size, :] = scale.reshape(-1,6).cpu().detach().numpy()
                        h[f'shear_{check}'][i*batch_size:(i+1)*batch_size, :] = shear.reshape(-1,6).cpu().detach().numpy()
                        h[f'rotation_{check}'][i*batch_size:(i+1)*batch_size, :] = rotation.reshape(-1,6).cpu().detach().numpy()
                        h[f'translation_{check}'][i*batch_size:(i+1)*batch_size, :] = translation.reshape(-1,6).cpu().detach().numpy()

    def generate_by_range(self,
                          orig_shape,
                         generator_iters=50,
                         averaging_number=50,
                         overwrite=False,
                         ranges=None,
                         channels=None
                         ):
        """Generates images as the variables traverse the latent space.
        Saves to embedding h5 dataset

        Args:
            embedding (tensor, optional): embedding to predict with. Defaults to None.
            folder_name (str, optional): name of folder where images are saved. Defaults to ''.
            ranges (list, optional): sets the range to generate images over. Defaults to None.
            generator_iters (int, optional): number of iterations to use in generation. Defaults to 200.
            averaging_number (int, optional): number of embeddings to average. Defaults to 100.
            graph_layout (list, optional): layout parameters of the graph (#graphs,#perrow). Defaults to [2, 2].
            shape_ (list, optional): initial shape of the image. Defaults to [256, 256, 256, 256].
        """
        if channels is None: channels = range(self.embedding_size)

        # # gets the embedding
        # try:
        #     with h5py.File(self.emb_h5_path,'r+') as he:
        #         data = he[f'embedding_{self.check}']
        #         scale = he[f'scale_{self.check}']
        #         shear = he[f'shear_{self.check}']
        #         rotation = he[f'rotation_{self.check}']
        #         translation = he[f'translation_{self.check}']
        # except Exception as error:
        #     print(error)
        #     assert False,"No h5_dataset embedding dataset created"

        try: # try opening h5 file
            try: # make new file
                hg = h5py.File(self.gen_h5_path,'w')
                print(f'creating {self.emb_h5_path} file')
            except: # open existing file
                hg = h5py.File(self.gen_h5_path,'r+')

            try: # make new dataset
                if overwrite and self.check in hg: del hg[self.check]
                generated = hg.create_dataset(self.check,
                                              shape=(generator_iters,
                                                    len(channels),
                                                    orig_shape[2], orig_shape[3]) )
            except: # open existing dataset for checkpoint
                self.generated = hg[self.check]
                
        except Exception as error: # cannot open h5
            print(error)
            assert False,"No h5_dataset generated dataset created"

        with h5py.File(self.gen_h5_path,'r+') as hg:
            with h5py.File(self.emb_h5_path,'r+') as he:
                data = he[f'embedding_{self.check}']
                try: scale = he[f'scale_{self.check}']
                except: scale = None
                try: shear = he[f'shear_{self.check}']
                except: shear = None
                try: rotation = he[f'rotation_{self.check}']
                except: rotation = None
                try: translation = he[f'translation_{self.check}']
                except: translation = None
                try: mask_parameter = he[f'mask_parameter_{self.check}']
                except: mask_parameter = None
                
                generated = hg[self.check]
                # loops around the number of iterations to generate
                for i in tqdm(range(generator_iters)):
                    # loops around all of the embeddings
                    for j, channel in enumerate(channels):

                        if ranges is None: 
                            ranges = np.stack((np.min(data, axis=0),
                                            np.max(data, axis=0)), axis=1)

                        # linear space values for the embeddings
                        value = np.linspace(ranges[j][0], ranges[j][1],
                                            generator_iters)
                        dec_kwargs = self.decoder_kwargs(
                                            channel=channel,
                                            ref_value=value[i], 
                                            averaging_number=averaging_number,
                                            data=data[:], 
                                            scale=scale, 
                                            shear=shear, 
                                            rotation=rotation, 
                                            translation=translation,
                                            mask_parameter=mask_parameter)
                        # generates diffraction pattern
                        generated[i,j] =\
                            self.generate_spectra(**dec_kwargs).squeeze().cpu().detach().numpy()       
    
    def decoder_kwargs(self,channel,ref_value,data,averaging_number,**kwargs):
        idx = find_nearest(data[channel], ref_value, averaging_number)
        idx.sort()
        # finds the idx of nearest `averaging_number` of points to the value
        # TODO: try this with all embeddings at 0, except the current is the value[i]
        # computes the mean of the selected indices to yield (embsize) length vector
        gen_value = data[idx].mean(axis=0)
        gen_value[channel] = ref_value
        
        for key, value in kwargs.items():
            try: kwargs[key] = value[idx].mean(axis=0)
            except: pass
        kwargs['x'] = gen_value
        
        return kwargs
        
    def generate_spectra(self, x, **kwargs):
        x = torch.from_numpy(np.atleast_2d(x)).to(self.device)
        for key, value in kwargs.items():
            try: kwargs[key] = torch.from_numpy(value.reshape(1,2,3)).float().to(self.device)
            except: pass
        return self.autoencoder._decoder(x.float(), **kwargs)


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
            loss = F.mse_loss(x.squeeze(), predicted_x.squeeze(), reduction='mean')
            
            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            # backward pass
            train_loss += loss.item()
            loss.backward()
            # update the weights
            self.optimizer.step()

        return train_loss