import sys
import os

# remove eventually
sys.path.append('/home/m3learning/Northwestern/M3Learning-Util/src')
sys.path.append('/home/m3learning/Northwestern/AutoPhysLearn/src')

from random import shuffle
from m3util.util.IO import make_folder
from m3util.ml.regularization import Weighted_LN_loss, ContrastiveLoss, DivergenceLoss, Sparse_Max_Loss

from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Conv_Block, FC_Block, block_factory

from ..data.custom_sampler import Gaussian_Sampler

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datetime import date
from tqdm import tqdm

#TODO: make classes out of functions. 
class pseudovoigt_1D_fitters():
    def __init__(self, limits=[1,1,975]):
        self.limits = limits
    
    def scale_parameters(self, embedding):
        A = self.limits[0] * embedding[..., 0] # area under curve TODO: best way to scale this?
        # Ib = limits[1] * nn.ReLU()(embedding[..., 1])
        x = self.limits[1] * embedding[..., 1] # mean
        w = self.limits[2] * embedding[..., 2] # fwhm
        nu = embedding[..., 3] # fraction voight character
        return torch.stack([A,x,w,nu],axis=2)

    def apply_activations(self, embedding):
        '''This function takes an embedding and scales it to the limits of the parameters
        
        This function implements the Pseudo-Voigt profile as described in:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/
        
        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - A: Area under curve (index 0)
                - x: Mean position (index 1)
                - w: Full Width at Half Maximum (FWHM) (index 2)
                - nu: Lorentzian character fraction (index 3)
            limits (list): Scale factors for [A, x, w]. Defaults to [1, 1, 975]
        '''
        A = nn.ReLU()(embedding[..., 0]) # area under curve 
        # Ib = limits[1] * nn.ReLU()(embedding[..., 1])
        x = torch.clamp(nn.Tanh()(embedding[..., 1])/2 + 0.5, min=1e-3) # mean
        w = torch.clamp(nn.Tanh()(embedding[..., 2])/2 + 0.5, min=1e-3) # fwhm
        nu = 0.5 * nn.Tanh()(embedding[..., 3]) + 0.5 # fraction voight character
        return torch.stack([A,x,w,nu],axis=2)

    def _gaussian_component(self, A, x, x_, w):
        """Calculate the Gaussian component of the Pseudo-Voigt profile
        
        Args:
            A (torch.Tensor): Area under curve
            x (torch.Tensor): Mean positions
            x_ (torch.Tensor): X-axis points
            w (torch.Tensor): Full Width at Half Maximum (FWHM)
        """
        gaussian_factor = (4 * torch.log(torch.tensor(2)) / torch.pi) ** 0.5
        gaussian = (A * gaussian_factor / w * 
                torch.exp(-4 * torch.log(torch.tensor(2)) / w**2 * 
                            (x_ - x)**2))
        return gaussian

    def _lorentzian_component(self, A, x, x_, w):
        """Calculate the Lorentzian component of the Pseudo-Voigt profile
        
        Args:
            A (torch.Tensor): Area under curve
            x (torch.Tensor): Mean positions
            x_ (torch.Tensor): X-axis points
            w (torch.Tensor): Full Width at Half Maximum (FWHM)
        """
        lorentzian = (A * (2/torch.pi * w) / (4 * (x_ - x)**2 + w**2))
        return lorentzian

    def generate_fit(self, embedding, dset, spec_len=None):
        """Generate 1D Pseudo-Voigt profiles from embedding parameters.

        This function implements the Pseudo-Voigt profile as described in:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/

        The Pseudo-Voigt profile is a linear combination of Gaussian and Lorentzian profiles,
        controlled by the mixing parameter nu.

        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - A: Area under curve (index 0)
                - x: Mean position (index 1)
                - w: Full Width at Half Maximum (FWHM) (index 2)
                - nu: Lorentzian character fraction (index 3)
            dset: Dataset containing spectral information with attribute spec_len
            return_params (bool): If True, returns both profile and parameters. Defaults to False

        Returns:
            torch.Tensor: Pseudo-Voigt profiles of shape (batch_size, num_fits, spec_len)
            torch.Tensor: (Optional) Parameters [A, x, w, nu] if return_params=True
        """
        device = embedding.device
        # Unpack embedding tensor along last dimension (shape: [..., 4])
        A = embedding[..., 0].unsqueeze(-1)  # Area
        x = embedding[..., 1].unsqueeze(-1)  # Mean position
        w = embedding[..., 2].unsqueeze(-1)  # FWHM
        nu = embedding[..., 3].unsqueeze(-1) # Lorentzian character fraction
        
        s = x.shape  # (_, num_fits)    
        if spec_len is not None:
            s = (s[0],-1,spec_len)
        
        x_ = torch.arange(dset.spec_len, dtype=torch.float32).repeat(s[0],s[1],1).to(device)
        
        # Calculate components
        gaussian = self._gaussian_component(A, x, x_, w)
        lorentzian = self._lorentzian_component(A, x, x_, w)
        
        # Pseudo-Voigt profile
        pseudovoigt = nu * lorentzian + (1 - nu) * gaussian

        return pseudovoigt.to(torch.float32)


class pseudovoigt_1D_fitters_new():
    '''https://www.surfacesciencewestern.com/wp-content/uploads/ass18_biesinger.pdf'''
    
    def __init__(self, limits=[1,1,975]):
        self.limits = limits
    
    def scale_parameters(self, embedding):
        h = self.limits[0] * embedding[..., 0] # amplitude
        E = self.limits[1] * embedding[..., 1] # mean
        F = self.limits[2] * embedding[..., 2] # fwhm
        nu = embedding[..., 3] # fraction voight character
        return torch.stack([h,E,F,nu],axis=2)

    def apply_activations(self, embedding):
        '''This function takes an embedding and scales it to the limits of the parameters
        
        This function implements the Pseudo-Voigt profile as described in:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/
        
        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - A: Area under curve (index 0)
                - x: Mean position (index 1)
                - w: Full Width at Half Maximum (FWHM) (index 2)
                - nu: Lorentzian character fraction (index 3)
            limits (list): Scale factors for [A, x, w]. Defaults to [1, 1, 975]
        '''
        h = nn.ReLU()(embedding[..., 0]) # amplitude
        E = torch.clamp(nn.Tanh()(embedding[..., 1])/2 + 0.5, min=1e-3) # mean
        F = torch.clamp(nn.Tanh()(embedding[..., 2])/2 + 0.5, min=1e-3) # fwhm
        nu = 0.5 * nn.Tanh()(embedding[..., 3]) + 0.5 # fraction voight character
        return torch.stack([h,E,F,nu],axis=2)

    def _gaussian_component(self, h, E, x_, F):
        """Calculate the Gaussian component of the Pseudo-Voigt profile
        
        Args:
            h (torch.Tensor): amplitude
            E (torch.Tensor): mean
            x_ (torch.Tensor): X-axis points
            sigma (torch.Tensor): standard deviation
        """
        gaussian = h * torch.exp( -4 * torch.log(torch.tensor(2)) \
                                     * ((x_-E)/F)**2 )
        return gaussian

    def _lorentzian_component(self, h, E, x_, F):
        """Calculate the Lorentzian component of the Pseudo-Voigt profile
        
        Args:
            h (torch.Tensor): amplitude
            E (torch.Tensor): mean
            x_ (torch.Tensor): X-axis points
            F (torch.Tensor): Full Width at Half Maximum (FWHM)
        """
        lorentzian = h /(1  + 4*((x_-E)/F)**2)
        return lorentzian

    def generate_fit(self, embedding, dset, spec_len=None):
        """Generate 1D Pseudo-Voigt profiles from embedding parameters.

        This function implements the Pseudo-Voigt profile as described in:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/

        The Pseudo-Voigt profile is a linear combination of Gaussian and Lorentzian profiles,
        controlled by the mixing parameter nu.

        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - A: Area under curve (index 0)
                - x: Mean position (index 1)
                - w: Full Width at Half Maximum (FWHM) (index 2)
                - nu: Lorentzian character fraction (index 3)
            dset: Dataset containing spectral information with attribute spec_len
            return_params (bool): If True, returns both profile and parameters. Defaults to False

        Returns:
            torch.Tensor: Pseudo-Voigt profiles of shape (batch_size, num_fits, spec_len)
            torch.Tensor: (Optional) Parameters [A, x, w, nu] if return_params=True
        """
        device = embedding.device
        # Unpack embedding tensor along last dimension (shape: [..., 4])
        h = embedding[..., 0].unsqueeze(-1)  # amplitude
        E = embedding[..., 1].unsqueeze(-1)  # mean
        F = embedding[..., 2].unsqueeze(-1)  # FWHM
        nu = embedding[..., 3].unsqueeze(-1) # Lorentzian character fraction
        
        s = h.shape  # (_, num_fits)    
        if spec_len is not None:
            s = (s[0],-1,spec_len)
        
        x_ = torch.arange(dset.spec_len, dtype=torch.float32).repeat(s[0],s[1],1).to(device)
        
        # Calculate components
        gaussian = self._gaussian_component(h, E, x_, F)
        lorentzian = self._lorentzian_component(h, E, x_, F)
        
        # Pseudo-Voigt profile
        pseudovoigt = nu * lorentzian + (1 - nu) * gaussian

        return pseudovoigt.to(torch.float32)


class Fitter_AE:
    """Autoencoder-based fitter for spectroscopic data.

    This class implements an autoencoder architecture for fitting spectroscopic data,
    particularly designed for Pseudo-Voigt profiles.

    Args:
        function (callable): Function to generate profiles from embeddings
        dset (Dataset): Dataset containing spectroscopic data
        num_params (int): Number of parameters in the embedding
        num_fits (int): Number of profiles to fit simultaneously
        limits (list): Scale factors for the profile parameters
        learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-3
        device (str, optional): Device to run computations on. Defaults to 'cuda:0'
        encoder (class, optional): Encoder architecture class. Defaults to Multiscale1DFitter
        encoder_params (dict, optional): Parameters for the encoder architecture

    Attributes:
        dset: The input dataset
        num_fits: Number of profiles to fit
        limits: Scale factors for parameters
        device: Computation device
        learning_rate: Optimizer learning rate
        encoder: The encoder model
        optimizer: Adam optimizer
        best_train_loss: Best training loss achieved
        checkpoint: Path to latest checkpoint
        folder: Directory for saving checkpoints
    """
    def __init__(self,
                 function, 
                 dset,
                 num_params,
                 num_fits,
                 input_channels,
                 learning_rate=3e-5,
                 device='cuda:0',
                 encoder = Multiscale1DFitter,
                 encoder_params = { "model_block_dict": { # factory wrapper for blocks
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
                "skip_connections": {"hidden_xfc": "hidden_embedding"} },
                checkpoint_folder='./checkpoints',
                sampler=None,
                sampler_params={},
                collate_fn=None,
            ):
        self.dset = dset
        self.num_fits = num_fits
        self.num_params = num_params
        self.device = device
        self.learning_rate = learning_rate
        self.collate_fn = collate_fn
        self.encoder_params = encoder_params
        self.encoder = encoder(function = function,
                                x_data = dset,
                                input_channels = input_channels,
                                num_fits = num_fits,
                                num_params = num_params,
                                device=device,
                                **encoder_params
                                ).to(self.device).type(torch.float32)
        self.optimizer = optim.Adam( self.encoder.parameters(), lr=self.learning_rate )
        self.configure_dataloader_sampler(sampler=sampler, **sampler_params)
        self.configure_dataloader(collate_fn=collate_fn)
        
        self.start_epoch = 0
        self.best_train_loss = float('inf')
        self.checkpoint = None
        self.scheduler = None
        self._checkpoint_folder = checkpoint_folder
        
    @property
    def dataloader_sampler(self): return self._dataloader_sampler   
    def configure_dataloader_sampler(self, **kwargs):
        '''Set the sampler for the dataloader'''
        batch_size = kwargs.get('batch_size', 32)
        orig_shape = kwargs.get('orig_shape', self.dset.shape)
        gaussian_std=kwargs.get('gaussian_std', 5)
        num_neighbors=kwargs.get('num_neighbors', 10)
        sampler = kwargs.get('sampler', None)
        
        # builds the dataloader
        if sampler is None: 
            self._dataloader_sampler = None
            self.binning = False
        else:
            self._dataloader_sampler = Gaussian_Sampler(self.dset, orig_shape, batch_size, gaussian_std, num_neighbors)
            self.binning = True
    
    @property
    def dataloader(self): return self._dataloader
    def configure_dataloader(self, **kwargs):
        '''Set the dataloader for the fitter
        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            sampler (Sampler, optional): Defines the strategy to draw samples from the dataset. Defaults to None.
            collate_fn (callable, optional): Merges a list of samples to form a mini-batch. Defaults to None.
            shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to False.
        '''
        batch_size = kwargs.get('batch_size', None)
        collate_fn = kwargs.get('collate_fn', None)
        shuffle = kwargs.get('shuffle', False)
        
        # builds the dataloader
        if self.dataloader_sampler is None: 
            self._dataloader = DataLoader(self.dset, batch_size=batch_size, shuffle=shuffle)
            self.binning = False
        else:
            self._dataloader = DataLoader(self.dset, batch_size=batch_size, sampler=self.dataloader_sampler, collate_fn=collate_fn, shuffle=shuffle)
            self.binning = True
        
    @property
    def checkpoint_folder(self): return self._checkpoint_folder
    @property 
    def checkpoint_file(self): return self._checkpoint_file
    @property
    def check(self): return self._check
    
    @property
    def checkpoint(self): return self._checkpoint
    @checkpoint.setter
    def checkpoint(self, value):
        self._checkpoint = value
        try:
            checkpoint_folder,checkpoint_file = os.path.split(self._checkpoint)
            self._checkpoint_file = checkpoint_file
            self._check = checkpoint_file.split('.pkl')[0]
            self._checkpoint_folder = checkpoint_folder
        except:
            self._check = None
            self._checkpoint_folder = None
            self._checkpoint_file = None
            
    
    def train(self, seed=42, epochs=100, weight_by_distance=False, save_every=1, batch_size=100):
        """Train the model.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42
            epochs (int): Number of training epochs. Defaults to 100
            binning (bool): Whether to use binning in loss calculation. Defaults to True
            weight_by_distance (bool): Whether to weight samples by distance. Defaults to True
        """
        today = date.today()
        save_date=today.strftime('(%Y-%m-%d)')
        make_folder(self.checkpoint_folder)
        print(os.path.abspath(self.checkpoint_folder))

        # set seed
        torch.manual_seed(seed)
        
        if self.binning:
            self.configure_dataloader(collate_fn=self.dataloader_sampler.custom_collate_fn)
        else:
            self.configure_dataloader(shuffle=True, batch_size=batch_size)
        
        # training loop
        for epoch in range(self.start_epoch, epochs):
            fill_embeddings = False # TODO: fill embeddings during training

            loss_dict = self.loss_function( self.dataloader,
                                            binning=self.binning,
                                            weight_by_distance=weight_by_distance, )
            
            # divide by batches inplace
            loss_dict.update( (k,v/len(self.dataloader)) for k,v in loss_dict.items())
            
            print(
                f'Epoch: {epoch:03d}/{epochs:03d} | Train Loss: {loss_dict["train_loss"]:.4f}')
            print('.............................')

          #  schedular.step()
          # TODO: add regularization losses
          # TODO: add embedding saver
          # TODO: add lr scheduler
            lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
            self.checkpoint = self.checkpoint_folder + f'/{save_date}_epoch:{epoch:04d}_lr:{lr_}_trainloss:{loss_dict["train_loss"]:.4f}.pkl'
            if epoch % save_every == 0: self.save_checkpoint(epoch, loss_dict=loss_dict,)

    def save_checkpoint(self,epoch,loss_dict,**kwargs): # TODO: needs to save sampler
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss_dict': loss_dict,
            'loss_params': kwargs,
            'sampler': self.dataloader_sampler,
        }
        torch.save(checkpoint, self.checkpoint)

    def load_weights(self, path_checkpoint):
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        try: self.configure_dataloader_sampler(sampler=checkpoint['sampler'])
        except: self.configure_dataloader_sampler(sampler=None)
        
        try: self.loss_dict = checkpoint['loss_dict']
        except: self.loss_dict = None
        
        try: self.loss_params = checkpoint['loss_params']
        except: self.loss_params = None

    def get_embedding(self, dset, batch_size=100):
        self.configure_dataloader_sampler(sampler=None)
        self.configure_dataloader(batch_size=batch_size)
        
        for i, (idx, x) in enumerate(tqdm(self.dataloader, leave=True, total=len(self.dataloader))):
            fits, params = self.encoder(x)
            
        return 
    # Loss stuff
    def _initialize_loss_components(self, train_iterator, coef1, coef2, coef3, coef4):
        """Initialize loss components and their coefficients"""
        components = {
            'weighted_ln': (Weighted_LN_loss(coef=coef1, channels=self.num_fits).to(self.device) if coef1 > 0 else None),
            'contrastive': (ContrastiveLoss(coef2).to(self.device) if coef2 > 0 else None),
            'divergence': (DivergenceLoss(train_iterator.batch_size, coef3).to(self.device) if coef3 > 0 else None),
            'sparse_max': (Sparse_Max_Loss(min_threshold=self.learning_rate, channels=self.num_fits, coef=coef4).to(self.device) if coef4 > 0 else None)
        }
        return components

    def _process_batch_binning(self, x, predicted_x, idx, weight_by_distance=False):
        """Process batch with binning and optional distance weighting"""
        x = list(torch.split(x, self.dataloader_sampler.num_neighbors))
        predicted_x = list(torch.split(predicted_x, self.dataloader_sampler.num_neighbors))
        
        if not weight_by_distance:
            x = torch.stack([x_.mean(dim=0) for x_ in x])
            predicted_x = torch.stack([x_.mean(dim=0) for x_ in predicted_x])
            return x.squeeze(), predicted_x.squeeze()
        
        # Weight by distance logic
        idx = torch.split(idx, self.dataloader_sampler.num_neighbors)
        weight_list = []
        
        for i_, sample_group in enumerate(idx):
            coords = [(int(ind % self.dset.shape[1]), int(ind / self.dset.shape[0])) for ind in sample_group]
            weights = torch.tensor([1] + [1 / (1 + ((coords[0][0]-coord[0])**2 + (coords[0][1]-coord[1])**2)**0.5) 
                                        for coord in coords[1:]], device=self.device)
            
            x[i_] = x[i_] * weights.unsqueeze(-1).unsqueeze(-1)
            predicted_x[i_] = predicted_x[i_] * weights.unsqueeze(-1).unsqueeze(-1)
            weight_list.append(weights)
        # TODO: check if weights correct, fix x shape
        weight_sums = torch.stack([w.sum(dim=0) for w in weight_list]).unsqueeze(-1).unsqueeze(-1)
        x = torch.stack([x_.sum(dim=0) for x_ in x]) / weight_sums
        predicted_x = torch.stack([x_.sum(dim=0) for x_ in predicted_x]) / weight_sums
        
        return x, predicted_x

    def _compute_losses(self, embedding, x, predicted_x, loss_components, coef5):
        """Compute all loss components"""
        loss_dict = {
            'weighted_ln_loss': 0, 'mse_loss': 0, 'train_loss': 0,
            'sparse_max_loss': 0, 'l2_batchwise_loss': 0
        }
        
        # Compute individual losses
        losses = {
            'reg_loss_1': loss_components['weighted_ln'](embedding[:,:,0]) if loss_components['weighted_ln'] else 0,
            'contras_loss': loss_components['contrastive'](embedding[:,:,0]) if loss_components['contrastive'] else 0,
            'maxi_loss': loss_components['divergence'](embedding[:,:,0]) if loss_components['divergence'] else 0,
            'sparse_max_loss': loss_components['sparse_max'](embedding[:,:,0]) if loss_components['sparse_max'] else 0,
        }
        
        # L2 batchwise loss
        if coef5 > 0:
            losses['l2_loss'] = coef5 * ((embedding[:,:,1]/embedding[:,:,2]).max(dim=0).values - 
                                        (embedding[:,:,1]/embedding[:,:,2]).min(dim=0).values).mean()
        else:
            losses['l2_loss'] = 0
        
        # MSE loss
        mse_loss = F.mse_loss(x, predicted_x, reduction='mean')
        
        # Update loss dictionary
        loss_dict.update({k: v for k, v in losses.items() if v != 0})
        loss_dict['mse_loss'] = mse_loss.item()
        
        # Compute total loss
        total_loss = mse_loss + losses['reg_loss_1'] + losses['contras_loss'] - losses['maxi_loss'] + losses['l2_loss']
        loss_dict['train_loss'] = total_loss.item()
        
        return total_loss, loss_dict

    def loss_function(self, train_iterator, coef1=0, coef2=0, coef3=0, coef4=0, coef5=0,
                     ln_parm=1, beta=None, fill_embeddings=False, minibatch_logging_rate=None,
                     binning=False, weight_by_distance=False):
        """Calculate the loss for training.

        Combines multiple loss components:
        - MSE loss between input and reconstructed spectra
        - Weighted LN loss for regularization
        - Contrastive loss for embedding space structure
        - Divergence loss for embedding distribution
        - Sparse max loss for sparsity
        - L2 batchwise loss for parameter consistency

        Args:
            train_iterator: DataLoader for training data
            coef1-5 (float): Coefficients for different loss components
            ln_parm (float): Parameter for weighted LN loss
            beta (float, optional): Parameter for variational loss
            fill_embeddings (bool): Whether to store embeddings
            minibatch_logging_rate (int, optional): Logging frequency
            binning (bool): Whether to use binning
            weight_by_distance (bool): Whether to weight samples by distance

        Returns:
            dict: Dictionary containing different loss components and total loss
        """
        self.encoder.train()
        loss_components = self._initialize_loss_components(train_iterator, coef1, coef2, coef3, coef4)
        accumulated_loss_dict = {'weighted_ln_loss': 0, 'mse_loss': 0, 'train_loss': 0,
                               'sparse_max_loss': 0, 'l2_batchwise_loss': 0}

        for i, (idx, x) in enumerate(tqdm(train_iterator, leave=True, total=len(train_iterator))):
            idx = idx.to(self.device).squeeze()
            x = x.to(self.device, dtype=torch.float)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if beta is None:
                predicted_x, embedding = self.encoder(x)
            else:
                predicted_x, embedding, sd, mn = self.encoder(x, beta)
            
            # Process binning if needed
            if binning:
                x, predicted_x = self._process_batch_binning(x, predicted_x, idx, weight_by_distance)
            
            # Compute losses
            loss, batch_loss_dict = self._compute_losses(embedding, x, predicted_x.sum(axis=1), loss_components, coef5)
            
            # Update accumulated losses
            for k in accumulated_loss_dict:
                if k in batch_loss_dict:
                    accumulated_loss_dict[k] += batch_loss_dict[k]
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Handle embeddings and logging
            if fill_embeddings:
                sorted_idx, indices = torch.sort(idx)
                self.embedding[sorted_idx.detach().numpy()] = embedding[indices].cpu().detach().numpy()
                
            if minibatch_logging_rate and i % minibatch_logging_rate == 0:
                wandb.log({k: v/(i+1) for k, v in accumulated_loss_dict.items()})

        return accumulated_loss_dict
