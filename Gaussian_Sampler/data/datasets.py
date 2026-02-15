from types import NoneType
from typing import Iterable
import numpy as np
import torch
# from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
import torch.nn.functional as F
from torch.autograd import Variable
import dask.array as da        
from tqdm import tqdm
import h5py 
import joblib
import io
import os

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from ..models.pv_fitter import Fitter_AE

def draw_m_in_array(size_=100):
    '''
        # self.mask = np.ones((self.shape[0], self.shape[1])); self.mask[40:60,30:50] = 0; self.mask = self.mask.flatten()
        self.mask = draw_m_in_array(self.shape[0]).flatten()
    '''
    arr_ = np.zeros((size_, size_), dtype=int)
    w=size_//10
    size=int(size_/1.5)
    arr = np.zeros((size, size), dtype=int)

    for i in range(size):
        # Left vertical line
        arr[i, 0:w] = 1
        # Right vertical line
        arr[i, size-w:size] = 1
        # Diagonal from left to middle
        if w <= i < size // 2:
            arr[size-i-w:size-i+w,i] = 1
            # Diagonal from right to middle
            arr[size-i-w:size-i+w, size-(i+1)] = 1
        arr_[size_//6:size_//6+size,size_//6:size_ //6+size] = arr
    return arr_


class Poisson_Sampled_PV_Dataset(torch.utils.data.Dataset): # tODO: set seed for random number generation
    # TODO: try loading scaler/param classes if it exists, 
    # TODO: getitem unscaled dataset
    def __init__(self, scaled=False, 
                 shape=[100,100,500], 
                 save_folder='./', 
                 overwrite=False, 
                 pv_fitter=None,
                 num_classes=5,
                 num_curves=3,
                 scaler='default',
                 norm_calculation=lambda x: np.linalg.norm(x,axis=-1, ord=np.inf),
                 dset_num = 0):
        '''dset is x*y,spec_len'''
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        self.pv_fitter = pv_fitter
        self.h5_name = f'{self.save_folder}_poisson_sampled_pv.h5'
        self._dset_name = f'{1:06.3f}_sample_rate'
        self.shape = shape
        self.spec_len = self.shape[-1]
        self.scaler = scaler
        self.norm_calculation = norm_calculation
        
        # set parameters for generating PV curves
        if overwrite:
            self.pv_param_classes = {
                'a': np.random.random_integers(0, 10, (num_classes, num_curves)),
                'E': np.random.random_integers(0, shape[-1], (num_classes, num_curves,)),
                'F': np.random.random_integers(1, shape[-1] // 2, (num_classes, num_curves,)),
                'nu': np.random.random((num_classes, num_curves,))
            }
            self.generate_pv_data()
        else: 
            self._read_pv_param_classes()
            self.dset_names = self.h5_keys()
            self._dset_index = 0
            if scaler == 'default': self._read_scaler()
        
        self.zero_dset = self.getitem_zero_dset(range(self.shape[0]*self.shape[1]))[1]
        self.maxes = self.zero_dset.max(axis=-1).reshape(self.shape[:-1]+(1,))
    
    @property
    def dset_index(self): return self._dset_index
    @dset_index.setter
    def dset_index(self, i):
        self._dset_index = i 
    
    @property
    def dset_name(self): return self.dset_names[self._dset_index]    
    @dset_name.setter
    def dset_name(self, name):
        self._dset_index = self.dset_names.index(name)
            
    def lower_signal(self,y, sample_rate=1, background_noise=0):
        """
        Simulate low-signal measurement with Poisson statistics.
        args:
            y: torch.Tensor, the signal to be reduced
            sample_rate: float, the sample rate to be applied
            background_noise: float, the background noise to be added
        returns:
            torch.Tensor, the reduced signal
        """
        # Reduce signal intensity (simulating short exposure/weak source)
        reduced = y * sample_rate + background_noise
        
        if sample_rate == 1: return reduced
        else: return torch.poisson(reduced)
    
    def fit_scaler(self, data):
        if self.scaler is not None: self.scaler.fit(data)
        
        if self.norm_calculation is not None:
            self.maxes = self.norm_calculation(data)
        
    def scale_data(self, data): 
        if self.scaler is None: return data
        return self.scaler.transform(data)

    def unscale_data(self, scaled_data, recalculate_maxes=True):
        if recalculate_maxes: self.calculate_maxes()
        try: 
            return self.scaler.inverse_transform(scaled_data.reshape(-1, scaled_data.shape[-1])).reshape(scaled_data.shape)
        except: 
            return self.maxes.reshape((-1,) + (1,) * (scaled_data.ndim - 1)) * scaled_data # TODO: write maxes as metadata
        
    def calculate_maxes(self):
        with self.open_h5_file() as f:
            self.maxes = self.norm_calculation(f['unscaled'][self.dset_name][:]).reshape(-1)
        
    def __len__(self): return (self.shape[0]*self.shape[1])

    def __getitem__(self, idx):
        # idx=7889
        with self.open_h5_file() as f:
            try: data = np.array([f['scaled'][self.dset_name][i] for i in idx])
            except: data = f['scaled'][self.dset_name][idx]

            return idx, data
        
    def getitem_zero_dset(self,idx):
        # idx=7889
        with self.open_h5_file() as f:
            try: data = np.array([f['scaled'][self.dset_names[0]][i] for i in idx])
            except: data = f['scaled'][self.dset_names[0]][idx]
            
        return idx,data
    
    def open_h5_file(self): return h5py.File(self.h5_name, 'a')
    
    def h5_keys(self): 
        with self.open_h5_file() as f:
            keys = list(f['unscaled'].keys())
        return keys
    
    def create_concentric_circles(self, fits):
        """Create filled concentric circles where each ring corresponds to a class from fits."""
        # Convert to torch if needed
        if not isinstance(fits, torch.Tensor):
            fits = torch.tensor(fits)
        
        n, m, s = self.shape
        numclasses = fits.shape[0]
        device = fits.device
        dtype = fits.dtype
        
        y, x = torch.meshgrid(torch.arange(n, device=device), torch.arange(m, device=device), indexing='ij')
        r = torch.sqrt((x - m/2)**2 + (y - n/2)**2)
        max_r = torch.sqrt(torch.tensor((n/2)**2 + (m/2)**2, device=device))
        # Use 90% of max radius so circles don't touch edges
        circle_radius = max_r * 0.75
        ring_idx = torch.clamp((r / circle_radius * numclasses).long(), 0, numclasses - 1)
        # Set pixels outside the outermost circle to 0
        mask = r <= circle_radius
        result = fits[ring_idx] * mask.unsqueeze(-1)
        return result
    
    def _write_unscaled_dataset(self, dset_name, sampled_data, fit_shape):
        """Write unscaled dataset to h5 file."""
        with self.open_h5_file() as f:
            # write pv curve generation parameters to h5 file unscaled group
            try: f.create_group('unscaled')
            except: pass
            
            for k,v in self.pv_param_classes.items():
                f['unscaled'].attrs[k] = v
            
            try: del f['unscaled'][dset_name]
            except: pass
            
            dset = f['unscaled'].create_dataset(dset_name,
                                                data=sampled_data.reshape(-1, fit_shape[-1]),
                                                dtype=np.float32)
            f.flush()
          
    def _write_scaled_dataset(self):
        """Write scaled dataset to h5 file."""
        print("Writing scaled dataset...")
        with self.open_h5_file() as f:      
            # write scaler to h5 file scaled group
            try: f.create_group('scaled')
            except: pass
            
            
            for i in tqdm(range(10)):
                dset_name = self.dset_names[i]
                sampled_data = f['unscaled'][dset_name][:]
                self.fit_scaler(data=sampled_data)
                
                try: del f['scaled'][dset_name]
                except: pass
                
                dset = f['scaled'].create_dataset(dset_name,
                                              data=self.scale_data(sampled_data),
                                              dtype=np.float32)
                for k,v in self._get_scaler_buf().items():
                    dset.attrs[k] = v
                
            f.flush()
      
    def _get_scaler_buf(self):
        '''Get dset.scaler as bytes buffer.'''
        buf = io.BytesIO()
        joblib.dump(self.scaler, buf)
        buf.seek(0)
        return {
            "scaler_joblib": np.void(buf.read()),
        } 
        
    def _read_scaler_buf(self, dset_path):
        '''Read h5 attrs scaler as bytes buffer.'''
        with self.open_h5_file() as f:
            self.scaler = joblib.load(io.BytesIO(f[dset_path].attrs["scaler_joblib"]))
        return self.scaler
   
    def _read_pv_param_classes(self):
        with self.open_h5_file() as f:
            self.pv_param_classes = {k: v for k,v in f['unscaled'].attrs.items()}
            return self.pv_param_classes
        
    def generate_pv_data(self):
        '''This function takes a dictionary of parameters classes and returns a numpy array of parameters'''
        
        print('Generating data...')
        embeddings = torch.stack( [torch.tensor(x) for x in self.pv_param_classes.values()], axis=2) # shape (numclasses, numcurves, params)
        fits = self.pv_fitter.generate_fit(embeddings,spec_len=self.spec_len)
        fits = fits.sum(axis=1)
        fit = self.create_concentric_circles(fits).reshape(self.shape[0]*self.shape[1], -1)
        # make tile this in 100x100 square
        for i in tqdm(range(10)):
            sample_rate = 10**(-(i / 10))
            dset_name = f'{i:02d}_{sample_rate:06.3f}_sample_rate'
            sampled_data = self.lower_signal(y=fit, sample_rate=sample_rate, background_noise=0.1)
            self._write_unscaled_dataset(dset_name, sampled_data, fit.shape)
        self.dset_names = self.h5_keys()
        self._write_scaled_dataset()


class Poisson_Sampled_PV_Embeddings():
    """Dataset class for accessing embeddings (fits and params) from h5 file."""
    
    def __init__(self, model, dset, checkpoint, scaled=True):
        """
        Args:
            model: Model object (Fitter_AE) that has embedding_h5_name attribute. must be initialized
            dset: Poisson_Sampled_PV_Dataset object. must be initialized
            scaled: If True, use scaled embeddings; if False, use unscaled
        """
        self.model = model
        self.dset = dset
        # Set initial dataset name
        self.checkpoint = checkpoint # setting is done with setter method
        
        # Determine which group to use
        self.group = 'scaled' if scaled else 'unscaled'
        
        
    @property
    def checkpoint(self): return self._checkpoint
    
    @checkpoint.setter
    def checkpoint(self, checkpoint_):
        self._checkpoint = checkpoint_
        self.model.load_weights(checkpoint_)
        self.dset.dset_name = checkpoint_.split('/')[-2]
    
    def open_embedding_h5(self):
        """Open the embedding h5 file."""
        return h5py.File(self.model.embedding_h5_name, 'r')
    
    def _check_embedding_tree_structure(self):
        """Check and create embedding tree structure if needed, efficiently."""
        with h5py.File(self.model.embedding_h5_name, 'a') as f:
            # Ensure dataset group exists (created only if missing)
            dset_grp = f.require_group(self.dset.dset_name)
            scaled_grp = dset_grp.require_group('scaled')
            unscaled_grp = dset_grp.require_group('unscaled')
            
            # Helper for datasets with checkpoint metadata
            def ensure_dataset(g, name, shape):
                if name not in g:
                    ds = g.create_dataset(name, shape=shape, dtype=np.float32)
                    # Store checkpoint name as metadata
                    ds.attrs['checkpoint'] = self.model.check
                else:
                    # Update checkpoint metadata
                    g[name].attrs['checkpoint'] = self.model.check

            # Ensure scaled datasets
            ensure_dataset(scaled_grp, 'fits', (len(self.dset), self.model.num_fits, self.dset.shape[-1]))
            ensure_dataset(scaled_grp, 'params', (len(self.dset), self.model.num_fits, self.model.num_params))
            # Ensure unscaled datasets
            ensure_dataset(unscaled_grp, 'fits', (len(self.dset), self.model.num_fits, self.dset.shape[-1]))
            ensure_dataset(unscaled_grp, 'params', (len(self.dset), self.model.num_fits, self.model.num_params))

            f.flush()
    
    def _unscale_embedding(self):
        """Write unscaled dataset to h5 file."""
        with h5py.File(self.model.embedding_h5_name, 'a') as f:
            # Read scaler from original dataset file
            self.dset.scaler = self.dset._read_scaler_buf(dset_path=f'scaled/{self.dset.dset_name}')
            # Unscale fits
            f[self.dset.dset_name]['unscaled']['fits'][:] = self.dset.unscale_data(
                f[self.dset.dset_name]['scaled']['fits'][:]
            )
            # Copy params (most don't need unscaling)
            f[self.dset.dset_name]['unscaled']['params'][:] = f[self.dset.dset_name]['scaled']['params'][:]
            # Unscale first parameter (Amplitude) only
            f[self.dset.dset_name]['unscaled']['params'][...,0] = self.dset.unscale_data(
                f[self.dset.dset_name]['unscaled']['params'][...,0], 
                recalculate_maxes=False
            )
                   
    def _write_scaled_embedding(self, batch_size=100):
        """Write scaled dataset to h5 file."""
        with h5py.File(self.model.embedding_h5_name, 'a') as f:
            for i, (idx, x) in enumerate(tqdm(self.model.dataloader, leave=True, total=len(self.model.dataloader))):
                with torch.no_grad():
                    fits, params = self.model.encoder(x.to(self.model.device))
                    f[self.dset.dset_name]['scaled']['fits'][i*batch_size:(i+1)*batch_size] = fits.cpu().numpy()
                    f[self.dset.dset_name]['scaled']['params'][i*batch_size:(i+1)*batch_size] = params.cpu().numpy()

            f.flush()
    
    def write_embeddings(self, batch_size=100):
        """Write embeddings to h5 file.
        
        Saved in folder with dataset scaling method (ie, '../../toy_dataset/l1_norm') 
        File structure:
            embedding_h5_File
            |-- dset_name_group
            |   |-- scaled_group
            |   |   |-- fits (not summed over fits, with checkpoint as attribute)
            |   |   |-- params (with checkpoint as attribute)
            |   |-- unscaled_group
            |   |   |-- fits (not summed over fits, with checkpoint as attribute)
            |   |   |-- params (with checkpoint as attribute)
        
        The checkpoint name is stored as metadata (attribute) on each dataset.
        Args:
            batch_size (int): Batch size for writing embeddings. Defaults to 100.
        """
        # write embeddings
        self.model.configure_dataloader_sampler(sampler=None)
        self.model.configure_dataloader(batch_size=batch_size)
        self._check_embedding_tree_structure()
        
        self._write_scaled_embedding()
        self._unscale_embedding()
    
    def __len__(self):
        """Return the length of the dataset."""
        with self.open_embedding_h5() as f:
            return len(f[self.dset.dset_name][self.group]['fits'])
    
    def __getitem__(self, idx, which='fits'):
        """Return fits and params for a given index.
        
        Args:
            idx: Index or slice of indices
            
        Returns:
            tuple: (fits, params) where:
                - fits: numpy array of shape (num_fits, spec_len) or (len(idx), num_fits, spec_len)
                - params: numpy array of shape (num_fits, num_params) or (len(idx), num_fits, num_params)
        """
        with self.open_embedding_h5() as f:
            fits = f[self.dset.dset_name][self.group]['fits'][idx]
            params = f[self.dset.dset_name][self.group]['params'][idx]
            return fits, params

