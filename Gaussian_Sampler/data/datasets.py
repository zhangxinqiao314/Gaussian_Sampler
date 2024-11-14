import m3_learning
from m3_learning.nn.STEM_AE import STEM_AE
import numpy as np
import torch
from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from tqdm import tqdm
import torch.nn.functional as F
import py4DSTEM
import dask.array as da        
import aux_func
from tqdm import tqdm

class Py4DSTEM_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_data, binfactor, block=None, center=None, **kwargs):
        '''
        dm4 file
        '''
        print('Loading dataset...')
        self.raw_data = py4DSTEM.import_file(file_data, binfactor=binfactor)
        self.raw_data.get_dp_mean()
        self.data = self.raw_data.data
        self.block=block
        
        if center is not None: self.center = center
        else: self.center = [self.data.shape[-2]//2, self.data.shape[-1]//2]
            
        print('Preprocessing data...')
        self.log_data = self._clean_data(center=self.center, **kwargs)
            
        print('Done.')
        self.shape = self.raw_data.shape[:2]+self.log_data.shape[-2:]

    def _clean_data(self, hot_px_threshold=None, log=True, standard=True, minmax=True, center=None, stdv_thresh=None):
        '''
        Remove hot pxs
        do log
        do minmax scaling
        '''
        
        if hot_px_threshold is not None:
            print('Removing hot pixels...')
            self.raw_data.get_dp_mean()
            dataset, mask_hot_pixels = aux_func.remove_hot_pixels(self.raw_data, 
                                                                  self.raw_data.tree['dp_mean'].data, 
                                                                  relative_threshold=hot_px_threshold )
            dataset = dataset.data
        else: dataset = self.raw_data.data
        
        if center is not None:
            print('Centering data...')
            bound = min([center[0],center[1],abs(dataset.shape[-2]-center[0]), abs(dataset.shape[-1]-center[1])])
            dataset = dataset[...,
                              self.center[0]-bound:self.center[0]+bound,
                              self.center[1]-bound:self.center[1]+bound]
                
        print('to dask array...')
        data = da.from_array(dataset, chunks='auto')
        data = data.reshape(-1, 
                        dataset.shape[-2], 
                        dataset.shape[-1])
        
        if self.block>0:
            print('Blocking center beam...')
            for i in range(dataset.shape[-1]):
                for j in range(dataset.shape[-2]):
                    if ( (dataset.shape[-2]//2-i)**2 + (dataset.shape[-1]//2-j)**2 ) < (self.block**2):
                        dataset[:,:,i,j] = 0
                        
        if stdv_thresh is not None:
            thresh = data.mean() + data.std() * stdv_thresh
            print(f'Thresholding to {thresh.compute():0.2f}/{data.max().compute():0.2f}...')
            inds = da.argwhere(data > thresh).compute()
            for ind in tqdm(inds):
                data[tuple(ind)] = thresh
                
        if standard:
            print('Standard scaling data...')
            data = (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)
        
        print('Removing NaNs...')
        data = da.nan_to_num(data)    
        
        if log:
            print('Log scaling data...')
            data = data - data.min() + 1 + 1e-10
            data = da.log(data)
        
        if minmax:
            print('Minmax scaling data...')
            data = (data - data.min()) / (data.max() - data.min())
            
        
        print('Computing to np array...')
        data = data.compute()
        return data
        
    def __len__(self):
        return len(self.log_data)

    def __getitem__(self, idx):
        return self.log_data[idx]

class Py4DSTEM_Embeddings(torch.utils.data.Dataset):
    def __init__(self, dset, checkpoint, model, embedding=None, **kwargs):
        '''
        dm4 file
        '''
        self.checkpoint = checkpoint
        self.model = model
        self.model.load_weights(self.checkpoint)
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = self._get_embedding(dset,**kwargs)
        self.shape = self.embedding.shape
        
    def _get_embedding(self, *args, **kwargs):
        self.embedding = self.model.get_embedding(self.data,batch_size=100)
        
    def _save_embedding(self, path='./embedding.npy'):
        np.save(path, self.embedding)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.embedding[idx]