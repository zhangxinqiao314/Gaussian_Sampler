import m3_learning
from m3_learning.nn.STEM_AE import STEM_AE
import numpy as np
import torch
from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from tqdm import tqdm
import torch.nn.functional as F
import py4DSTEM
import dask.array as da        


class Py4DSTEM_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_data, binfactor, **kwargs):
        '''
        dm4 file
        '''
        print('Loading dataset...')
        self.raw_data = py4DSTEM.import_file(file_data, binfactor=binfactor)
        self.data = self.raw_data.data
        print('Preprocessing data...')
        self.log_data = self._clean_data(self.data.reshape(-1, 
                                                               self.data.data.shape[-2], 
                                                               self.data.data.shape[-1]))
        print('Done.')
        self.shape = self.log_data.shape

    def _clean_data(self, dat):
        '''
        Remove NaNs and Infs from the data
        do log
        do minmax scaling
        '''
        data = da.from_array(dat, chunks='auto')
        
        print('Log scaling data...')
        data = data - data.min() + 1 + 1e-10
        data = da.log(data)
        
        print('Minmax scaling data...')
        data = (data - data.min()) / (data.max() - data.min())
        
        # # Show progress bar
        # with tqdm(total=data.npartitions, desc="Cleaning data") as pbar:
        #     def update_pbar(future):
        #         pbar.update()
                
        #     data = data.persist()
        #     da.compute(data, scheduler='threads', callback=update_pbar)
        
        print('Convert to np array...')
        data = data.compute()
        return data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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