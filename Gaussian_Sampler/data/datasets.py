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
        print('Preprocessing data...')
        self.data = self._clean_data(self.dataset.data.reshape(-1, 1, 
                                                               self.dataset.data.shape[-2], 
                                                               self.dataset.data.shape[-1]))
        print('Done.')
        self.shape = self.data.shape

    def _clean_data(self, dat):
        '''
        Remove NaNs and Infs from the data
        do log
        do minmax scaling
        '''
        data = da.from_array(dat, chunks='auto')
        
        # print('Removing NaNs and Infs from data...')
        # data = da.where(da.isnan(data), 0, data)
        # data = da.where(da.isinf(data), 0, data)
        
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
