import numpy as np
import torch
# from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from tqdm import tqdm
import torch.nn.functional as F
import py4DSTEM
import dask.array as da        
from . import aux_func
from tqdm import tqdm
import h5py 

class Fake_PV_Dataset(torch.utils.data.Dataset):
    def __init__(self, scaled=False, shape=(100,100,500), save_folder='./', overwrite=False):
        '''dset is x*y,spec_len'''
        self.save_folder = save_folder
        self.fwhm, self.nu_ = 50, 0.7
        self.shape = shape
        self.spec_len = self.shape[-1]
        self.mask = np.zeros((self.shape[0], self.shape[1])); self.mask[10:,10:] = 1
        if overwrite: self.generate_pv_data()
        
        self.zero_dset = self.open_h5()[list(self.open_h5().keys())[0]][:]
        self.maxes = self.zero_dset.max(axis=-1).reshape(self.shape[:-1]+(1,))
        self.scale = scaled
        self.noise_ = self.h5_keys()[0]
        
    @staticmethod
    def noise(i): 
        # return (i/20)**(1.5)
        return i/20
    
    @staticmethod
    def write_pseudovoight(A,x,w=5,nu=0.25,spec_len=500,x_max=500):
        x_ = np.linspace(0,x_max-1,spec_len)
        lorentz = A*( nu*2/np.pi*w/(4*(x-x_)**2 + w**2) )
        gauss = A * (4*np.log(2)/np.pi**0.5 /w) * np.exp(-4*np.log(2)*(x-x_)**2/w**2)
        y = nu*lorentz + (1-nu)*gauss
        return y

    @staticmethod
    def add_noise(I,y,noise=0.1,spec_len=500):
        noise = np.random.normal(0, noise*(I), spec_len) # make some noise even if 0
        noisy = y + noise
        noisy[noisy<0] = 0
        return noisy

    @staticmethod
    def pv_area(I,w,nu): return I*w*np.pi/2/ ((1-nu)*(np.pi*np.log(2))**0.5 + nu)

    def __len__(self): return (self.shape[0]*self.shape[1])

    def __getitem__(self, idx):
        with self.open_h5() as f:
            data = f[self.noise_][idx]
            if self.scale:
                maxes = self.maxes[idx]
                non_zero = np.where(maxes[0,0]>0)
                data[non_zero] = data[non_zero]/maxes[non_zero]
            return idx, data
    
    def open_h5(self): return h5py.File(f'{self.save_folder}fake_pv_uniform.h5','a')
    
    def h5_keys(self): return list(self.open_h5().keys())
    
    def unscale(self, data, idx): return data*torch.tensor(self.maxes[idx]).to(data.device)

    def generate_pv_data(self):
        raw_data_path = f'{self.save_folder}fake_pv_uniform.h5'
        print('Generating data...')
        with h5py.File(raw_data_path,'a') as f:
            for i in tqdm(range(20)):
                noise_ = Fake_PV_Dataset.noise(i)
                try: dset = f[f'{noise_:06.3f}_noise']
                except: dset = f.create_dataset(f'{noise_:06.3f}_noise', shape=(100,100,500), dtype=np.float32)
                for x_ in range(dset.shape[0]):
                    for y_ in range(dset.shape[1]):
                        I = y_/5
                        A = Fake_PV_Dataset.pv_area(I, w=self.fwhm, nu=self.nu_)
                        if self.mask[x_, y_]:
                            dset[x_, y_] = Fake_PV_Dataset.add_noise(I,
                                                    Fake_PV_Dataset.write_pseudovoight(A, x_*2, self.fwhm, self.nu_),
                                                    noise = noise_)
                        else: dset[x_, y_] = Fake_PV_Dataset.add_noise(I,
                                                       np.zeros(dset.shape[2]),
                                                       noise = noise_)
                f.flush()

class Fake_PV_Embeddings(torch.utils.data.Dataset):
    def __init__(self, dset, model, checkpoint, **kwargs):
        self.model = model
        self.checkpoint = checkpoint
        self.model.load_weights(self.checkpoint)
        self.embedding = self.model.get_embedding(dset, batch_size=100)
        self.shape = self.embedding.shape




class Py4DSTEM_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_data, binfactor, block=0, center=None, **kwargs):
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
            
        data = da.from_array(dataset, chunks='auto')
        data = data.reshape(-1, 
                        dataset.shape[-2], 
                        dataset.shape[-1])
        
        if self.block>0:
            print('Blocking center beam...')
            for i in range(data.shape[-1]):
                for j in range(data.shape[-2]):
                    if ( (data.shape[-2]//2-i)**2 + (data.shape[-1]//2-j)**2 ) < (self.block**2):
                        data[:,:,i,j] = 0
                        
        if stdv_thresh is not None:
            print(f'Thresholding {stdv_thresh} standard deviations...')
            thresh = data.mean(axis=(1, 2)) + data.std(axis=(1, 2)) * stdv_thresh
            mask = data > thresh[:, None, None]
            data = da.where(mask, thresh[:, None, None], data)
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