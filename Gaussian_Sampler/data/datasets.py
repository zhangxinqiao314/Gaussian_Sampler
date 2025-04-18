from typing import Iterable
import numpy as np
import torch
# from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
import torch.nn.functional as F
from torch.autograd import Variable
import dask.array as da        
from tqdm import tqdm
import h5py 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


def draw_m_in_array(size_=100):
    arr_ = np.zeros((size_, size_), dtype=int)
    w=size_//10
    size=size_//2
    arr = np.zeros((size, size), dtype=int)

    for i in range(size):
        # Left vertical line
        arr[i, 0:w] = 1
        # Right vertical line
        arr[i, size-w:size] = 1
        # Diagonal from left to middle
        if w <= i < size // 2:
            arr[i-w:i+w,i] = 1
            # Diagonal from right to middle
            arr[i-w:i+w, size-(i+1)] = 1
        arr_[size_//4:size_//4+size,size_//4:size_ //4+size] = arr
    return arr_


class Fake_PV_Dataset(torch.utils.data.Dataset):
    def __init__(self, scaled=False, shape=[100,100,500], save_folder='./', overwrite=False, 
                 scaler=Pipeline([('scaler', StandardScaler()), ('minmax', MinMaxScaler())]),
                 scaling_kernel_size = 1, noise_level = 0):
        '''dset is x*y,spec_len'''
        self.save_folder = save_folder
        self.h5_name = f'{self.save_folder}fake_pv_uniform.h5'
        self.fwhm, self.nu_ = 50, 0.7
        self.shape = shape
        self.spec_len = self.shape[-1]
        self.mask = np.ones((self.shape[0], self.shape[1])); self.mask[40:60,30:50] = 0; self.mask = self.mask.flatten()
        # self.mask = draw_m_in_array(self.shape[0]).flatten()
        if overwrite: self.generate_pv_data()
        
        self.scale = scaled
        self.noise_levels = list(self.h5_keys())
        self._noise = self.noise_levels[noise_level]
        self.scaler = scaler
        self.scaling_kernel_size = scaling_kernel_size
        self.zero_dset = self.open_h5()[list(self.open_h5().keys())[0]][:]

        if scaled:
            self.fit_scalers()
            for i in tqdm(range(len(self.zero_dset)), desc='Scaling zero dset'):
                self.zero_dset[i] = self.scale_data(self.zero_dset[i], i)
        
    @property
    def noise_(self): return self._noise
    @noise_.setter
    def noise_(self, i):
        old_noise = self._noise
        self._noise = self.h5_keys()[i] if isinstance(i, int) else i
        if old_noise != self._noise and self.scale:
            self.fit_scalers()
        
    @staticmethod
    def noise(i): 
        # return (i/20)**(1.5)
        return i/20
    
    def write_pseudovoight(self,A,x,w=5,nu=0.25,):
        x_ = np.linspace(0,self.shape[-1]-1,self.shape[-1])
        lorentz = A*( nu*2/np.pi*w/(4*(x-x_)**2 + w**2) )
        gauss = A * (4*np.log(2)/np.pi**0.5 /w) * np.exp(-4*np.log(2)*(x-x_)**2/w**2)
        y = nu*lorentz + (1-nu)*gauss
        return y

    def add_noise(self,I,y,noise=0.1,):
        noise = np.random.normal(0, noise*(I), self.shape[-1]) # make some noise even if 0
        noisy = y + noise
        noisy[noisy<0] = 0
        return noisy
    
    def fit_scalers(self):
        ''' scales pixel of data using a kernel of size kernel_size
        args:
            kernel_size: size of the kernel to use for scaling
            scalers: list of scalers to use for scaling
        returns:
            data: scaled data
        '''
        if self.scaling_kernel_size%2 == 0: 
            raise ValueError('kernel_size must be odd')
        if self.scaling_kernel_size > self.shape[0] or self.scaling_kernel_size > self.shape[1]: 
            raise ValueError('kernel_size must be less than the shape of the data')
        
        self.kernel_scalers = []
        with self.open_h5() as f:
            if self.scaling_kernel_size==1:
                self.kernel_scalers = []
                for dat in tqdm(f[self.noise_], desc=f"Fitting scalers for {self.noise_}"):
                    self.kernel_scalers.append( self.scaler.fit(dat.reshape(-1, 1)) )
            else:
                for idx in tqdm(range(self.shape[0]*self.shape[1]), desc=f'Fitting scalers for {self.noise_}'):
                    x_ = idx//self.shape[1]
                    y_ = idx%self.shape[1]
                    
                    # Calculate valid kernel boundaries
                    x_start = max(0, x_-self.scaling_kernel_size//2)
                    x_end = min(self.shape[1], x_+self.scaling_kernel_size//2+1)
                    y_start = max(0, y_-self.scaling_kernel_size//2)
                    y_end = min(self.shape[0], y_+self.scaling_kernel_size//2+1)
                    
                    # Calculate flattened indices for the kernel region
                    points = [y*self.shape[1]+x for y in range(y_start,y_end) for x in range(x_start, x_end)]
                    data = f[self.noise_][points]
                    self.kernel_scalers.append(self.scaler.fit(data.reshape(-1, 1)))
 
    def scale_data(self, data, idx):
        if isinstance(idx, int):
            scaled_data = self.kernel_scalers[idx].transform(data.reshape(-1, 1)).reshape(data.shape)
        elif isinstance(idx, slice):
            scalers = [self.kernel_scalers[i] for i in range(*idx.indices(len(self.kernel_scalers)))]
            scaled_data = np.array([scaler.transform(dat.reshape(-1, 1)).reshape(dat.shape) \
                                    for scaler,dat in zip(scalers, data)])
        else:
            scalers = [self.kernel_scalers[i] for i in idx]
            scaled_data = np.array([scaler.transform(dat.reshape(-1, 1)).reshape(dat.shape) \
                                    for scaler,dat in zip(scalers, data)])
        return scaled_data
       
    def unscale_data(self, unscaled_data, scaled_data):
        self.scaler.fit(unscaled_data.reshape(-1, unscaled_data.shape[-1]))
        unscaled_data = self.scaler.inverse_transform(scaled_data.reshape(-1, scaled_data.shape[-1])).reshape(scaled_data.shape)
        return unscaled_data

    @staticmethod
    def pv_area(I,w,nu): return I*w*np.pi/2/ ((1-nu)*(np.pi*np.log(2))**0.5 + nu)

    def __len__(self): return (self.shape[0]*self.shape[1])

    def __getitem__(self, idx):
        idx=7889
        with self.open_h5() as f:
            try: data = np.array([f[self.noise_][i] for i in idx])
            except: data = f[self.noise_][idx]
            
            if self.scale: data = self.scale_data(data, idx)
            
            return idx, data
    
    
    def open_h5(self): return h5py.File(self.h5_name, 'a')
    
    def h5_keys(self): return list(self.open_h5().keys())
    
    def generate_pv_data(self):
        print('Generating data...')
        with self.open_h5() as f:   

            for i in tqdm(range(20)):
                noise_ = Fake_PV_Dataset.noise(i)
                try: del f[f'{noise_:06.3f}_noise']
                except: pass
                dset = f.create_dataset(f'{noise_:06.3f}_noise', 
                                                shape=(self.shape[0]*self.shape[1],self.shape[2]), dtype=np.float32)
                for x_ in range(self.shape[0]):
                    for y_ in range(self.shape[1]):
                        I = y_/5
                        A = Fake_PV_Dataset.pv_area(I, w=self.fwhm, nu=self.nu_)
                        if self.mask[y_+x_*self.shape[0]] == 1:
                            dset[x_+y_*self.shape[0]] = self.add_noise(I,
                                                    self.write_pseudovoight(A, x_*2, self.fwhm, self.nu_),
                                                    noise = noise_)
                        else: dset[x_+y_*self.shape[0]] = self.add_noise(I,
                                                       np.zeros(self.shape[-1]),
                                                       noise = noise_)
                f.flush()

class Fake_PV_Embeddings(torch.utils.data.Dataset):
    def __init__(self, dset, model, checkpoint_path, **kwargs):
        self.dset = dset
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.model.load_weights(self.checkpoint_path)
        self.h5_name = self.model.checkpoint_folder + '/embeddings.h5'
        self.device = self.model.encoder.device
        self.noise_levels = list(self.dset.h5_keys())
        self._noise = self.checkpoint_path.split('/')[-2]
        self.which = None
        
    # @property
    # def noise_(self): return self._noise
    # @noise_.setter
    # def noise_(self, i): 
    #     self._noise = self.dset.h5_keys()[i]
    #     self.model.load_weights(self.checkpoint_path)
    
    def open_h5(self): return h5py.File(self.h5_name, 'a')
    
    def h5_keys(self): return list(self.open_h5().keys())
    
    def write_embeddings(self, batch_size=100, overwrite=False):
        with self.open_h5() as f:
            if not overwrite:
                try: 
                    fits = f[f'{self.model.check}_fits']
                    return
                except: 
                    fits = f.create_dataset(f'{self.model.check}_fits', 
                                                shape=(len(self.dset), 
                                                        self.model.num_fits, 
                                                        self.dset.shape[-1]), 
                                                dtype=np.float32)
                    overwrite = True
                try: 
                    params = f[f'{self.model.check}_params']
                    return
                except: 
                    params = f.create_dataset(f'{self.model.check}_params', 
                                                shape=(len(self.dset), 
                                                        self.model.num_fits, 
                                                        self.model.num_params), 
                                                dtype=np.float32)
                    overwrite = True
        
            if overwrite:
                try: 
                    del f[f'{self.model.check}_fits']
                    fits = f.create_dataset(f'{self.model.check}_fits', 
                                                shape=(len(self.dset), 
                                                        self.model.num_fits, 
                                                        self.dset.shape[-1]), 
                                                dtype=np.float32)
                except: pass
                try: 
                    del f[f'{self.model.check}_params']
                    params = f.create_dataset(f'{self.model.check}_params', 
                                                shape=(len(self.dset), 
                                                        self.model.num_fits, 
                                                        self.model.num_params), 
                                                dtype=np.float32)
                except: pass
            

                self.model.configure_dataloader_sampler(sampler=None)
                self.model.configure_dataloader(batch_size=batch_size)
                
                for i, (idx, x) in enumerate(tqdm(self.model.dataloader, leave=True, total=len(self.model.dataloader), desc="Writing embeddings")):
                    with torch.no_grad():
                        value = x
                        batch_size = x.shape[0]
                        test_value = Variable(value)
                        test_value = test_value.float().to(self.device)
                        fits_, params_ = self.model.encoder(test_value)
                        
                        fits[i*batch_size:(i+1)*batch_size] = fits_.cpu().numpy()
                        params[i*batch_size:(i+1)*batch_size] = params_.cpu().numpy()
                    
    def __getitem__(self, idx):
        with self.open_h5() as f:
           return f[f'{self.model.check}_fits'][idx], f[f'{self.model.check}_params'][idx]
        
    def __len__(self):
        with self.open_h5() as f:
            return f[f'{self.model.check}_fits'].shape[0]
