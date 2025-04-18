import numpy as np
import torch
# from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import dask.array as da        
from tqdm import tqdm
import h5py 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from ..models.pv_fitter import Fitter_AE
import os
class Fake_PV_Dataset(torch.utils.data.Dataset):
    def __init__(self, scaled=False, shape=[100,100,500], save_folder='./', overwrite=False):
        '''dset is x*y,spec_len'''
        self.save_folder = save_folder
        self.h5_name = f'{self.save_folder}fake_pv_uniform.h5'
        self.fwhm, self.nu_ = 50, 0.7
        self.shape = shape
        self.spec_len = self.shape[-1]
        self.mask = np.ones((self.shape[0], self.shape[1])); self.mask[40:60,30:50] = 0; self.mask = self.mask.flatten()
        if overwrite: self.generate_pv_data()
        
        self.zero_dset = self.open_h5()[list(self.open_h5().keys())[0]][:]
        self.maxes = self.zero_dset.max(axis=-1).reshape(self.shape[:-1]+(1,))
        self.scale = scaled
        self.noise_levels = list(self.h5_keys())
        self._noise = self.noise_levels[0]
        
    @property
    def noise_(self): return self._noise
    @noise_.setter
    def noise_(self, i): self._noise = self.noise_levels[i]
        
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

    def scale_data(self, unscaled_data):
        self.scaler = Pipeline([('scaler', StandardScaler()), ('minmax', MinMaxScaler())])
        scaled_data = self.scaler.fit_transform(unscaled_data.reshape(-1, unscaled_data.shape[-1])).reshape(unscaled_data.shape)
        return scaled_data
    
    def unscale_data(self, unscaled_data, scaled_data):
        self.scaler.fit(unscaled_data.reshape(-1, unscaled_data.shape[-1]))
        unscaled_data = self.scaler.inverse_transform(scaled_data.reshape(-1, scaled_data.shape[-1])).reshape(scaled_data.shape)
        return unscaled_data

    @staticmethod
    def pv_area(I,w,nu): return I*w*np.pi/2/ ((1-nu)*(np.pi*np.log(2))**0.5 + nu)

    def __len__(self): return (self.shape[0]*self.shape[1])

    def __getitem__(self, idx):
        with self.open_h5() as f:
            try: data = np.array([f[self.noise_][i] for i in idx])
            except: data = f[self.noise_][idx]
            
            if self.scale: data = self.scale_data(data)
                
            return idx, data
    
    
    def open_h5(self): return h5py.File(self.h5_name, 'a')
    
    def h5_keys(self): return list(self.open_h5().keys())
    
    def unscale(self, data, idx): return data*torch.tensor(self.maxes[idx]).to(data.device)

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

class Fake_PV_Embeddings(Fake_PV_Dataset, Fitter_AE):
    def __init__(self, dset, model, checkpoints_folder, **kwargs):        
        self.__dict__.update(dset.__dict__)
        self.__dict__.update(model.__dict__)
        self.device = self.model.encoder.device
        self.h5_name = os.path.split(self.h5_name)[0] + 'embeddings.h5'
        self.noise_ = 0
        
    @property
    def noise_(self): return self._noise
    @noise_.setter
    def noise_(self, i): 
        self._noise = self.noise_levels[i]
        self.dset.noise_ = self._noise 
        self.model.load_weights(self.checkpoint_path)
        
    
        
    
    def open_h5(self): return h5py.File(self.h5_name, 'a')
    
    def h5_keys(self): return list(self.open_h5().keys())
    
    def write_embeddings(self, batch_size=100):
        with self.open_h5() as f:
            try: fits = f[f'{self.model.check}_fits']
            except: fits = f.create_dataset(f'{self.model.check}_fits', 
                                            shape=(self.dset.shape[0], 
                                                self.model.num_fits, 
                                                self.dset.shape[-1]), 
                                            dtype=np.float32)
            try: params = f[f'{self.model.check}_params']
            except: params = f.create_dataset(f'{self.model.check}_params', 
                                            shape=(self.dset.shape[0], 
                                                    self.model.num_fits, 
                                                    self.model.num_params), 
                                            dtype=np.float32)

            self.model.configure_dataloader_sampler(sampler=None)
            self.model.configure_dataloader(batch_size=batch_size)
            
            for i, (idx, x) in enumerate(tqdm(self.model.dataloader, leave=True, total=len(self.model.dataloader))):
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
            fits = f[f'{self.model.check}_fits'][idx]
            params = f[f'{self.model.check}_params'][idx]
            return idx, fits, params
        
    def __len__(self):
        with self.open_h5() as f:
            return f[f'{self.model.check}_fits'].shape[0]
