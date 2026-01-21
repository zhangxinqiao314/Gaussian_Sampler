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


class Poisson_Sampled_PV_Dataset(torch.utils.data.Dataset): #TODO: try loading scaler/param classes if it exists, TODO: getitem unscaled dataset
    def __init__(self, scaled=False, 
                 shape=[100,100,500], 
                 save_folder='./', 
                 overwrite=False, 
                 pv_fitter=None,
                 num_classes=5,
                 num_curves=3,
                 scaler='default',
                 norm_calculation=lambda x: np.linalg.norm(x,axis=-1, ord='max'),
                 dset_num = 0):
        '''dset is x*y,spec_len'''
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        self.pv_fitter = pv_fitter
        self.h5_name = f'{self.save_folder}_poisson_sampled_pv.h5'
        self._dset_name = f'{1:06.3f}_sample_rate'
        self.shape = shape
        self.spec_len = self.shape[-1]
        # set parameters for generating PV curves
        if overwrite:
            self.pv_param_classes = {
                'a': np.random.random_integers(0, 10, (num_classes, num_curves)),
                'E': np.random.random_integers(0, shape[-1], (num_classes, num_curves,)),
                'F': np.random.random_integers(1, shape[-1] // 2, (num_classes, num_curves,)),
                'nu': np.random.random((num_classes, num_curves,))
            }
            self.scaler = scaler
            self.norm_calculation = norm_calculation
            self.generate_pv_data()
        else: 
            self._read_pv_param_classes()
            self.dset_names = self.h5_keys()
            self._dset_index = 0
            if scaler == 'default': self._read_scaler()
            else: self.scaler = scaler
        
        self.zero_dset = self.getitem_zero_dset(range(self.shape[0]*self.shape[1]))[1]
        self.maxes = self.zero_dset.max(axis=-1).reshape(self.shape[:-1]+(1,))
        
    @property
    def dset_index(self): return self.dset_names[self._dset_index]
    @dset_index.setter
    def dset_index(self, i):
        self._dset_index = i 
    
    @property
    def dset_name(self): return self.dset_names[self._dset_index]    
    @dset_name.setter
    def dset_name(self, name):
        self._dset_index = self.dset_names.index(name)
            
    def low_signal(self,y, sample_rate=1, background_noise=0):
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
            self.scaler.set_params(**{'max': self.maxes})
        
    def scale_data(self, data): 
        if self.scaler is None: return data
        return self.scaler.transform(data)
        
    @staticmethod
    def pv_area(I,w,nu): return I*w*np.pi/2/ ((1-nu)*(np.pi*np.log(2))**0.5 + nu)
     
    def unscale_data(self, unscaled_data, scaled_data):
        self.scaler.fit(unscaled_data.reshape(-1, unscaled_data.shape[-1]))
        unscaled_data = self.scaler.inverse_transform(scaled_data.reshape(-1, scaled_data.shape[-1])).reshape(scaled_data.shape)
        return unscaled_data

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
            
            
            for i in tqdm(range(20)):
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
        for i in tqdm(range(20)):
            sample_rate = 5/(5+i)
            self.dset_name = f'{i:02d}_{sample_rate:06.3f}_sample_rate'
            sampled_data = self.low_signal(y=fit, sample_rate=sample_rate)
            self._write_unscaled_dataset(self.dset_name, sampled_data, fit.shape)
        self.dset_names = self.h5_keys()
        self._write_scaled_dataset()

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
                        data[...,i,j] = 0
                        
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
    
    def open_h5_file(self): return h5py.File(self.h5_name, 'a')
    
    def h5_keys(self): return list(self.open_h5_file().keys())
    
    def write_embeddings(self, batch_size=100, overwrite=False):
        with self.open_h5_file() as f:
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
        with self.open_h5_file() as f:
           return f[f'{self.model.check}_fits'][idx], f[f'{self.model.check}_params'][idx]
        
    def __len__(self):
        with self.open_h5_file() as f:
            return f[f'{self.model.check}_fits'].shape[0]

class NP_metadata(torch.utils.data.Dataset):
    def __init__(self, save_path, data_path,overwrite=False,**kwargs):
        self.save_path = save_path
        self.h5_name = f'{save_path}/nanoparticle_metadata.h5'
        
        if not os.path.exists(self.h5_name): h = h5py.File(self.h5_name,'w')
        else: h = h5py.File(self.h5_name,'r+')
        
class NP_EELS_Dataset(torch.utils.data.Dataset):
    def __init__(self, save_path, data_path,overwrite=False,**kwargs):
        self.save_path = save_path
        self.h5_name = f'{save_path}/nanoparticle_EELS_data.h5'
        
        if not os.path.exists(self.h5_name): h = h5py.File(self.h5_name,'w')
        else: h = h5py.File(self.h5_name,'r+')
        
# class NP_STEM_Dataset(torch.utils.data.Dataset):
#     """Class for the STEM dataset.
#     """

#     def __init__(self, save_path, data_path,overwrite=False,**kwargs):
#         """Initialization of the class.

#         Args:
#             save_path (string): path where the hyperspy file is located
#         """
#         self.save_path = save_path
#         self.h5_name = f'{save_path}/combined_data.h5'

#         # create and sort metadata 
#         self.meta = {}
#         path_list = glob.glob(f'{save_path}/*/*/*/SI data (*)/Diffraction SI.dm4')
#         def get_number(path):
#             return int(path.split('/')[-2].split(' ')[-1][1:-1])
#         def get_particle(path):
#             return path.split('/')[-3]
#         path_list.sort(key=get_number)
#         path_list.sort(key=get_particle)
#         self.meta['path_list'] = path_list
        
#         # create/ open h5 file
#         if not os.path.exists(self.h5_name): h = h5py.File(self.h5_name,'w')
#         else: h = h5py.File(self.h5_name,'r+')

#         print('fetching metadata...')
#         self.meta['particle_list'] = []
#         self.data_list = []
#         self.meta['shape_list'] = []
#         self.bad_files = []
#         self.meta['particle_inds'] = [0]
#         self.meta['sample_inds'] = [0]

#         # go through data files and fill metadata
#         for i,path in enumerate(tqdm(self.meta['path_list'])):
#             try:
#                 s = hs.load(path, lazy=True)
#                 self.data_list.append(s.data)
#                 self.meta['particle_list'].append(path.split('/')[-3] + path.split('/')[-2].split(' ')[-1])
#                 self.meta['shape_list'].append(s.data.shape)
#                 self.meta['particle_inds'].append(self.meta['particle_inds'][-1] + s.data.shape[0]*s.data.shape[1])
#                 if i>1 and self.meta['particle_list'][-1].split('(')[0] != self.meta['particle_list'][-2].split('(')[0]:
#                     self.meta['sample_inds'].append(i) # start of new sample
#                 # # print(path)
#             except:
#                 self.bad_files.append(path)
#                 self.meta['path_list'].remove(path)
#                 print('bad',path)
#         print(len(self.meta['shape_list']), 'valid samples')

#         self.shape = self.__len__(),128,128

#         # create h5 dataset, fille metadata, and transfer data from dm4 files to h5
#         if overwrite or 'processed_data' not in h:
#             if 'processed_data' in h: del h['processed_data']
#             print('writing processed_data h5 dataset')
#             h.create_dataset('processed_data',
#                               shape=(sum( [shp[0]*shp[1] for shp in self.meta['shape_list']] ),
#                                     128, 128),
#                               dtype=float)
            
#             for k,v in self.meta.items(): # write metadata
#                     h['processed_data'].attrs[k] = v

#             for i,data in enumerate(tqdm(self.data_list)): # fill data
#                 h['processed_data'][self.meta['particle_inds'][i]:self.meta['particle_inds'][i+1]] = \
#                     np.log(np.array(data.reshape((-1, 128,128))) + 1)    
#                     # da.log(data.reshape((-1, 128,128)) + 1) 

#         # scaling
#         print("fitting scaler...")
#         # sample = h['processed_data'][np.arange(0,self.__len__(),10000)]
#         self.scaler = StandardScaler()
#         self.scaler.fit( h['processed_data'][0:self.__len__():5000].reshape(-1,128*128) )

#         print('done')

#     def __len__(self):
#         return sum( [shp[0]*shp[1] for shp in self.meta['shape_list']] )
    
#     def __getitem__(self,index):
#         with h5py.File(self.h5_name, 'r+') as h5:
#             img = h5['processed_data'][index]
#             img = img.reshape(-1,128*128)
#             img = self.scaler.transform(img)
#             img = img.reshape(128,128)
#             mean = img.mean()
#             std = img.std()
#             mask = abs(img)<mean+std*5

#             # return img
#             return index,img*mask

#     def open_h5_file(self):
#         return h5py.File(self.h5_name, 'r+')

#     def view_log(self,index):
#         with h5py.File(self.h5_name, 'r+') as h5:
#             return h5['processed_data'][index]


#         # # Determine which dask array to access based on the index
#         # dask_array_index = index // (self.dask_arrays[0].shape[0] * self.dask_arrays[0].shape[1])
#         # dask_array_offset = index % (self.dask_arrays[0].shape[0] * self.dask_arrays[0].shape[1])

#         # # Load the diffraction pattern from dask array
#         # diffraction_pattern = self.dask_arrays[dask_array_index][dask_array_offset // self.dask_arrays[0].shape[1],
#         #                                                          dask_array_offset % self.dask_arrays[0].shape[1]]

#         # # preprocessing

#         # # Return the diffraction pattern as input and a dummy label (can be anything since we're not using it)
#         # return diffraction_pattern, torch.tensor(0)
    
#     # def crop(self,bbox):
#     #     (bx1,bx2,by1,by2)=bbox
#     #     h = h5py.File(self.h5_name,'r+')
#     #     del h['processed']
#     #     h.create_dataset('processed',
#     #                      data=np.log(h['raw_data'][bx1:bx2,by1,by2] + 1),
#     #                      dtype=float)
#     #     h.close()

#     def subtract_background(self,img,**kwargs):
#         return img - gaussian_filter(img,**kwargs)
    
#     def apply_scaler(self):
#         h = h5py.File(self.h5_name,'r+')
#         t,a,b,x,y = h['raw_data'].shape
#         data = h['processed'][:].T.reshape(x*y,-1)
#         print('standard scaling')
#         data = StandardScaler().fit_transform(data)
#         print('normalizing 0-1')
#         data -= data.min(axis=0)
#         data /= data.max(axis=0)
#         print('writing to h5')
#         h['processed'][:] = data.reshape(y,x,-1).T
#         h.close()
        
#     def apply_mask(self,bbox=None,center=None,radius=None):
#         """apply a mask in the shape of a circle. 
#         Arguments can either include a square around the brightfield or the center and radius

#         Args:
#             square (tuple, optional): (x1,x2,y1,y2) of bounding box. Defaults to None.
#             center (tuple, optional): (x,y) indices of center of mask. Defaults to None.
#             radius (int, optional): radius of mask. Defaults to None.
#         """        
#         h = h5py.File(self.h5_name,'r+')
#         print('Masking')
#         for sample,i in enumerate(tqdm()):
#             h['processed'][i]=data*mask+(-mask+1)*h['processed'][i].mean()
#         h.close()

#     def apply_threshold(self,thresh):
#         h = h5py.File(self.h5_name,'r+')
#         args = np.argwhere(h['processed']>thresh)
#         h['processed'][args] = thresh
#         h.close()

#     # # def preprocess(self,mask_center=False,crop=False,sub_bkg=False,thresh=False):
#     # #     # mask_center
#     # #     # try:
#     # #         (bx1,bx2,by1,by2)=mask_center
#     # #         center = int((bx1+bx2)/2),int((by1+by2)/2)
#     # #         r = max(abs(bx2-center[0])+1,abs(by2-center[1])+1)
#     # #         rr,cc = disk(center,r)

#     # #         # mask=np.zeros((self.data.shape[-2],self.data.shape[-1]))
#     # #         # mask[rr,cc] = 0
#     # #         # self.processed = da.map_blocks(lambda x: x[:,:,:, rr, cc] * 0.0, 
#     # #         #                                self.processed, 
#     # #         #                                dtype=self.processed.dtype,
#     # #         #                                chunks=self.processed.chunks)
#     # #         # self.processed=test
#     # #     #     for i,j in list(zip(rr,cc)):
#     # #     #             self.processed[:,:,:,i,j] = 0
#     # #     # # except: print('No mask')


