# batch norm after whole blocks
from math import ceil
import matplotlib.pyplot as plt
import torch
from m3util.viz.colorbars import add_colorbar

plt.clf()
test_inds_= [10,923,2346,3456,5873,7889,8245,9357,9795]
# batch = next(iter(fitter.dataloader))
# print(batch[1].shape)

def_db_tensor_imshow(data):
    plt.imshow()

def plot_batch(fitter, dset, test_inds=test_inds_, noise=None):
    fitter.encoder.eval()
    if noise is None:
        dset.noise_ = fitter.checkpoint_folder.split('/')[-1]
    else:
        dset.noise_ = noise
    test_batch = torch.Tensor(dset[test_inds][1]).reshape(-1,1,dset.spec_len).float().to('cuda:0')
    out = fitter.encoder(test_batch.to('cuda:0'))
    # print(out[0].shape)
    px = 0
    fig, ax = plt.subplots(int(len(test_inds)**0.5),ceil(len(test_inds)/len(test_inds)**0.5),figsize=(10,10))
    try: ax=ax.flatten()
    except: ax=[ax]
    fig.suptitle(f'with batch norm after whole blocks: Random sampler: noise {dset.noise_}' )
                #  f',scaling kernel size {dset.scaling_kernel_size}')
    lines = []

    for i,ind in enumerate(test_inds):
        a = ax[i].plot(test_batch[i].cpu().detach().numpy().flatten(), label='input')
        if i==0: lines.append(a[0])
        for f in range(out[0].shape[1]):
            a = ax[i].plot(out[0][i,f].cpu().detach().numpy(),'-.', linewidth=1, label=f'fit {f}')
            if i==0: lines.append(a[0]) 
        a = ax[i].plot(out[0][i].sum(dim=0).cpu().detach().numpy(), 'k--', label='sum')
        if i==0: lines.append(a[0])

    for i,ind in enumerate(test_inds):
        a = ax[i].plot(dset.zero_dset[ind], label='0-noise')
        ax[i].set_ylim(0, 20)
        if i==0: lines.append(a[0])

    ax[0].legend(handles=lines, loc='upper right')

    plt.show()
    
    
def training_viz(x,y,s,dset,fitter, suptitle_label='', save=False):
    plt.rcParams['image.origin'] = 'lower'

    # plot the spectrum, image, and mean image
    fig, ax = plt.subplots(3,2, figsize=(6,8))
    fig.suptitle(f'params at {x,y}, {suptitle_label}')
    i = x*dset.shape[1]+y
    
    ax[0,0].plot(dset[i][1])
    ax[0,0].axvline(s)
    ax[0,0].set_title(f'spec at {x,y}')

    im = ax[1,0].imshow(dset[:,s][1].reshape(dset.shape[0],dset.shape[1]))
    ax[1,0].scatter(x,y,color='red')
    ax[1,0].set_title(f'im at {s}')
    add_colorbar(im, ax[1,0])
    
    im = ax[2,0].imshow(dset[:][1].mean(axis=1).reshape(dset.shape[0],dset.shape[1]))
    ax[2,0].scatter(x,y,color='red')
    ax[2,0].set_title(f'mean im')
    add_colorbar(im, ax[2,0])
        
 
    fit, params = fitter.encoder(torch.Tensor(dset[:][1].reshape(dset.shape[0]*dset.shape[1], 1, -1)).to(fitter.device))
    fit = fit.cpu().detach().numpy()
    params = params.cpu().detach().numpy()
    print(fit[i].sum(axis=1).shape)
    for f in range(4):
        ax[0,1].plot(fit[i,f], linestyle=':', linewidth=1)
    
    ax[0,1].plot(fit[i].sum(axis=0))
    ax[0,1].axvline(s)
    ax[0,1].set_title(f'fits at {x,y}')

    im = ax[1,1].imshow(fit[...,s].sum(axis=1).reshape(dset.shape[0],dset.shape[1]))
    ax[1,1].scatter(x,y,color='red')
    ax[1,1].set_title(f'fitted im at {s}')
    add_colorbar(im, ax[1,1])
    
    im = ax[2,1].imshow(fit[...,s].mean(axis=1).reshape(dset.shape[0],dset.shape[1]))
    ax[2,1].scatter(x,y,color='red')
    ax[2,1].set_title(f'mean fitted im at {s}')
    add_colorbar(im, ax[2,1])
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    if save: plt.savefig(f'spec_({x,y})_'+suptitle_label.replace(' ','_')+'.png')
    
    # plot the parameters
    param_labels = ['a', 'x', 'w', '$\mu$']
    fig, ax = plt.subplots(fitter.num_fits, fitter.num_params, 
                           figsize=((fitter.num_params+1)*2, (fitter.num_fits)*2))
    fig.suptitle(f'params at {x,y}, {suptitle_label}')
    for f in range(fitter.num_fits):
        for p in range(fitter.num_params):
            im = ax[f,p].imshow(params[:,f,p].reshape(dset.shape[0],dset.shape[1]))
            ax[f,p].scatter(x,y,color='red')
            ax[f,p].set_title(param_labels[p])
            add_colorbar(im, ax[f,p])

    plt.tight_layout()
    plt.show()
    if save: plt.savefig(f'params_({x,y})_'+suptitle_label.replace(' ','_')+'.png')
    
    
    plt.rcParams['image.origin'] = 'upper'