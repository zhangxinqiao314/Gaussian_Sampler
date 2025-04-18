# batch norm after whole blocks
from math import ceil
import matplotlib.pyplot as plt
import torch
plt.clf()
test_inds_= [10,923,2346,3456,5873,7889,8245,9357,9795]
# batch = next(iter(fitter.dataloader))
# print(batch[1].shape)

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
    fig.suptitle(f'with batch norm after whole blocks: Random sampler: noise {dset.noise_}, scaling kernel size {dset.scaling_kernel_size}')
    lines = []

    for i,ind in enumerate(test_inds):
        a = ax[i].plot(test_batch[i].cpu().detach().numpy().flatten(), label='input')
        if i==0: lines.append(a[0])
        for f in range(out[0].shape[1]):
            a = ax[i].plot(out[0][i,f].cpu().detach().numpy(),'-.', linewidth=0.5, label=f'fit {f}')
            if i==0: lines.append(a[0]) 
        a = ax[i].plot(out[0][i].sum(dim=0).cpu().detach().numpy(), 'k--', label='sum')
        if i==0: lines.append(a[0])

    for i,ind in enumerate(test_inds):
        a = ax[i].plot(dset.zero_dset[ind], label='0-noise')
        ax[i].set_ylim(0, 20)
        if i==0: lines.append(a[0])

    ax[0].legend(handles=lines, loc='upper right')

    plt.show()