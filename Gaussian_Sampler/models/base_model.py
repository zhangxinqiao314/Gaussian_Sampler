import m3_learning
from m3_learning.nn.STEM_AE import STEM_AE
import numpy as np
import torch
from torch.utils.data import Sampler
from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from tqdm import tqdm
import torch.nn.functional as F

class Gaussian_Sampler(Sampler):
    def __init__(self, dset, batch_size, gaussian_std=5, num_neighbors=10):
        """
        Custom Gaussian Sampler for stacked EELS dataset with multiple datacubes of different sizes.

        Args:
            dataset_shapes (list of tuples): List of shapes of each datacube in the dataset, e.g., [(128, 128, 2, 969), (140, 140, 2, 969), ...].
            batch_size (int): Number of total points per minibatch.
            gaussian_std (int): Standard deviation for Gaussian sampling around the first sampled point.
            num_neighbors (int): Number of additional points to sample around the first point. ( best if batch_size % num_neighbors == 0)
        """
        self.dset = dset
        self.shape = dset.shape # (H, W, X, Y)
        self.batch_size = batch_size
        self.gaussian_std = gaussian_std
        self.num_neighbors = num_neighbors

    def _split_list(self, batch):
        split_batches = []
        for i in range(0, len(batch)-1, self.num_neighbors):
            split_batches.append(batch[i:i+self.num_neighbors])
        split_batches.append(batch[i+self.num_neighbors:])
        return split_batches

    def __iter__(self):
        """Return a batch of indices for each iteration."""
        self.batches = 0
        while self.batches < len(self)-1: # drop last
            batch = []

            while len(batch) < self.batch_size:
                ind = torch.randint(0, len(self.dset),(1,)).item()

                x, y = int(ind % self.shape[0]), int(ind / self.shape[0])  # find x,y coords
                
                neighbors = set()
                neighbors.add((x,y))
                # Get neighbors around the selected point in the H*W flattened image
                while len(neighbors) < self.num_neighbors:
                    # Sample a shift from a normal distribution, apply it within the H*W flattened space
                    x_ = int(torch.randn(1).item() * self.gaussian_std)
                    y_ = int(torch.randn(1).item() * self.gaussian_std)
                    new_x, new_y = x + x_, y + y_
                    if not (0 <= new_x < self.shape[0] and 0 <= new_y < self.shape[1]):
                        continue  # skip if the new coordinates are out of bounds
                    if (new_x, new_y) in neighbors:
                        continue  # skip if the new coordinates are already in neighbors
                    neighbors.add((new_x, new_y))

                batch += [coord[1]*self.shape[0] + coord[0] for coord in neighbors]
                if len(batch) >= self.batch_size: break
            self.batches += 1
            
            yield batch[:self.batch_size]

    def __len__(self):
        """Return the number of batches per epoch."""
        # This can be adjusted based on the desired number of iterations per epoch
        return len(self.dset) // self.batch_size


def custom_collate_fn(batch): 
    # batch = [tuple(map(torch.tensor, elem)) for elem in batch[0]]
    # idx,data = zip(*batch)
    return torch.tensor(batch[0][0]), torch.tensor(batch[0][1])


class Averaging_Loss_AE(STEM_AE.ConvAutoencoder):
    def __init__(self, sampler, sampler_kwargs, collate_fn, *args, **kwargs):
        super(Averaging_Loss_AE, self).__init__(*args, **kwargs)
        self.sampler = sampler(**sampler_kwargs)
        
    def loss_function(self,
                      train_iterator,
                      coef=0,
                      coef1=0,
                      coef2=0,
                      ln_parm=1,
                      binning=False,
                      weight_by_distance=False):
        """computes the loss function for the training

        Args:
            train_iterator (torch.Dataloader): dataloader for the training
            coef (float, optional): Ln hyperparameter. Defaults to 0.
            coef1 (float, optional): hyperparameter for contrastive loss. Defaults to 0.
            coef2 (float, optional): hyperparameter for divergence loss. Defaults to 0.
            ln_parm (float, optional): order of the regularization. Defaults to 1.
            beta (float, optional): beta value for VAE. Defaults to None.

        Returns:
            _type_: _description_
        """

        # set the train mode
        self.autoencoder.train()

        # loss of the epoch
        train_loss = 0
        con_l = ContrastiveLoss(coef1).to(self.device)

        for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):

            x = x.to(self.device, dtype=torch.float)
            maxi_ = DivergenceLoss(x.shape[0], coef2).to(self.device)
            
            
            # update the gradients to zero
            self.optimizer.zero_grad()

            embedding = self.encoder(x)
            reg_loss_1 = coef * torch.norm(embedding, ln_parm).to(self.device)/x.shape[0]
            if reg_loss_1 == 0: reg_loss_1 = 0.5
            predicted_x = self.decoder(embedding)
            contras_loss = con_l(embedding)
            maxi_loss = maxi_(embedding)
            
            # i, idx, x are lists. Each list is the gaussian samples from a batch
            if binning:
                # x = list(torch.split(x, self.dataloader_sampler.num_neighbors)) # Split the batch into groups based on the number of neighbors
                # predicted_x = list(torch.split(predicted_x, self.dataloader_sampler.num_neighbors))
                
                x = self.sampler._split_list(x) # Split the batch into groups based on the number of neighbors
                predicted_x = self.sampler._split_list(predicted_x)
                
                if weight_by_distance:
                    idx = self.sampler._split_list(idx, self.dataloader_sampler.num_neighbors) # Split the indices into groups based on the number of neighbors
                    weight_list = []
                    for i_, sample_group in enumerate(idx):
                        coords = [(ind % self.shape[0], int(ind /self.shape[0])) for ind in sample_group] # Calculate the coordinates relative to the first point in the group
                        weights = torch.tensor([1] +[1 /(1 +((coords[0][0]-coord[0])**2 +(coords[0][1]-coord[1])**2)**0.5) for coord in coords[1:]], # Calculate weights based on the Euclidean distance to the first point
                                                device = self.device) 
                        x[i_] = x[i_] *weights.unsqueeze(-1).unsqueeze(-1)
                        predicted_x[i_] = predicted_x[i_]*weights.unsqueeze(-1).unsqueeze(-1)  
                        weight_list.append(weights)
                    weight_sums = torch.stack([weight_.sum(dim=0) for weight_ in weight_list]).unsqueeze(-1).unsqueeze(-1) # Sum the weights in each group and stack them into a new batch
                    x = torch.stack([x_.sum(dim=0) for x_ in x])/weight_sums # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.sum(dim=0) for x_ in predicted_x])/weight_sums
                else:        
                    x = torch.stack([x_.mean(dim=0) for x_ in x]) # Sum the tensors in each group and stack them into a new batch
                    predicted_x = torch.stack([x_.mean(dim=0) for x_ in predicted_x])
                    # print('')

            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction='mean')
            
            loss = loss + reg_loss_1 + contras_loss - maxi_loss

            # backward pass
            train_loss += loss.item()
            loss.backward()
            # update the weights
            self.optimizer.step()

        return train_loss