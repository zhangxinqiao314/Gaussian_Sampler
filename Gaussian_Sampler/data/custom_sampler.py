import numpy as np
import torch
from torch.utils.data import Sampler

class Gaussian_Sampler(Sampler):
    def __init__(self, dset, orig_shape, batch_size, gaussian_std=5, num_neighbors=10):
        """
        Custom Gaussian Sampler for stacked EELS dataset with multiple datacubes of different sizes.

        Args:
            dataset_shapes (list of tuples): List of shapes of each datacube in the dataset, e.g., [(128, 128, 2, 969), (140, 140, 2, 969), ...].
            batch_size (int): Number of total points per minibatch.
            gaussian_std (int): Standard deviation for Gaussian sampling around the first sampled point.
            num_neighbors (int): Number of additional points to sample around the first point. ( best if batch_size % num_neighbors == 0)
        """
        self.dset = dset
        self.shape = orig_shape # (H, W, X, Y)
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
    return torch.tensor(batch[0])
