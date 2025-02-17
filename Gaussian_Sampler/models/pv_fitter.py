


class Fitter_AE(nn.module):
    def __init__(function=self.function,
                                 x_data = self.dset,
                                 input_channels=self.dset.shape[1],
                                 num_params=self.num_params,
                                 num_fits=self.num_fits,
                                 limits=self.limits,
                                 device='cuda:0',
                                 flatten_from = 1,)