import torch.utils.data as Data

class DataLoader(Data.Dataset):
    def __init__(self, batch_size, is_cuda):
        self.batch_size = batch_size
        self.is_cuda = is_cuda

    #
    # def __len__(self):
    #
    # def __getitem__(self, index):