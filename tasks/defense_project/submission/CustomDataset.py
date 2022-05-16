import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, trainset_size):
        self.trainset_size = trainset_size
        self.inputs = []
        self.labels = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __add__(self, other):
        assert type(other) == tuple
        assert type(other[0]) == torch.tensor
        assert type(other[1]) == torch.tensor

        self.torch.cat(other[0])
        self.torch.cat(other[1])


    def append(self, other):
        if len(self.inputs) == self.trainset_size:
            return
        self.__add__(other)

