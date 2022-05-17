import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, trainset_size):
        self.trainset_size = trainset_size
        self.inputs = None
        self.labels = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __add__(self, other):
        if self.inputs is None:
            self.inputs = other[0]
            self.labels = other[1]
        else:
            self.inputs = torch.cat((self.inputs, other[0]))
            self.labels = torch.cat((self.labels, other[1]))

    def append(self, other):
        if self.inputs is not None and self.inputs.shape[0] == self.trainset_size:
            return
        self.__add__(other)

    def clear(self):
        self.inputs = None
        self.labels = None

