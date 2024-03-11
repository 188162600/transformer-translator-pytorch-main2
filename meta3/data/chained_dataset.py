import torch
class ChainedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.len = sum(len(d) for d in datasets)

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                #print("index",index,dataset[index])
                return dataset[index]
            index -= len(dataset)
        raise IndexError("Index out of range")

    def __len__(self):
        return self.len