# Reference: https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
import torch
from torch.utils.data import TensorDataset, DataLoader


class PTDataFactory(object):
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager

    def get_data_as_loader(self, sequences, labels):
        tensor_x = torch.from_numpy(sequences)
        tensor_y = torch.from_numpy(labels)

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset)
        return dataloader

    def get_train_dataloader(self):
        return self.get_data_as_loader(self.dataset_manager["train_sequences"], self.dataset_manager["train_labels"])

    def get_valid_dataloader(self):
        return self.get_data_as_loader(self.dataset_manager["valid_sequences"], self.dataset_manager["valid_labels"])

    def get_test_dataloader(self):
        return self.get_data_as_loader(self.dataset_manager["test_sequences"], self.dataset_manager["test_labels"])
