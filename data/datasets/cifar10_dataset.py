import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        if train:
            self.X = np.empty((0, 3, 32, 32))
            self.y = np.empty((0,))
            for i in range(1, 6):
                with open(os.path.join(base_folder, f"data_batch_{i}"), "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                    self.X = np.concatenate((self.X, data[b"data"].reshape(-1, 3, 32, 32)), axis=0)
                    self.y = np.concatenate((self.y, data[b"labels"]), axis=0)
        else:
            with open(os.path.join(base_folder, "test_batch"), "rb") as f:
                data = pickle.load(f, encoding='bytes')
                self.X = data[b"data"].reshape(-1, 3, 32, 32)
                # NOTE: !!! y here is list
                self.y = np.array(data[b"labels"])

        self.X = self.X.astype(np.float32) / 255
        self.transforms = [] if transforms is None else transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        for t in self.transforms:
            X = t(X)
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
