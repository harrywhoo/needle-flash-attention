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
        self.transforms = transforms

        # Load data from batch files
        if train:
            data = []
            labels = []
            for i in range(1, 6):
                fp = f"{base_folder}/data_batch_{i}"
                with open(fp, 'rb') as fo:
                    curr_batch = pickle.load(fo, encoding='bytes')
                    data.append(curr_batch[b'data'])
                    labels.append(curr_batch[b'labels'])
            self.X = np.concatenate(data, axis=0)
            self.y = np.concatenate(labels, axis=0)
        else:
            fp = f"{base_folder}/test_batch"
            with open(fp, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                self.X = batch[b'data']
                self.y = np.array(batch[b'labels'])

        # Divide pixel values by 255 
        self.X = self.X.astype(np.float32) / 255.0

        # Reshape from (N, 3072) to (N, 3, 32, 32)
        self.X = self.X.reshape(-1, 3, 32, 32)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image = self.X[index]
        label = self.y[index]

        if self.transforms:
            for transform in self.transforms:
                image = transform(image)

        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
