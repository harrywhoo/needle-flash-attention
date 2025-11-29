import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.batch_idx = 0
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_idx >= len(self.ordering):
            raise StopIteration

        batch_indices = self.ordering[self.batch_idx]
        self.batch_idx += 1

        # Collect batch samples
        batch_samples = [self.dataset[int(i)] for i in batch_indices]

        # Determine structure - check first sample
        first_sample = batch_samples[0]

        if isinstance(first_sample, tuple):
            # Dataset returns tuples (e.g., NDArrayDataset, MNISTDataset)
            num_elements = len(first_sample)
            batches = []
            for i in range(num_elements):
                batch_data = np.array([sample[i] for sample in batch_samples])
                batches.append(Tensor(batch_data))
            return tuple(batches)
        else:
            # Single value returned
            batch_data = np.array(batch_samples)
            return Tensor(batch_data)
        ### END YOUR SOLUTION

