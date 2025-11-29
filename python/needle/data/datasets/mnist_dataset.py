from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        import gzip
        import struct

        # Parse images
        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols, 1)

        # Parse labels
        with gzip.open(label_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        self.images = images
        self.labels = labels
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index].astype(np.float32) / 255.0
        label = self.labels[index]

        if self.transforms:
            for transform in self.transforms:
                img = transform(img)

        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION