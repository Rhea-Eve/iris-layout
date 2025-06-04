import os.path
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset

import cv2
import torch

class Iris(VisionDataset):
    """IRIS Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        self.data: Any = []
        self.targets = []

        train_list = ['housekeeping-poly.pkl', 'silife-poly.pkl']
        test_list = ['wrapped_snn_network-poly.pkl']

        if self.train:
            file_list = train_list
        else:
            file_list = test_list

        # now load the picked numpy arrays
        for file_name in file_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                print("Loading data...")
                print(f"Loading {file_name}...")
                entry = pickle.load(f, encoding="latin1")
                print("Finished loading.")
                print(f"{len(file_list)} files to load.\n")
                for d in entry["data"]:
                    self.data.append(d)
                if False: # manual check data input
                    check = cv2.vconcat(entry["data"][:32])
                    cv2.imshow('check', check)
                    print(f'{entry["labels"][:32]}')
                    cv2.waitKey(0)
                for l in entry["labels"]:
                    self.targets.extend([l])

        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 64)
        #self.data = self.data.transpose((0, 3, 2, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, 'housekeeping-poly.meta')
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data['label_names']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #cv2.imshow("data", img)
        #cv2.waitKey(0)
        img = Image.fromarray(img).convert("L")  # Convert to grayscale
        #img = Image.fromarray(img) #for rgb

        # Get original image size. NOTE: this may need to be changed?
        width, height = img.size
        image_size = torch.tensor([width, height], dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, image_size

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

