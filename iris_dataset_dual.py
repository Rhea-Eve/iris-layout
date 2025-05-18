import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset


class IrisDualView(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if self.train:
            file_bases = ['housekeeping-poly', 'silife-poly']
        else:
            file_bases = ['wrapped_snn_network-poly']

        self.data_0 = []
        self.data_90 = []
        self.targets = []

        for base in file_bases:
            with open(os.path.join(root, f"{base}.pkl"), "rb") as f:
                entry_0 = pickle.load(f, encoding="latin1")
                self.data_0.extend(entry_0["data"])
                self.targets.extend(entry_0["labels"])

        with open(os.path.join(root, f"{base}_psi90.pkl"), "rb") as f:
            entry_90 = pickle.load(f, encoding="latin1")
            self.data_90.extend(entry_90["data"])

        # Sanity check
        assert len(self.data_0) == len(self.data_90) == len(self.targets), "Mismatched dataset lengths"

        # Load class labels from any of the meta files (they're the same)
        meta_file = os.path.join(root, f"{file_bases[0]}.meta")
        with open(meta_file, "rb") as meta_f:
            meta = pickle.load(meta_f, encoding="latin1")
            self.classes = meta["label_names"]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}


        self.train = train
        self.data_0: Any = []
        self.data_90: Any = [] #adding the matxhing 90 degree data set
        self.targets = []


        # NOTE: Change these filenames to be more xlear in general!

        if self.train:
            base = "housekeeping-poly" # Training set
        else:
            base = "wrapped_snn_network-poly" # Testing set (inference)

        # Load 0° data
        with open(os.path.join(root, f"{base}.pkl"), "rb") as f:
            entry_0 = pickle.load(f, encoding="latin1")
            self.data_0 = entry_0["data"]
            self.targets = entry_0["labels"]

        # Load 90° data
        with open(os.path.join(root, f"{base}_psi90.pkl"), "rb") as f:
            entry_90 = pickle.load(f, encoding="latin1")
            self.data_90 = entry_90["data"]

        # Quick sanity check to make sure everything is lined up 1:1
        assert len(self.data_0) == len(self.data_90) == len(self.targets), "Mismatched dataset lengths" 

        # Load meta info
        with open(os.path.join(root, f"{base}.meta"), "rb") as meta_f:
            meta = pickle.load(meta_f, encoding="latin1")
            self.classes = meta["label_names"]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img0 = Image.fromarray(self.data_0[index]).convert("L")
        img90 = Image.fromarray(self.data_90[index]).convert("L")
        label = self.targets[index]

        width, height = img0.size
        image_size = torch.tensor([width, height], dtype=torch.float32)

        if self.transform:
            img0 = self.transform(img0)
            img90 = self.transform(img90)

        # Stack into shape [2, H, W], instead of [1, H, W]
        stacked = torch.cat([img0, img90], dim=0)

        if self.target_transform:
            label = self.target_transform(label)

        return stacked, label, image_size

    def __len__(self):
        return len(self.targets)

    def extra_repr(self):
        return "Split: Train" if self.train else "Split: Test"
