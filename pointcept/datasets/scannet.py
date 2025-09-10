"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        data_dict = {}
        coords = []
        colors = []
        # normals = []
        distances = []

        with open(data_path, "r") as f:
            for line in f:
                if line.startswith("//") or not line.strip():
                    continue
                parts = line.strip().split()
                # if len(parts) < 9:
                #     continue
                x, y, z = map(float, parts[0:3])
                r, g, b = map(int, parts[3:6])
                # rs, gs, bs = map(float, parts[6:9])
                distance = float(parts[6])

                if distance <= 0 or distance > 5 or np.isnan(distance):
                    continue
                coords.append([x, y, z])
                colors.append([r, g, b])
                # normals.append([rs, gs, bs])
                distances.append(distance)

        coords = np.array(coords, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32) / 255.0
        # normals = np.array(normals, dtype=np.float32)
        distances = np.array(distances, dtype=np.float32)

        data_dict["coord"] = coords
        data_dict["color"] = colors
        # data_dict["normal"] = normals
        data_dict["name"] = name
        data_dict["split"] = split
        data_dict["segment"] = distances
        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)
