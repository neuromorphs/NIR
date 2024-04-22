import os
from typing import Callable, Optional, Tuple
import pytorch_lightning as pl

import numpy as np
import tonic
from tonic import (DiskCachedDataset, MemoryCachedDataset, SlicedDataset,
                   datasets, slicers, transforms)
from torch.utils.data import DataLoader


class NMNISTFrames(pl.LightningDataModule):
    """
    This dataset provides 3 frames for each sample in the original NMNIST dataset.
    The dataset length is 3*60000 for training and 3*10000 for testing set. 
    The frames are cached to disk in an efficient format.

    Parameters:
        save_to: str path where to save raw data to.
        batch_size: the dataloader batch size.
        augmentation: An optional callable that will be applied to each sample.
        cache_path: Where to store cached versions of all the frames.
        metadata_path: Store metadata about how recordings are sliced in individual samples.
                       Providing the path to store the metadata saves time when loading the dataset the next time.
        num_workers: the number of threads for the dataloader.
        precision: can be 16 for half or 32 for full precision.
    """

    def __init__(
        self,
        save_to: str,
        batch_size: int,
        augmentation: Optional[Callable] = None,
        cache_path: str = 'cache/frames',
        metadata_path: str = 'metadata/frames',
        num_workers: int = 6,
        precision: int = 32,
    ):
        super().__init__()
        self.save_to = save_to
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.cache_path = cache_path
        self.metadata_path = metadata_path
        self.num_workers = num_workers
        self.precision = precision

    def prepare_data(self):
        datasets.NMNIST(save_to=self.save_to, train=True)
        datasets.NMNIST(save_to=self.save_to, train=False)

    def get_train_or_testset(self, train: bool):
        dataset = datasets.NMNIST(save_to=self.save_to, train=train)
        
        slicer = slicers.SliceByTimeBins(3)
        image_transform = transforms.ToImage(sensor_size=dataset.sensor_size)

        dtype = {
            32: np.float32,
            16: np.float16,
        }

        sliced_dataset = SlicedDataset(
            dataset,
            slicer=slicer,
            metadata_path=os.path.join(self.metadata_path, f"train_{train}"),
            transform=lambda x: image_transform(x).astype(dtype[self.precision]),
        )

        return DiskCachedDataset(
            dataset=sliced_dataset,
            cache_path=os.path.join(self.cache_path, f"train_{train}", f"precision_{self.precision}"),
            transform=self.augmentation,
        )

    def setup(self, stage=None):
        self.train_data = self.get_train_or_testset(True)
        self.test_data = self.get_train_or_testset(False)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)
  
    def val_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return self.val_dataloader()


class NMNISTRaster(pl.LightningDataModule):
    """
    This dataset provides the original NMNIST samples as rasters
    and caches them to disk.

    Parameters:
        save_to: str path where to save raw data to.
        batch_size: The batch size.
        n_time_bins: How many time bins per sample.
        augmentation: An optional callable that will be applied to each sample.
        cache_path: Where to store cached versions of all the frames.
        num_workers: the number of threads for the dataloader.
        precision: can be 16 for half or 32 for full precision.
    """

    def __init__(
        self,
        save_to: str,
        batch_size: int,
        n_time_bins: int,
        augmentation: Optional[Callable] = None,
        cache_path: str = 'cache/rasters',
        num_workers: int = 6,
        precision: int = 32,
    ):
        super().__init__()
        self.save_to = save_to
        self.batch_size = batch_size
        self.n_time_bins = n_time_bins
        self.augmentation = augmentation
        self.cache_path = cache_path
        self.num_workers = num_workers
        self.precision = precision

    def prepare_data(self):
        datasets.NMNIST(save_to=self.save_to, train=True)
        datasets.NMNIST(save_to=self.save_to, train=False)

    def get_train_or_testset(self, train: bool):
        frame_transform = transforms.ToFrame(sensor_size=datasets.NMNIST.sensor_size, n_time_bins=self.n_time_bins)

        dtype = {
            32: np.float32,
            16: np.float16,
        }

        dataset = datasets.NMNIST(
            save_to=self.save_to,
            train=train,
            transform=lambda x: frame_transform(x).astype(dtype[self.precision]),
        )
        
        return DiskCachedDataset(
            dataset=dataset,
            cache_path=os.path.join(self.cache_path, f"train_{train}", f"precision_{self.precision}"),
            transform=self.augmentation,
        )

    def setup(self, stage=None):
        self.train_data = self.get_train_or_testset(True)
        self.test_data = self.get_train_or_testset(False)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)#, collate_fn=tonic.collation.PadTensors())
  
    def val_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.batch_size)#, collate_fn=tonic.collation.PadTensors())

    def test_dataloader(self):
        return self.val_dataloader()

