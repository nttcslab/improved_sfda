from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.data.components.imagelist import ImageList


class TgtTrainDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset: ImageList,
        train_transforms: transforms,
        test_transforms: transforms,
        strong_transforms: transforms = None,
        M: int = 1,
        K: int = 0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        root = dataset.root
        data_list = dataset.samples
        classes = dataset.classes

        self.train_dataset = ImageList(root=root, classes=classes, data_list=data_list, transform=train_transforms, transform_t=strong_transforms, M=M, K=K)
        self.val_dataset = ImageList(root=root, classes=classes, data_list=data_list, transform=test_transforms)
        self.test_dataset = dataset
        self.test_dataset.transform = test_transforms

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @property
    def num_classes(self) -> int:
        return self.train_dataset.num_classes

    @property
    def trainloader_size(self) -> int:
        return len(self.train_dataset) // self.batch_size
    
    def __len__(self) -> int:
        return len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )