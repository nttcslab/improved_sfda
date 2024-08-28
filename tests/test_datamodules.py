from pathlib import Path

import pytest
import torch

from src.data.DA_datamodule import DADataModule
from src.data.components.office31 import Office31
from torchvision.transforms import transforms

@pytest.mark.parametrize("batch_size", [32, 128])
def test_da_datamodule(batch_size):
    data_dir = "data/"

    t = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dm = DADataModule(source_dataset=Office31(data_dir, "A", transform=t), target_dataset=Office31(data_dir, "W", transform=t), batch_size=batch_size)

    assert Path(data_dir, "office31").exists()
    assert Path(data_dir, "office31", "amazon").exists()
    assert Path(data_dir, "office31", "webcam").exists()
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()
    batch = next(iter(dm.train_dataloader()))
    x, y, idx = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert len(idx) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert idx.dtype == torch.int64
