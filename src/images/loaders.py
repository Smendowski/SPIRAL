import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_image_loaders(
    base_dir: str,
    batch_size: int = 64,
    image_size: int = 64,
    seed: int = 42,
    val_split: float = 0.2,
):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = datasets.ImageFolder(Path(base_dir) / "train", transform=tf)
    test_ds = datasets.ImageFolder(Path(base_dir) / "test", transform=tf)

    train_size = int((1 - val_split) * len(train_ds))
    val_size = len(train_ds) - train_size
    train_subset, val_subset = random_split(
        train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    num_workers = 4

    return (
        DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
        ),
        DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
        ),
    )
