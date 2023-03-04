import numpy as np
from torch.utils.data import DataLoader

from data.kitti_dataset import KITTIDataset


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_train_loader(cfg, split, num_workers=4):
    if cfg['type'] == 'KITTI':
        dataset = KITTIDataset(cfg, split)
    else:
        raise NotImplementedError

    return DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=dataset.collate_batch,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )


def build_test_loader(cfg, split, num_workers=4):
    if cfg['type'] == 'KITTI':
        dataset = KITTIDataset(cfg, split, is_training=False, augment_data=False)
    else:
        raise NotImplementedError

    return DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=dataset.collate_batch,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
