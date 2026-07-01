import os
import numpy as np
import torch


class SignalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        is_validate=False,
        read_limit=None,
        train_ratio=0.99,
        seed=42,
    ):
        super().__init__()
        self.data_path = data_path
        self.signal_raw = np.load(
            os.path.join(data_path, 'signal_raw.npy'), mmap_mode='r',
        )
        total_cnt = len(self.signal_raw)
        train_cnt = int(total_cnt * train_ratio)
        valid_cnt = total_cnt - train_cnt

        _indices = np.arange(total_cnt)
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(_indices)

        if is_validate:
            if valid_cnt <= 0:
                raise RuntimeError('validation set is empty')
            self.indices = _indices[-valid_cnt:]
            self.dataset_size = valid_cnt
        else:
            if train_cnt <= 0:
                raise RuntimeError('train set is empty')
            self.indices = _indices[:train_cnt]
            self.dataset_size = train_cnt

        if read_limit is not None:
            self.dataset_size = min(read_limit, self.dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.signal_raw[idx].astype(np.int64)
