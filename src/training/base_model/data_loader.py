import numpy as np
import torch


class DataLoader:
    def __init__(self, config):
        self.config = config
        # Mapping of binary files without loading into RAM
        self.train_data = np.memmap(config.train_bin_path, dtype=np.uint32, mode="r")
        self.val_data = np.memmap(config.val_bin_path, dtype=np.uint32, mode="r")

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data

        # Generate random indices within the mapped file
        ix = torch.randint(
            len(data) - self.config.block_size, (self.config.batch_size,)
        )

        # Extract sequences and convert them to PyTorch tensors
        x = torch.stack(
            [
                torch.from_numpy(
                    (data[i : i + self.config.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + self.config.block_size + 1]).astype(np.int64)
                )
                for i in ix
            ]
        )

        if self.config.device == "cuda":
            # Asynchronous transfer to the GPU
            x, y = x.pin_memory().to(
                self.config.device, non_blocking=True
            ), y.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)

        return x, y
