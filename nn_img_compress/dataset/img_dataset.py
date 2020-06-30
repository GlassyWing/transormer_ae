import time

import torch
from torch.utils.data import Dataset
from glob import glob
import os
import cv2
import numpy as np


class ImageFolderDataset(Dataset):

    def __init__(self, image_dir, n_visible=350, n_inference=450, is_train=True, split=0.8, is_test=False):
        self.img_paths = glob(os.path.join(image_dir, "*.jpg"))

        bp = int(len(self.img_paths) * split)
        if is_train:
            self.img_paths = self.img_paths[:bp]
        else:
            self.img_paths = self.img_paths[bp:]
        self.n_visible = n_visible
        self.n_inference = n_inference
        self.is_test = is_test

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx]) / 255
        h, w, c = image.shape
        v_grid_x = np.random.randint(0, w, self.n_visible)
        v_grid_y = np.random.randint(0, h, self.n_visible)

        v_value = image[v_grid_y, v_grid_x, :].reshape(-1, 3)

        v_grid_x = v_grid_x.astype(np.float) / w
        v_grid_y = v_grid_y.astype(np.float) / h
        v_grid = np.stack([v_grid_y, v_grid_x], 1)

        i_grid_x = np.random.randint(0, w, self.n_inference)
        i_grid_y = np.random.randint(0, h, self.n_inference)

        if self.is_test:
            i_grid_x, i_grid_y = np.meshgrid(np.arange(w), np.arange(h))
            i_grid_x = i_grid_x.flatten()
            i_grid_y = i_grid_y.flatten()

        i_value = image[i_grid_y, i_grid_x, :].reshape(-1, 3)
        i_grid_x = i_grid_x.astype(np.float) / w
        i_grid_y = i_grid_y.astype(np.float) / h
        i_grid = np.stack([i_grid_y, i_grid_x], 1)

        return torch.tensor(v_grid, dtype=torch.float32), \
               torch.tensor(v_value, dtype=torch.float32), \
               torch.tensor(i_grid, dtype=torch.float32), \
               torch.tensor(i_value, dtype=torch.float32), \
               h, w, idx

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = ImageFolderDataset("E:/data/coco/val2017", is_train=False, is_test=False, n_inference=32000)
    v_grid, v_value, i_grid, i_value, h, w, idx = ds[50]
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[(i_grid[:, 0] * h).round().int(), (i_grid[:, 1] * w).round().int()] = (i_value * 255).numpy().astype(np.uint8)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
