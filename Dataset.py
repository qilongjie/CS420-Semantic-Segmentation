import os
import glob
import cv2
import numpy as np
import torch


class ISBIDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, labels_dir, flip=False):
        self.imgs = sorted(glob.glob(os.path.join(imgs_dir,'*.png')))
        self.labels = sorted(glob.glob(os.path.join(labels_dir,'*.png')))
        self.flip = flip

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        x = cv2.imread(self.imgs[item], 0) / 255
        y = cv2.imread(self.labels[item], 0) / 255

        img = x[:, :, np.newaxis]
        label = y[:, :, np.newaxis]

        if self.flip:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                label = np.fliplr(label)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
                label = np.flipud(label)

        img = img.transpose(2, 0, 1).astype(np.float32)
        label = label.transpose(2, 0, 1).astype(np.float32)

        ret = {
            'img': torch.from_numpy(img),
            'label': torch.from_numpy(label),
        }
        return ret

