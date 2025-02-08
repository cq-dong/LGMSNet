import os
import random
import cv2

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image

class PH2Dataset(Dataset) :
    def __init__(self, dataset_dir, mode, transform=None, target_transform=None):
        super(PH2Dataset, self).__init__()
        self.image_folder = 'images'
        self.label_folder = 'masks'
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))

        print(len(self.frame))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_folder, self.frame.image_path[idx])
        label_path = os.path.join(self.dataset_dir, self.label_folder, self.frame.mask_path[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transform(image)
            self._set_seed(seed); label = self.target_transform(label)

        label[label >= 0.5] = 1; label[label < 0.5] = 0

        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)




class PH2Datasetmy(Dataset) :
    def __init__(self, dataset_dir, mode, transform=None):
        super(PH2Datasetmy, self).__init__()
        self.image_folder = 'images'
        self.label_folder = 'masks'
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))

        print("self.frame",len(self.frame))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_folder, self.frame.image_path[idx])
        label_path = os.path.join(self.dataset_dir, self.label_folder, self.frame.mask_path[idx])


        image = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)[..., None]

        if self.transform:        
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)
        label[label >= 0.5] = 1; label[label < 0.5] = 0
        sample = {"image": image, "label": label}
        return sample


    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)