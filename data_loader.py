# data loader
from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = A.Compose([
            A.Resize(360, 360),
            A.RandomCrop(width=320, height=320),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_name_list[idx], cv2.IMREAD_GRAYSCALE)

        transformed = self.transform(image=image, mask=label)
        transformed['mask'] = transformed['mask'][None]/255.
        return transformed
