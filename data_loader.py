# data loader
import numpy as np
from torch.utils.data import Dataset
import cv2


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(self.label_name_list) > 0:
            label = cv2.imread(self.label_name_list[idx], cv2.IMREAD_GRAYSCALE)
        else:
            label = np.zeros_like(image[:,:,0])

        transformed = self.transform(image=image, mask=label)
        transformed['mask'] = transformed['mask'][None]/255.
        return transformed
