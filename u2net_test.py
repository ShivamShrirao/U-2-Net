import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from torch.utils.data import DataLoader

from tqdm import tqdm
import glob
import os
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp

    data_dir = "/home/ubuntu/transparent_shadow/data/"
    image_dir = os.path.join(data_dir, "image/")
    prediction_dir = os.path.join(data_dir, "output/")

    # save results to test_results folder
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    img_name_list = glob.glob(image_dir + os.sep + '*')

    batch_size = 32
    
    transform = A.Compose([
            A.Resize(320, 320),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transform
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4, pin_memory=True)

    # --------- 3. model define ---------
    if(model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif(model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    net.load_state_dict(torch.load(model_dir+"u2net_bce_itr_88000_train_0.59157470703125_tar_0.08326338195800781.pth"))
    net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])
        with torch.inference_mode(), amp.autocast():
            inputs_test = data_test['image'].cuda()
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_test)

            pred = torch.sigmoid_(d0.detach_()[:,0]) * 255
            pred = pred.type(torch.uint8).cpu().numpy()
        
        for i,p in enumerate(pred):
            cv2.imwrite(prediction_dir + img_name_list[i_test*batch_size + i].split(os.sep)[-1], p)

        del d0, d1, d2, d3, d4, d5, d6

if __name__ == "__main__":
    main()
