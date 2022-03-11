import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda import amp

from tqdm import tqdm
import glob
import os

from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

bce_loss = nn.BCEWithLogitsLoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (loss0.data.item(), loss1.data.item(),
    #       loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net'  # 'u2netp'

data_dir = os.path.join("/home/ubuntu/transparent_shadow/data/")
tra_image_dir = os.path.join("image/")
tra_label_dir = os.path.join("label/")

image_ext = '.png'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 100
batch_size_train = 32
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
)
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train,
                               shuffle=True, num_workers=4, pin_memory=True)

# ------- 3. define model --------
# define the net
if(model_name == 'u2net'):
    net = U2NET(3, 1)
elif(model_name == 'u2netp'):
    net = U2NETP(3, 1)

# net.load_state_dict(torch.load(model_dir+model_name+'.pth'))

net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scaler = amp.GradScaler()

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 200  # save the model every 2000 iterations
print_frq = 10

for epoch in range(0, epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num += 1
        ite_num4val += 1

        inputs, labels = data['image'], data['mask']
        inputs_v, labels_v = inputs.cuda(), labels.cuda()

        # y zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)
        
        with amp.autocast():
            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # # print statistics
        running_loss += loss.data.detach_()
        running_tar_loss += loss2.data.detach_()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        if ite_num % print_frq == 0:
            print(f"[epoch: {epoch + 1}/{epoch_num}, batch: {(i + 1) * batch_size_train}/{train_num}, ite: {ite_num}] train loss: {running_loss.item() / ite_num4val}, tar: {running_tar_loss.item() / ite_num4val}")

        if ite_num % save_frq == 0:
            save_file = model_name + f"_bce_itr_{ite_num}_train_{running_loss.item() / ite_num4val}_tar_{running_tar_loss.item() / ite_num4val}.pth"
            print(f"Saving model to {save_file}")
            torch.save(net.state_dict(), model_dir + save_file)
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
