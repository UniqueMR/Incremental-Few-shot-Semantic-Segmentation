import os
import os.path as osp
import tqdm
import pickle
import time
import random
import argparse
from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataloader.train_loader import Dataset as Dataset_base_train
from dataloader.val_loader import Dataset as Dataset_base_val
from models.models import CaNet, CaNet_EAUS, EHNet
from models.backbone import Bottleneck, ResNet
from tensorboardX import SummaryWriter

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='gpu nums')
parser.add_argument('--model', type=str, default='CaNet', help='model name')

args = parser.parse_args()

# settings
train_fold = [0] #training fold
val_fold = [0] #validation fold

# set device
gpu_list = [i for i in range(args.gpu)]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in gpu_list)

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.cuda.init()
cudnn.enabled = True

# load memory
memory_path = 'checkpoint/pretrained/memory.pkl'
print('loading memory from {}'.format(memory_path))
with open(memory_path, 'rb') as f:
    memory = pickle.load(f)

# set dataloader
train_set = Dataset_base_train(data_dir=data_dir, fold=train_fold, input_size=input_size, normalize_mean=IMG_MEAN,
                  normalize_std=IMG_STD,prob=prob)
trainloader = data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=4)

val_set = Dataset_base_val(data_dir=data_dir, fold=val_fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD)
valloader = data.DataLoader(val_set, batch_size=batch_size_val, shuffle=False, num_workers=4, drop_last=False)

# set model
# model_path = 'checkpoint/pretrained/model/best.pth'
# print('loading model from {}'.format(model_path))
print('model name: {}'.format(args.model))
if args.model == 'CaNet':
    model = CaNet(Bottleneck,[3, 4, 6, 3], 2)
elif args.model == 'CaNet_EAUS':
    model = CaNet_EAUS(Bottleneck,[3, 4, 6, 3], 2, memory_pool=memory)
elif args.model == 'EHNet':
    model = EHNet(Bottleneck,[3, 4, 6, 3], 2, memory_pool=memory)
model = load_resnet50_param(model=model, stop_layer='layer4')
turn_off(model)

# set optimizer
optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': 10 * learning_rate}],
                          lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# set result saver
checkpoint_dir = 'checkpoint/stage0/' + args.model
check_dir(checkpoint_dir)
writer = SummaryWriter(checkpoint_dir)
save_pred_every =len(trainloader)
loss_list, iou_list = [], []
highest_iou, tempory_loss, best_epoch = 0, 0, 0

# start base training
model = nn.DataParallel(model, device_ids=gpu_list)
# state_dict = torch.load(model_path)
# model.load_state_dict(state_dict, strict=False)
model = model.cuda()
model.train()
for epoch in range(num_epoch):
    begin_time = time.time()
    tqdm_train = tqdm.tqdm(trainloader)

    # training loop
    for i_iter, batch in enumerate(tqdm_train):
        query_rgb, query_mask,support_rgb, support_mask,history_mask,sample_class,index= batch
        query_rgb, support_rgb, support_mask, query_mask, history_mask = (query_rgb).cuda(0), (support_rgb).cuda(0), (support_mask).cuda(0), (query_mask).cuda(0).long(), (history_mask).cuda(0)
        query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use

        optimizer.zero_grad()

        # pred = model(query_rgb, support_rgb, support_mask, history_mask)
        if args.model == 'CaNet':
            pred = model(query_rgb, support_rgb, support_mask, history_mask)
        elif args.model == 'CaNet_EAUS':
            pred = model(query_rgb, support_rgb, support_mask, history_mask, sample_class)
        elif args.model == 'EHNet':
            pred = model(query_rgb, support_rgb, support_mask, history_mask, sample_class)

        pred_softmax = F.softmax(pred, dim=1).data.cpu()

        # update history mask
        for j in range (support_mask.shape[0]):
            sub_index=index[j]
            train_set.history_mask_list[sub_index]=pred_softmax[j]

        pred = nn.functional.interpolate(pred,size=input_size, mode='bilinear',align_corners=True)#upsample

        loss = loss_calc_v1(pred, query_mask, 0)
        loss.backward()
        optimizer.step()

        # track training loss
        tqdm_train.set_description('e:%d loss = %.4f-:%.4f' % (
        epoch, loss.item(),highest_iou))

        #save training loss
        writer.add_scalar('train_loss', loss.item(), epoch * len(trainloader) + i_iter)
        tempory_loss += loss.item()
        if i_iter % save_pred_every == 0 and i_iter != 0:
            loss_list.append(tempory_loss / save_pred_every)
            plot_loss(checkpoint_dir, loss_list, save_pred_every)
            np.savetxt(os.path.join(checkpoint_dir, 'loss_history.txt'), np.array(loss_list))
            tempory_loss = 0

    # validation
    with torch.no_grad():
        print('start validation')
        model = model.eval()

        val_set.history_mask_list = [None] * len(val_set)
        best_iou = 0

        for eval_iter in range(eval_iter_time):
            all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
            for i_val, batch in enumerate(valloader):
                query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index = batch
                query_rgb, support_rgb, support_mask, query_mask, history_mask = (query_rgb).cuda(0), (support_rgb).cuda(0), (support_mask).cuda(0), (query_mask).cuda(0).long(), (history_mask).cuda(0)
                query_mask = query_mask[:, 0, :, :] 

                if args.model == 'CaNet':
                    pred = model(query_rgb, support_rgb, support_mask, history_mask)
                elif args.model == 'CaNet_EAUS':
                    pred = model(query_rgb, support_rgb, support_mask, history_mask, sample_class)
                elif args.model == 'EHNet':
                    pred = model(query_rgb, support_rgb, support_mask, history_mask, sample_class)
                pred_softmax = F.softmax(pred, dim=1).data.cpu()

                # update history mask
                for j in range(support_mask.shape[0]):
                    sub_index = index[j]
                    val_set.history_mask_list[sub_index] = pred_softmax[j]
                
                pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
                _, pred_label = torch.max(pred, 1)
                inter_list, union_list, _, predict_list = get_iou_v1(query_mask, pred_label)

                for j in range(query_mask.shape[0]):#batch size
                    all_inter[(sample_class[j] - 1) % 5] += inter_list[j]
                    all_union[(sample_class[j] - 1) % 5] += union_list[j]

            IOU = [0] * 5

            for j in range(5):
                IOU[j] = all_inter[j] / (all_union[j] + 1e-10)
            mean_iou = np.mean(IOU)
            print('mean_iou:', mean_iou)
            if mean_iou > best_iou:
                best_iou = mean_iou
            else:
                break

        iou_list.append(best_iou)
        writer.add_scalar('val_iou', best_iou, epoch)
        plot_iou(checkpoint_dir, iou_list)
        np.savetxt(os.path.join(checkpoint_dir, 'iou_history.txt'), np.array(iou_list))
        memory_path = osp.join(checkpoint_dir, 'memory.pkl')
        if best_iou > highest_iou:
            highest_iou = best_iou
            model = model.eval()
            if args.model != 'CaNet':
                with open(memory_path, 'wb') as f:
                    pickle.dump(model.module.memory_pool, f)
            torch.save(model.cpu().state_dict(), osp.join(checkpoint_dir, 'model', 'best' '.pth'))
            model = model.train()
            best_epoch = epoch
            print('A better model is saved')

        print('IOU for this epoch: %.4f' % (best_iou))
        model = model.train()
        model.cuda()

    epoch_time = time.time() - begin_time
    print('best epoch:%d ,iout:%.4f' % (best_epoch, highest_iou))
    print('This epoch taks:', epoch_time, 'second')
    print('still need hour:%.4f' % ((num_epoch - epoch) * epoch_time / 3600))