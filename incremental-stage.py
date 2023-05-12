import os
import time
import cv2
import pickle
import argparse
import random
import tqdm
import numpy as np
import os.path as osp
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils import data
from dataloader.train_loader import Dataset as Dataset_incremental_train
from dataloader.val_loader import Dataset as Dataset_incremental_val
from dataloader.few_shot_loader import generate_few_shot
from models.models import CaNet, CaNet_EAUS, EHNet
from models.backbone import Bottleneck
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='gpu nums')
parser.add_argument('--stage', type=int, default=1, help='stage id')
parser.add_argument('--model', type=str, default='CaNet', help='model name')
parser.add_argument('--new', type=str, default='False', help='new category or not')
parser.add_argument('--shots', type=int, default=5, help='number of shots')

args = parser.parse_args()

# settings
set_seed(20010302)
train_fold = [args.stage] #training fold
if args.new == 'True':
    val_fold = [args.stage] #validation fold
else:
    val_fold = [i for i in range(args.stage)] #validation fold

# set device
gpu_list = [i for i in range(args.gpu)]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_list])

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.cuda.init()
cudnn.enabled = True

# load memory
if args.model != 'CaNet':
    memory_path = 'checkpoint/stage{}/{}/memory.pkl'.format(args.stage - 1, args.model)
    print('load memory from {}'.format(memory_path))
    with open(memory_path, 'rb') as f:
        memory = pickle.load(f)

# set dataloader
train_set = Dataset_incremental_train(data_dir=data_dir, fold=train_fold, input_size=input_size, normalize_mean=IMG_MEAN,
                  normalize_std=IMG_STD,prob=prob, shots=args.shots)
trainloader = data.DataLoader(train_set, batch_size=batch_size_few_shot, shuffle=True, num_workers=4)
few_shot_set = generate_few_shot(trainloader, train_set.class_available)

val_set = Dataset_incremental_val(data_dir=data_dir, fold=val_fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD,shots=args.shots)
valloader = data.DataLoader(val_set, batch_size=batch_size_val, shuffle=False, num_workers=4, drop_last=False)

# set model
model_path = 'checkpoint/stage{}/{}/model/best.pth'.format(args.stage - 1, args.model)
print('load model from {}'.format(model_path))
if args.model == 'CaNet':
    model = CaNet(Bottleneck,[3, 4, 6, 3], 2)
elif args.model == 'CaNet_EAUS':
    model = CaNet_EAUS(Bottleneck,[3, 4, 6, 3], 2, memory_pool=memory)
elif args.model == 'EHNet':
    model = EHNet(Bottleneck,[3, 4, 6, 3], 2, memory_pool=memory)
model = load_resnet50_param(model, stop_layer='layer4')
turn_off(model)

# incremental training
optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': 10 * learning_rate}],
                          lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# set result saver
checkpoint_dir = 'checkpoint/stage{}/{}/'.format(args.stage, args.model)
check_dir(checkpoint_dir)
writer = SummaryWriter(checkpoint_dir)
save_pred_every = 1
loss_list, iou_list = [], []
highest_iou, tempory_loss, best_epoch = 0, 0, 0

# start incremental training
model = nn.DataParallel(model,gpu_list)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False)
model.cuda()
model.train()

begin_time = time.time()
tqdm_train = tqdm.tqdm(few_shot_set, total=len(few_shot_set), desc='e: loss = %.4f-:%.4f' % (0, 0))

# training loop
for i_iter, batch in enumerate(tqdm_train):
    query_rgb, query_mask,support_rgbs, support_masks,history_mask,sample_class,index= batch
    query_rgb, support_rgbs, support_masks, query_mask, history_mask = (query_rgb).cuda(), (support_rgbs).cuda(), (support_masks).cuda(), (query_mask).cuda().long(), (history_mask).cuda()
    query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use

    optimizer.zero_grad()

    if args.model == 'CaNet':
        pred = model(query_rgb, support_rgbs, support_masks, history_mask)
    elif args.model == 'CaNet_EAUS':
        pred = model(query_rgb, support_rgbs, support_masks, history_mask, sample_class)
    elif args.model == 'EHNet':
        pred = model(query_rgb, support_rgbs, support_masks, history_mask, sample_class)

    pred_softmax = F.softmax(pred, dim=1).data.cpu()

    # update history mask
    for j in range (support_masks[0].shape[0]):
        sub_index=index[j]
        train_set.history_mask_list[sub_index]=pred_softmax[j]

    pred = nn.functional.interpolate(pred,size=input_size, mode='bilinear',align_corners=True)#upsample

    loss = loss_calc_v1(pred, query_mask, 0)
    # loss.backward()
    # optimizer.step()

    # track training loss
    tqdm_train.set_description('e: loss = %.4f-:%.4f' % (loss.item(), highest_iou))

    #save training loss
    writer.add_scalar('train_loss', loss.item(), i_iter)
    tempory_loss += loss.item()
    if i_iter % save_pred_every == 0:
        loss_list.append(tempory_loss / save_pred_every)
        plot_loss(checkpoint_dir, loss_list, save_pred_every)
        np.savetxt(os.path.join(checkpoint_dir, 'loss_history.txt'), np.array(loss_list))
        tempory_loss = 0

memory_path = checkpoint_dir + '/memory.pkl'

if args.model != 'CaNet':
    with open(memory_path, 'wb') as f:
        pickle.dump(model.module.memory_pool, f)

# validation
with torch.no_grad():
    print('start validation')
    model = model.eval()

    val_set.history_mask_list = [None] * len(val_set)

    for eval_iter in range(eval_iter_time):
        all_inter, all_union, all_predict = [0] * 5 * len(val_fold), [0] * 5 * len(val_fold), [0] * 5 * len(val_fold)
        for i_val, batch in enumerate(valloader):
            query_rgb, query_mask, support_rgbs, support_masks, history_mask, sample_class, index = batch
            query_rgb, support_rgbs, support_masks, query_mask, history_mask = (query_rgb).cuda(0), (support_rgbs).cuda(0), (support_masks).cuda(0), (query_mask).cuda(0).long(), (history_mask).cuda(0)
            query_mask = query_mask[:, 0, :, :] 
            
            if args.model == 'CaNet':   
                pred = model(query_rgb, support_rgbs, support_masks, history_mask)
            elif args.model == 'CaNet_EAUS':
                pred = model(query_rgb, support_rgbs, support_masks, history_mask, sample_class)
            elif args.model == 'EHNet':
                pred = model(query_rgb, support_rgbs, support_masks, history_mask, sample_class)

            pred_softmax = F.softmax(pred, dim=1).data.cpu()

            # update history mask
            for j in range(query_mask.shape[0]):
                sub_index = index[j]
                val_set.history_mask_list[sub_index] = pred_softmax[j]
            
            pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
            _, pred_label = torch.max(pred, 1)
            inter_list, union_list, _, predict_list = get_iou_v1(query_mask, pred_label)

            for j in range(query_mask.shape[0]):#batch size
                if args.new == 'True':
                    all_inter[(sample_class[j] - 1) % 5] += inter_list[j]
                    all_union[(sample_class[j] - 1) % 5] += union_list[j]
                else:
                    all_inter[sample_class[j] - 1] += inter_list[j]
                    all_union[sample_class[j] - 1] += union_list[j]

        IOU = [0] * len(val_fold) * 5

        for j in range(5 * len(val_fold)):
            IOU[j] = all_inter[j] / (all_union[j] + 1e-10)
        mean_iou = np.mean(IOU)
        print('mean_iou:', mean_iou)
        best_iou = mean_iou

    iou_list.append(best_iou)
    plot_iou(checkpoint_dir, iou_list)
    np.savetxt(os.path.join(checkpoint_dir, 'iou_history.txt'), np.array(iou_list))
    if best_iou > highest_iou:
        highest_iou = best_iou
        model = model.eval()
        torch.save(model.cpu().state_dict(), osp.join(checkpoint_dir, 'model', 'best' '.pth'))
        model = model.train()
        print('A better model is saved')

    print('IOU for this epoch: %.4f' % (best_iou))
    model = model.train()
    model.cuda()