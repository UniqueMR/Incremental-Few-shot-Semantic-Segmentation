from models.backbone import Bottleneck
from models.embeddings import EmbeddingsMemory, PrototypeExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from dataloader.train_loader import Dataset 
import pickle
import tqdm
import argparse
from utils import *

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0,1')

args = parser.parse_args()

# construct feature extractor
gpu_list = [int(i) for i in args.gpu.split(',')]
extractor = PrototypeExtractor(Bottleneck, [3, 4, 6, 3], 2)
extractor = nn.DataParallel(extractor, gpu_list)
model_path = './checkpoint/stage0/model/best.pth'
extractor.load_state_dict(torch.load(model_path))
extractor.cuda()
extractor.eval()

# load dataset
train_set = Dataset(data_dir=data_dir, fold=[0,1,2,3], input_size=input_size, normalize_mean=IMG_MEAN,
                  normalize_std=IMG_STD,prob=prob)
trainloader = data.DataLoader(train_set, batch_size=batch_size_train, shuffle=False, num_workers=4)

# initialize memory
memory = EmbeddingsMemory()

tqdm_gen = tqdm.tqdm(trainloader)
with torch.no_grad():
    # extract prototypes to memory
    for i_iter, batch in enumerate(tqdm_gen):
        _, _, support_rgb, support_mask, _, sample_class, index = batch
        support_rgb, support_mask = support_rgb.cuda(), support_mask.cuda()

        prototype_batch = extractor(support_rgb, support_mask)
        for idx in range(prototype_batch.shape[0]):
            prototype = prototype_batch[idx].squeeze(-1).squeeze(-1)
            category_id = sample_class[idx]
            memory.accumulate_category_embeddings(embedding=prototype, category_id=category_id)

memory.generate_category_embeddings()
memory.generate_hyperclass_embeddings()

memory_path = './checkpoint/stage0/memory.pkl'

with open(memory_path, 'wb') as f:
    pickle.dump(memory, f)