import tqdm
import random
import torch

def generate_few_shot(trainloader, class_available):
    tqdm_gen = tqdm.tqdm(trainloader)
    category_groups = {i: [] for i in class_available}
    for batch in tqdm_gen:
        query_rgb, query_mask,support_rgbs, support_masks,history_mask,sample_class,index= batch
        category_groups[int(sample_class[0])].append([query_rgb[0], query_mask[0], support_rgbs[0], support_masks[0], history_mask[0], index[0]])

    for group in category_groups.values():
        random.shuffle(group)

    few_shot_data = []
    query_rgb_batch = []
    query_mask_batch = []
    support_rgb_batch = []
    support_mask_batch = []
    history_mask_batch = []
    sample_class_batch = []
    idx_batch = []
    choice_list = list(class_available.copy())
    random.shuffle(choice_list)
    for class_idx in choice_list:
        data_item = category_groups[class_idx][0]
        query_rgb_batch.append(data_item[0])
        query_mask_batch.append(data_item[1])
        support_rgb_batch.append(data_item[2])
        support_mask_batch.append(data_item[3])
        history_mask_batch.append(data_item[4])
        sample_class_batch.append(torch.tensor(class_idx))
        idx_batch.append(data_item[5])
    few_shot_data_item = [torch.stack(query_rgb_batch, dim=0),
                            torch.stack(query_mask_batch, dim=0),
                            torch.stack(support_rgb_batch, dim=0),
                            torch.stack(support_mask_batch, dim=0),
                            torch.stack(history_mask_batch, dim=0),
                            torch.stack(sample_class_batch, dim=0),
                            torch.stack(idx_batch, dim=0)]
    few_shot_data.append(few_shot_data_item)
    return few_shot_data

