from models.backbone import ResNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class CaNet(ResNet):
    '''
    model for base stage and baseline of incremental stage
    accept support image and support mask to generate prototype of a certain sample class
    segment the sample class on the given query image and use the query mask to supervise the segmentation
    '''

    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)
    
    def forward(self, query_rgb, support_rgbs, support_masks, history_mask):
        # important: do not optimize the RESNET backbone
        query_rgb = self.conv1(query_rgb)
        query_rgb = self.bn1(query_rgb)
        query_rgb = self.relu(query_rgb)
        query_rgb = self.maxpool(query_rgb)
        query_rgb = self.layer1(query_rgb)
        query_rgb = self.layer2(query_rgb)
        query_feat_layer2=query_rgb
        query_rgb = self.layer3(query_rgb)
        # query_rgb = self.layer4(query_rgb)
        query_rgb=torch.cat([query_feat_layer2,query_rgb],dim=1)

        query_rgb = self.layer5(query_rgb)

        feature_size = query_rgb.shape[-2:]
        z_list = []
        support_rgbs = support_rgbs.permute(1, 0, 2, 3, 4)
        support_masks = support_masks.permute(1, 0, 2, 3, 4)
        for support_rgb, support_mask in zip(support_rgbs, support_masks):
            support_rgb = self.conv1(support_rgb)
            support_rgb = self.bn1(support_rgb)
            support_rgb = self.relu(support_rgb)
            support_rgb = self.maxpool(support_rgb)
            support_rgb = self.layer1(support_rgb)
            support_rgb = self.layer2(support_rgb)
            support_feat_layer2 = support_rgb
            support_rgb = self.layer3(support_rgb)
            # support_rgb = self.layer4(support_rgb)
            support_rgb = torch.cat([support_feat_layer2, support_rgb], dim=1)
            support_rgb = self.layer5(support_rgb)

            support_mask = F.interpolate(support_mask, feature_size, mode='bilinear', align_corners=True)

            h, w = support_rgb.shape[-2:][0], support_rgb.shape[-2:][1]


            area = F.avg_pool2d(support_mask, support_rgb.shape[-2:]) * h * w + 0.0005
            z = support_mask * support_rgb
            z = F.avg_pool2d(input=z,
                            kernel_size=support_rgb.shape[-2:]) * h * w / area
            z_list.append(z)

        # calculate the avg of z_list
        z = torch.stack(z_list, dim=0).mean(dim=0)
        z = z.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat

        history_mask=F.interpolate(history_mask,feature_size,mode='bilinear',align_corners=True)

        out=torch.cat([query_rgb,z],dim=1)
        out = self.layer55(out)
        out_plus_history=torch.cat([out,history_mask],dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)




        global_feature=F.avg_pool2d(out,kernel_size=feature_size)
        global_feature=self.layer6_0(global_feature)
        global_feature=global_feature.expand(-1,-1,feature_size[0],feature_size[1])
        out=torch.cat([global_feature,self.layer6_1(out),self.layer6_2(out),self.layer6_3(out),self.layer6_4(out)],dim=1)
        out=self.layer7(out)

        out=self.layer9(out)
        return out

    
class CaNet_EAUS(ResNet):
    '''
    model for incremental stage using effective and adaptive update strategy
    training mode:
        adaptively update the category embedding in the memory pool
        use the updated category embedding to implement the segmentation
    testing mode:
        directly extract the category embedding in the memory pool to implement the segmentation'''
    def __init__(self, block, layers, num_classes, memory_pool=None):
        super().__init__(block, layers, num_classes)
        # the memory_pool is an object of class EmbeddingsMemory defined in embeddings_memory.py
        self.memory_pool = memory_pool
    
    # copy the forward function from ResNet defined in one_shot_network.py
    def forward(self, query_rgb,support_rgbs,support_masks,history_mask,sample_class):
        # important: do not optimize the RESNET backbone
        query_rgb = self.conv1(query_rgb)
        query_rgb = self.bn1(query_rgb)
        query_rgb = self.relu(query_rgb)
        query_rgb = self.maxpool(query_rgb)
        query_rgb = self.layer1(query_rgb)
        query_rgb = self.layer2(query_rgb)
        query_feat_layer2=query_rgb
        query_rgb = self.layer3(query_rgb)
        # query_rgb = self.layer4(query_rgb)
        query_rgb=torch.cat([query_feat_layer2,query_rgb],dim=1)

        query_rgb = self.layer5(query_rgb)

        feature_size = query_rgb.shape[-2:]

        support_rgbs = support_rgbs.permute(1, 0, 2, 3, 4)
        support_masks = support_masks.permute(1, 0, 2, 3, 4)

        for support_rgb, support_mask in zip(support_rgbs, support_masks):


        # if self.training:
            #side branch,get latent embedding z
            support_rgb = self.conv1(support_rgb)
            support_rgb = self.bn1(support_rgb)
            support_rgb = self.relu(support_rgb)
            support_rgb = self.maxpool(support_rgb)
            support_rgb = self.layer1(support_rgb)
            support_rgb = self.layer2(support_rgb)
            support_feat_layer2 = support_rgb
            support_rgb = self.layer3(support_rgb)

            #support_rgb = self.layer4(support_rgb)
            support_rgb = torch.cat([support_feat_layer2, support_rgb], dim=1)
            support_rgb = self.layer5(support_rgb)


            support_mask = F.interpolate(support_mask, support_rgb.shape[-2:], mode='bilinear',align_corners=True)
            h,w=support_rgb.shape[-2:][0],support_rgb.shape[-2:][1]


            area = F.avg_pool2d(support_mask, support_rgb.shape[-2:]) * h * w + 0.0005
            _z = support_mask * support_rgb
            _z = F.avg_pool2d(input=_z,
                            kernel_size=support_rgb.shape[-2:]) * h * w / area

            z = []
            for i, category_embedding in enumerate(_z):
                category = int(sample_class[i].detach().cpu().numpy())
                # update the category embedding in the memory pool
                z_item = self.adaptive_update(category=category,category_embedding=category_embedding)
                z.append(z_item)

            z = torch.stack(z, dim=0)


        z = z.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat

        # else:
        #     z = []
        #     for idx in sample_class:
        #         idx = int(idx.detach().cpu().numpy())
        #         z.append(torch.tensor(self.memory_pool.category_embeddings_pool[idx]).unsqueeze(-1).unsqueeze(-1).cuda())
        #     z = torch.stack(z, dim=0)
        #     z = z.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        

        history_mask=F.interpolate(history_mask,feature_size,mode='bilinear',align_corners=True)

        out=torch.cat([query_rgb,z],dim=1)
        out = self.layer55(out)
        out_plus_history=torch.cat([out,history_mask],dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        global_feature=F.avg_pool2d(out,kernel_size=feature_size)
        global_feature=self.layer6_0(global_feature)
        global_feature=global_feature.expand(-1,-1,feature_size[0],feature_size[1])
        out=torch.cat([global_feature,self.layer6_1(out),self.layer6_2(out),self.layer6_3(out),self.layer6_4(out)],dim=1)
        out=self.layer7(out)

        out=self.layer9(out)
        return out
        
    def compute_relation_coefficient(self, category_embeddings):
        # compute the relation coefficient between each two elements in memory_pool.category_embeddings by the inner product of the two elements
        # the result is a matrix with shape [num_categories, num_categories]
        e = torch.matmul(category_embeddings, category_embeddings.t())
        # normalize the matrix with a softmax function
        e = F.softmax(e, dim=1)
        return e
    
    def adaptive_update(self, category, category_embedding):
        category_embeddings = torch.tensor(list(self.memory_pool.category_embeddings_pool.values())).cuda()
        e = self.compute_relation_coefficient(category_embeddings=category_embeddings)
        # get the list of categories in the memory pool
        categories = list(self.memory_pool.category_embeddings_pool.keys())
        # get the index of the category in the list
        category_index = categories.index(category)
        # extract the row of the matrix e corresponding to the category
        e_row = e[category_index]
        category_embedding = category_embedding.squeeze(-1).squeeze(-1)
        new_category_embedding = category_embedding
        for i in range(len(categories)):
            # add the vector multiplication of vector e_row and vector category_embeddings[i] to new_category_embedding
            new_category_embedding = new_category_embedding + e_row[i] * (category_embedding - category_embeddings[i])
        if self.training:
            # update the category embedding in the memory pool
            self.memory_pool.category_embeddings_pool[category] = new_category_embedding.detach().cpu().numpy()
        return new_category_embedding.unsqueeze(-1).unsqueeze(-1)
    
class EHNet(CaNet_EAUS):
    def __init__(self, block, layers, num_classes, memory_pool=None):
        super().__init__(block, layers, num_classes, memory_pool)
    
    def forward(self, query_rgb,support_rgb,support_mask,history_mask,sample_class):
        # important: do not optimize the RESNET backbone
        query_rgb = self.conv1(query_rgb)
        query_rgb = self.bn1(query_rgb)
        query_rgb = self.relu(query_rgb)
        query_rgb = self.maxpool(query_rgb)
        query_rgb = self.layer1(query_rgb)
        query_rgb = self.layer2(query_rgb)
        query_feat_layer2=query_rgb
        query_rgb = self.layer3(query_rgb)
        # query_rgb = self.layer4(query_rgb)
        query_rgb=torch.cat([query_feat_layer2,query_rgb],dim=1)

        query_rgb = self.layer5(query_rgb)

        feature_size = query_rgb.shape[-2:]

        # if self.training:
        #side branch,get latent embedding z
        support_rgb = self.conv1(support_rgb)
        support_rgb = self.bn1(support_rgb)
        support_rgb = self.relu(support_rgb)
        support_rgb = self.maxpool(support_rgb)
        support_rgb = self.layer1(support_rgb)
        support_rgb = self.layer2(support_rgb)
        support_feat_layer2 = support_rgb
        support_rgb = self.layer3(support_rgb)

        #support_rgb = self.layer4(support_rgb)
        support_rgb = torch.cat([support_feat_layer2, support_rgb], dim=1)
        support_rgb = self.layer5(support_rgb)


        support_mask = F.interpolate(support_mask, support_rgb.shape[-2:], mode='bilinear',align_corners=True)
        h,w=support_rgb.shape[-2:][0],support_rgb.shape[-2:][1]


        area = F.avg_pool2d(support_mask, support_rgb.shape[-2:]) * h * w + 0.0005
        _z = support_mask * support_rgb
        _z = F.avg_pool2d(input=_z,
                        kernel_size=support_rgb.shape[-2:]) * h * w / area

        z = []
        for i, category_embedding in enumerate(_z):
            category = int(sample_class[i].detach().cpu().numpy())
            hyperclass_embedding = torch.tensor(self.memory_pool.obatin_hyperclass_embedding(category_id=category), dtype=torch.float32).cuda()
            category_embedding = self.fuse_embedding(category_embedding=category_embedding, hyperclass_embedding=hyperclass_embedding, category_id=category)
            # update the category embedding in the memory pool
            z_item = self.adaptive_update(category=category,category_embedding=category_embedding)
            z.append(z_item)

        z = torch.stack(z, dim=0)
        z = z.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat

        # else:
        #     z = []
        #     for idx in sample_class:
        #         idx = int(idx.detach().cpu().numpy())
        #         z.append(torch.tensor(self.memory_pool.category_embeddings_pool[idx]).unsqueeze(-1).unsqueeze(-1).cuda())
        #     z = torch.stack(z, dim=0)
        #     z = z.expand(-1, -1, feature_size[0], feature_size[1])  # tile for cat
        

        history_mask=F.interpolate(history_mask,feature_size,mode='bilinear',align_corners=True)

        out=torch.cat([query_rgb,z],dim=1)
        out = self.layer55(out)
        out_plus_history=torch.cat([out,history_mask],dim=1)
        out = out + self.residule1(out_plus_history)
        out = out + self.residule2(out)
        out = out + self.residule3(out)

        global_feature=F.avg_pool2d(out,kernel_size=feature_size)
        global_feature=self.layer6_0(global_feature)
        global_feature=global_feature.expand(-1,-1,feature_size[0],feature_size[1])
        out=torch.cat([global_feature,self.layer6_1(out),self.layer6_2(out),self.layer6_3(out),self.layer6_4(out)],dim=1)
        out=self.layer7(out)

        out=self.layer9(out)
        return out
    
    def fuse_embedding(self, category_embedding, hyperclass_embedding, category_id):
        category_embedding = category_embedding.squeeze(-1).squeeze(-1)
        category_embedding = torch.sigmoid(category_embedding)
        hyperclass_embedding = torch.sigmoid(hyperclass_embedding)
        fused_embedding = category_embedding * hyperclass_embedding
        hyperclass_id = self.memory_pool.hyperclass_embeddings_pool.category_id_dict[category_id]
        hyperclass_embedding = fused_embedding * hyperclass_embedding
        if self.training:
            self.memory_pool.hyperclass_embeddings_pool.embedding_dict[hyperclass_id] = hyperclass_embedding.detach().cpu().numpy()
        category_embedding = fused_embedding * category_embedding
        return category_embedding.unsqueeze(-1).unsqueeze(-1)