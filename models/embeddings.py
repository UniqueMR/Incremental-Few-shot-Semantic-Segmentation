from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F
from models.backbone import ResNet


class HyperclassPool():
    def __init__(self) -> None:
        self.category_id_dict = {}
        self.embedding_dict = {}
    
class EmbeddingsMemory():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.category_embeddings_pool = {}
        self.hyperclass_embeddings_pool = HyperclassPool()

    def accumulate_category_embeddings(self, embedding, category_id):
        category_id = int(category_id.cpu().detach().numpy().item())
        embedding = embedding.cpu().detach().numpy()
        if category_id not in self.category_embeddings_pool:
            embedding_list = []
            self.category_embeddings_pool[category_id] = embedding_list
        self.category_embeddings_pool[category_id].append(embedding)
        
    def generate_category_embeddings(self):
        for key in list(self.category_embeddings_pool.keys()):
            embedding_list = self.category_embeddings_pool[key]
            embedding = np.mean(embedding_list, axis=0)
            self.category_embeddings_pool[key] = embedding

    def generate_hyperclass_embeddings(self):
        category_id_list = list(self.category_embeddings_pool.keys())
        embedding_list = list(self.category_embeddings_pool.values())
        
        kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(embedding_list)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        for item in cluster_centers:
            self.hyperclass_embeddings_pool.embedding_dict[kmeans.predict([item])[0]] = item

        for i, category_id in enumerate(category_id_list):
            self.hyperclass_embeddings_pool.category_id_dict[category_id] = labels[i]
    
    def obatin_hyperclass_embedding(self, category_id):
        hyperclass_id = self.hyperclass_embeddings_pool.category_id_dict[category_id]
        return self.hyperclass_embeddings_pool.embedding_dict[hyperclass_id]
    
# define extractor for prototypes
class PrototypeExtractor(ResNet):
    '''
    model for extracting prototype for a given support image and support mask
    load the pretrained model of the base stage
    '''
    def __init__(self, block, layers, num_classes):
        super().__init__(block, layers, num_classes)

    def forward(self, support_rgb, support_mask):
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
        z = support_mask * support_rgb
        z = F.avg_pool2d(input=z,
                         kernel_size=support_rgb.shape[-2:]) * h * w / area

        return z