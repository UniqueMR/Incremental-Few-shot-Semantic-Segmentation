# Incremental Few-shot Semantic Segmentation

![Overall](https://github.com/UniqueMR/Incremental-Few-shot-Semantic-Segmentation/blob/main/img/overall.png)

This project is an implementation of incremental few-shot semantic segmentation based on [CaNet](https://arxiv.org/abs/1903.02351) and [EHNet](https://arxiv.org/abs/2207.12964). Incremental few-shot semantic segmentation task aims to construct a semantic segmentation model that can be trained on few samples to include new categories without forgetting the old categories. CaNet uses [ResNet-50](https://arxiv.org/abs/1512.03385) as backbone to extract feature maps for the given support set and query set. Semantic information are extracted from the middle stage of the backbone considering that new categories and known categories might share some mid-level features. The dense comparator is designed to obtain a similarity metric between feature maps of the support set and the query set. The mask of the support set is applied to filter the irrelevant information of the background and unintended objects, while the global average pooling is used to obtain a vector representation of support set for pixel-level comparison. The iterative optimizer is used to implement the segmentation for the query set based on the dense comparison result and the last prediction. Semantic information of multiple levels is extracted by [ASPP](https://arxiv.org/abs/1606.00915).

![EAUS+HR](https://github.com/UniqueMR/Incremental-Few-shot-Semantic-Segmentation/blob/main/img/EAUS_HR.png)

Embeddings Adaptively Update Strategy (EAUS) and Hyperclass Representation (HR) are proposed in [EHNet](https://arxiv.org/abs/2207.12964), trying to solve the catastrophic forgetting issue in the incremental learning stage and improve the performance on learning new categories. EAUS establishes a memory pool of all categories' embeddings to isolate the process of knowledge learning and knowledge representation. And it uses an attention mechanism over all categories to generate a discriminative representation for each support image. The memory pool is adaptively updated under the direction of attention mechanism. Based on the memory pool that EAUS constructed, HR uses category embeddings and hyperclass embeddings to represent the semantic information of a certain class, where the category embeddings indicate discriminative features and the hyperclass embeddings indicate shared features. Hyperclass embeddings are obtained by KMeans clustering on category embeddings and the information of the two embeddings are fused by the Cross Information Module. During the incremental learning stage, hyperclass embeddings maintain stable to keep the learned knowledge from shifting while the category embeddings are updated to adapt to the new class. 

