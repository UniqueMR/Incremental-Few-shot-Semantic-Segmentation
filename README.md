# Incremental Few-shot Semantic Segmentation

![Overall](https://github.com/UniqueMR/Incremental-Few-shot-Semantic-Segmentation/blob/main/img/overall.png)

This project is an implementation of incremental few-shot semantic segmentation based on [CaNet](https://arxiv.org/abs/1903.02351) and [EHNet](https://arxiv.org/abs/2207.12964). Incremental few-shot semantic segmentation task aims to construct a semantic segmentation model that can be trained on few samples to include new categories without forgetting the old categories. CaNet uses [ResNet-50](https://arxiv.org/abs/1512.03385) as backbone to extract feature maps for the given support set and query set. Semantic information are extracted from the middle stage of the backbone considering that new categories and known categories might share some mid-level features. The dense comparator is designed to obtain a similarity metric between feature maps of the support set and the query set. The mask of the support set is applied to filter the irrelevant information of the background and unintended objects, while the global average pooling is used to obtain a vector representation of support set for pixel-level comparison. The iterative optimizer is used to implement the segmentation for the query set based on the dense comparison result and the last prediction. Semantic information of multiple levels is extracted by [ASPP](https://arxiv.org/abs/1606.00915).

![EAUS+HR](https://github.com/UniqueMR/Incremental-Few-shot-Semantic-Segmentation/blob/main/img/EAUS_HR.png)

