# Incremental Few-shot Semantic Segmentation

![Overall](https://github.com/UniqueMR/Incremental-Few-shot-Semantic-Segmentation/blob/main/img/overall.png)

This project is an implementation of incremental few-shot semantic segmentation based on [CaNet](https://arxiv.org/abs/1903.02351) and [EHNet](https://arxiv.org/abs/2207.12964). Incremental few-shot semantic segmentation task aims to construct a semantic segmentation model that can be trained on few samples to include new categories without forgetting the old categories. CaNet uses [ResNet-50](https://arxiv.org/abs/1512.03385) as backbone to extract feature maps for the given support set and query set. Semantic information are extracted from the middle stage of the backbone considering that new categories and known categories might share some mid-level features. The dense comparator is designed to obtain a similarity metric between feature maps of the support set and the query set. The mask of the support set is applied to filter the irrelevant information of the background and unintended objects, while the global average pooling is used to obtain a vector representation of support set for pixel-level comparison. The iterative optimizer is used to implement the segmentation for the query set based on the dense comparison result and the last prediction. Semantic information of multiple levels is extracted by [ASPP](https://arxiv.org/abs/1606.00915).

![EAUS+HR](https://github.com/UniqueMR/Incremental-Few-shot-Semantic-Segmentation/blob/main/img/EAUS_HR.png)

Embeddings Adaptively Update Strategy (EAUS) and Hyperclass Representation (HR) are proposed in [EHNet](https://arxiv.org/abs/2207.12964), trying to solve the catastrophic forgetting issue in the incremental learning stage and improve the performance on learning new categories. EAUS establishes a memory pool of all categories' embeddings to isolate the process of knowledge learning and knowledge representation. And it uses an attention mechanism over all categories to generate a discriminative representation for each support image. The memory pool is adaptively updated under the direction of attention mechanism. Based on the memory pool that EAUS constructed, HR uses category embeddings and hyperclass embeddings to represent the semantic information of a certain class, where the category embeddings indicate discriminative features and the hyperclass embeddings indicate shared features. Hyperclass embeddings are obtained by KMeans clustering on category embeddings and the information of the two embeddings are fused by the Cross Information Module. During the incremental learning stage, hyperclass embeddings maintain stable to keep the learned knowledge from shifting while the category embeddings are updated to adapt to the new class. 

## Notes

* The implementation of CaNet referenced [icoz69's repository](https://github.com/icoz69/CaNet), thanks for their awesome works
* The experiment on EAUS and HR didn't achieved the ideal result shown in [EHNet](https://arxiv.org/abs/2207.12964). There might still exist some misunderstandings on the experimental settings and the detailed implementation. Feel free to pull requests if you achieved better results.

## Requirements

* torch, torchvision
* opencv-python 
* scikit-learn

## Data Preparation

The experiment of this project is based on PASCAL-5i dataset which is derived from [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) dataset and specially designed for incremental few -shot learing task. You are supposed to download the dataset yourself and organize your dataset as followed. (the Binary_map_aug is prepared in this repository and you can use it directly)

```
├── dataset 
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── Annotations
│   │   │   ├── Binary_map_aug
│   │   │   ├── ImgaeSets
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── SegmentationObject
```

## Incremental Stage

For incremental stage, run the .sh file using the following command:

```
bash run.sh
```

You might need to modify the content of the .sh file to satisfy you demand. The content has the following structure:

```
python incremental-stage.py --model [MODEL_NAME] --new [] --shots [NUM_SHOTS] --stage 1
python incremental-stage.py --model [MODEL_NAME] --new [] --shots [NUM_SHOTS] --stage 2
python incremental-stage.py --model [MODEL_NAME] --new [] --shots [NUM_SHOTS] --stage 3
```

These commands will consecutively execute the training and evaluation from stage 1 to stage 3. The "--model" argument is used to select which model you want to use. It has 3 options: CaNet, CaNet_EAUS, and EHNet. Remember to keep the "--model" option the same for all the stages. The "--new" argument with two options, True and False, is used to decide whether the evaluation is executed on new categories or known categories. This argument won't change the behavior of the incremental training process. The "--shots" argument determines how many samples each support set contains. To evaluate the few-shot learning performance, this argument should be set to be small though all integer values are allowed. 

## Base Stage

If you want to execute the base stage yourself, you can try the following command:

```
python base-stage.py
```

This step is optional since the pertained weights have been provided in this repository. 

