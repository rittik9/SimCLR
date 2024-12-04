# SimCLR
A PyTorch Implementation Of [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)<br>
### Introduction: 
SimCLR, short for "Simple Contrastive Learning of Visual Representations," is a powerful self-supervised learning framework for learning high-quality image representations without requiring manual labels.It leverages contrastive learning, where the model is trained to pull together similar images and push apart dissimilar ones in a learned feature space.
<p align="center">
  <img src="https://sthalles.github.io/assets/contrastive-self-supervised/cover.png" width="500"/>
</p>
<p align="center"><a href="https://sthalles.github.io/simple-self-supervised-learning/">Image Source</a></p>

### Folder Structure:
```
/
│
├── Sim_CLR/
│   ├── Adam/
│   │   ├── Contrastive Training Loss per Epoch.png
│   │   ├── SIMclr_confusion_matrix.png
|   |   ├── training_validation_metrics_finetuning_simclr.png
|   |   ├── tsne_finetuning_dataset.png
|   |   ├── tsne_test_dataset.png
|   |   ├── Adam.ipynb
│   ├── SGD/
│   │   ├── Contrastive Training Loss per Epoch.png
│   │   ├── SIMclr_confusion_matrix.png
|   |   ├── training_validation_metrics_finetuning_simclr.png
|   |   ├── tsne_finetuning_dataset.png
|   |   ├── tsne_test_dataset.png
|   |   ├── SGD.ipynb
│   ├── Augmentation.png
├── Supervised_Resnet18_as_backbone/
│   ├── supervised_ Resnet18 as Feature Extractor_confusion_matrix.png
│   ├── training_validation_metrics_supervised.png
│   ├── tsne_finetuning_dataset_pretrained_resnet18.png
│   ├── tsne_test_dataset_pretrained_resnet18.png
│   ├── Supervised.ipynb
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt

```
### Implementation Overview:
In this implementation [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset has been used. For contrastive pretraining Resnet-18(pretrained=False) has been used as backbone. Before the pretraining, we performed two types of augmentation on the dataset, namely random cropping and color jittering.
<p align="center">
  <img src="https://github.com/rittik9/SimCLR/blob/master/SimCLR/Augmentaion.png" width="500"/>
</p>
<p align="center">Dataset Augmentation</p>

Then it is pretrained for 100 epochs using Adam([Adam.ipynb](https://github.com/rittik9/SimCLR/blob/master/SimCLR/Adam/Adam.ipynb)) &  Nesterov accelerated SGD([SGD.ipynb](https://github.com/rittik9/SimCLR/blob/master/SimCLR/SGD/SGD.ipynb)) and NT-Xent loss (Normalized temperature-scaled cross-entropy loss).

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*s02HAjs9xeG2ihBJyWXHLw.png" width="500"/>
</p>
<p align="center">Image Source: <a href="https://medium.com/self-supervised-learning/nt-xent-loss-normalized-temperature-scaled-cross-entropy-loss-ea5a1ede7c40">Medium</a></p>

After pretraining I threw away the projection head and I made a finetuning head and finetuned it using 6500 labeled datapoints for 20 epochs using Adam optimizer.<br>

I also implemented this project using supervised approach([Supervised.ipynb](https://github.com/rittik9/SimCLR/blob/master/Supervised_Resnet18_as_Backbone/Supervised.ipynb)).At first downloaded imagenet pretrained Resnet-18 and used it as a feature extractor for 6500 labeled datapoints and then made a finetuning head and finetuned it using those extracted features.

### How To Run:
- At first, make sure your project environment contains all the necessary dependencies by running 
```
pip install -r requirements.txt
```
- Then to run the corresponding notebooks, download them and run the cells sequentially.
- To use the trained models, download them from the links given below and change the paths in the notebooks accordingly.

### Trained Models:
-  SimCLR_Adam:
   - [simclr_backbone.pth](https://iiitbac-my.sharepoint.com/:u:/g/personal/rittik_panda_iiitb_ac_in/EZWI1EI0zpJNtx8yKDkxq7EBhpOWZN2KPTPBOfFbp1Vezw?e=I61hWJ)->This is Resnet-18(pretrained=False) backbone, which was pretrained by contrastive learning using Adam optimizer.
   - [simclr_projection_head.pth](https://iiitbac-my.sharepoint.com/:u:/g/personal/rittik_panda_iiitb_ac_in/EZGu4CAAFD9Jua0oRv7yrjcB-2akNoA2kxXtpubfb7NGRg?e=bSGcuZ)->This is the projection head used in contrastive pretraining to project the images in 128 dimension and calculate NT-Xent loss, after pretraining I threw it away.
   - [finetuned_model.pth](https://iiitbac-my.sharepoint.com/:u:/g/personal/rittik_panda_iiitb_ac_in/EdS874W3U11CpFSj6V4eW2QBEuIhzTrHjPQifJYT4w6quw?e=ottLLb)->This is the finetuning head which was finetuned by 6500 labeled datapoints(after feature extraction of those datapoints by contrastive pretrained Resnet-18).

-  SimCLR_SGD:
   - [simclr_backbone.pth](https://iiitbac-my.sharepoint.com/:u:/g/personal/rittik_panda_iiitb_ac_in/Ec0iPedOxhxPrnqW6XvDKrsBoRSoouEs__nv5O7KJUh6oA?e=AOG935)->This is Resnet-18(pretrained=False) backbone, which was pretrained by contrastive learning using Nesterov accelerated SGD optimizer.
   - [simclr_projection_head.pth](https://iiitbac-my.sharepoint.com/:u:/g/personal/rittik_panda_iiitb_ac_in/ESv0YOm_89hJqPL8XvI6SywBR4F_mpa9z_YaE5-v_qCdmw?e=BcfU3q)->This is the projection head used in contrastive pretraining to project the images in 128 dimension and calculate NT-Xent loss, after pretraining I threw it away.
   - [finetuned_model.pth](https://iiitbac-my.sharepoint.com/:u:/g/personal/rittik_panda_iiitb_ac_in/ESv0YOm_89hJqPL8XvI6SywBR4F_mpa9z_YaE5-v_qCdmw?e=BcfU3q)->This is the finetuning head which was finetuned by 6500 labeled datapoints(after feature extraction of those datapoints by contrastive pretrained Resnet-18).

-  Supervised: 
   - [supervised_model.pth](https://iiitbac-my.sharepoint.com/:u:/g/personal/rittik_panda_iiitb_ac_in/ETci1-1ZL75JmkVDmvMz2cIBfzmhkpe4Xzuu00Ujhs55ug?e=SLPLKE)->This is the finetuning head of supervised approach, which was used after feature extraction by imagenet pretrained Resnet-18.


### Results:
All the experiments were performed on **CIFAR-10** dataset.<br>
| Method  | Batch Size | Backbone | Projection Dimension| Contrastive Pretraining Epochs|Finetuning Epochs | Optimizer |Test Accuracy|Test F1 Score
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SimCLR | 256 | ResNet18(pretrained=False) | 128 | 100 |20 |Adam(pretraining),Adam(finetuning) | 0.6137 | 0.6122 |
| SimCLR | 256 | ResNet18(pretrained=False) | 128 | 100 |20 |NAG(pretraining),Adam(finetuning) | 0.6884 | 0.6854 |
| Supervised| 256 | ResNet18(pretrained=True) | 1000 | - |20 |Adam(finetuning) | 0.7558 | 0.7543 |

### Visualizations:
| ![Contrastive Training Loss per Epoch (Adam)](https://github.com/rittik9/SimCLR/blob/master/SimCLR/Adam/Contrastive%20Training%20Loss%20per%20Epoch.png) | ![Contrastive Training Loss per Epoch (NAG)](https://github.com/rittik9/SimCLR/blob/master/SimCLR/SGD/Contrastive%20Training%20Loss%20per%20Epoch.png) |
|:---:|:---:|
| *Contrastive Training Loss per Epoch (Adam)* | *Contrastive Training Loss per Epoch (NAG)* |

| ![l_A_sgd](https://github.com/rittik9/SimCLR/blob/master/SimCLR/SGD/training_validation_metrics_finetuning_simclr.png) | ![l_a_s](https://github.com/rittik9/SimCLR/blob/master/Supervised_Resnet18_as_Backbone/training_validation_metrics_supervised.png) |
|:---:|:---:|
| *Train and Validation Metrics in Finetuning of SimCLR Method* | *Train and Validation Metrics in Finetuning of Supervised Method* |

| ![CF(NAG)](https://github.com/rittik9/SimCLR/blob/master/SimCLR/SGD/SIMclr_confusion_matrix.png) | ![CF (s)](https://github.com/rittik9/SimCLR/blob/master/Supervised_Resnet18_as_Backbone/supervised_%20Resnet18%20as%20Feature%20Extractor_confusion_matrix.png) |
|:---:|:---:|
| *NAG Optimizer Based SimCLR* | *Supervised Approach* |
### Reference:
- [Ref 1:](https://www.youtube.com/watch?v=wySLC4nszv8)
- [Ref 2:](https://sthalles.github.io/simple-self-supervised-learning/)
- [Ref 3:](https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c)
- [Ref 4:](https://github.com/sthalles/SimCLR?tab=readme-ov-file)
- [Ref 5:](https://github.com/thunderInfy/simclr)
- [Ref 6:](https://github.com/Spijkervet/SimCLR)
- [Ref 7:](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
