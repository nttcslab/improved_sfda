<div align="center">

# Official implementation of the paper "Understanding and Improving Source-free Domain Adaptation from a Theoretical Perspective" [CVPR2024]

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Static Badge](https://img.shields.io/badge/Conference-CVPR2024-blue)](https://openaccess.thecvf.com/content/CVPR2024/html/Mitsuzumi_Understanding_and_Improving_Source-free_Domain_Adaptation_from_a_Theoretical_Perspective_CVPR_2024_paper.html)

</div>

This is an official implementation of the CVPR paper "Understanding and Improving Source-free Domain Adaptation from a Theoretical Perspective

## Abstract

Source-free Domain Adaptation (SFDA) is an emerging and challenging research area that addresses the problem of unsupervised domain adaptation (UDA) without source data. Though numerous successful methods have been proposed for SFDA, a theoretical understanding of why these methods work well is still absent. In this paper, we shed light on the theoretical perspective of existing SFDA methods. Specifically, we find that SFDA loss functions comprising discriminability and diversity losses work in the same way as the training objective in the theory of self-training based on the expansion assumption, which shows the existence of the target error bound. This finding brings two novel insights that enable us to build an improved SFDA method comprising 1) Model Training with Auto-Adjusting Diversity Constraint and 2) Augmentation Training with Teacher-Student Framework, yielding a better recognition performance. Extensive experiments on three benchmark datasets demonstrate the validity of the theoretical analysis and our method.

## YouTube
<div align="center">

<a href="https://www.youtube.com/watch?v=SnWqZ_lb93Y"><img src="https://github.com/user-attachments/assets/86fa69a1-ee69-468f-ac99-d38fcb873934" alt="youtube video" width="600"/></a>

</div>

## How to run

### Create Environment

You can build the environment using `Dockerfile` in the `docker` directory.

Also you can build the environment by following the instruction below.

```bash
# clone project
git clone https://github.com/nttcslab/improvsed_sfda
cd improved_sfda

# create conda environment
cd docker
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/
conda install pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt
```

### Prepare Dataset

Please download the datasets from the original sources. Then, please place them as below.

```
data/
├── office31/
│   ├── amazon
│   ├── dslr
│   ├── webcam
│   └── image_list
├── officehome/
│   ├── Art
│   ├── Clipart
│   ├── Product
│   ├── Real_World
│   └── image_list 
└── visda2017/
    ├── train
    ├── validation
    └── image_list
```

### Source Training

Training configuration is based on [Hydra](https://hydra.cc). Please see there for the format and instructions on how to use it.

```bash
python src/train.py trainer=gpu experiment=office31_src
```

### Target Training

```bash
python src/train.py trainer=gpu experiment=office31_tgt_ours_pb_teachaug_directed
```

Please see details in: [configs/experiment/](configs/experiment/)

### To see results

The logs are managed by [mlflow](https://mlflow.org).

```bash
cd logs/mlflow

mlflow ui
```

## Acknowledgement

Our implementation is based on the following works. We greatly appreciate all these excellent works.

+ [AaD](https://github.com/Albert0147/AaD_SFDA)
+ [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template)
+ [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)
+ [DDA](https://github.com/moskomule/dda)

## Citation

```
@InProceedings{Mitsuzumi_2024_CVPR,
    author    = {Mitsuzumi, Yu and Kimura, Akisato and Kashima, Hisashi},
    title     = {Understanding and Improving Source-free Domain Adaptation from a Theoretical Perspective},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {28515-28524}
}
```
