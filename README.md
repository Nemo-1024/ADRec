# ADRec: Unlocking the Power of Diffusion Models in Sequential Recommendation [![Static Badge](https://img.shields.io/badge/Cite--us-007ec6?style=flat-square&logo=google-scholar&logoColor=white)](#-citation) [ArXiv](https://arxiv.org/abs/2505.19544)

An official implementation for the  KDD 2025 paper 'Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach'. 

Jialei Chen, Yuanbo Xu✉ and Yiheng Jiang

<img src="README.assets/overview.svg" alt="overview" style="zoom:150%;" />

## Requirements

The following environment packages must be installed to set up the required dependencies.

```
auto_mix_prep==0.2.0
einops==0.8.0
matplotlib==3.10.0
numpy==2.2.2
PyYAML==6.0.2
scipy==1.15.1
seaborn==0.13.2
torch==2.4.0
torchtune==0.4.0
tqdm==4.66.5
```

Our code has been tested, running under a Linux server with NVIDIA GeForce RTX 4090 GPU. 

## Usage

#### **First, navigate to the `src` directory.**

**We have provided pre-trained embedding weights, which can be directly used for subsequent backbone warm-up and full-parameter fine-tuning. You can directly run the below command for model training and evaluation.**

#### ADRec:

```
python main.py --dataset baby --model adrec
```

#### Pretrain embedding:

If you want to reproduce the pre-trained weights, you can run the following code:

```
python main.py --dataset baby --model pretrain
```

#### ADRec with multi-task framework PCGrad:

```
python main.py --dataset baby --model adrec --pcgrad true
```



### We also release some baselines. 

#### DiffuRec:

```
python main.py --dataset baby --model diffurec
```

#### DreamRec:

```
python main.py --dataset baby --model dreamrec
```

#### SASRec+:

```
python main.py --dataset baby --model sasrec
```



### **We also provide a script to run multiple models across various datasets.**

```
bash baseline.bash
```

#### 

### t-SNE visualization

The t-SNE visualization experiment can be conducted via `/src/t-SNE.ipynb`.

### Comprehensive evaluation of the original embedding space

A comprehensive evaluation of embedding representations in the original embedding space can be performed using `/src/embedding_metrics.ipynb`.

## Acknowledgements

[RecBole](https://recbole.io/), [DiffuRec](https://github.com/WHUIR/DiffuRec), [DreamRec](https://github.com/YangZhengyi98/DreamRec) and [SASRec+](https://github.com/antklen/sasrec-bert4rec-recsys23).

## 📄 Citation

If you find this work useful, please consider citing our paper:

```
@inproceedings{JLchen2025ADRec,
	title={Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach},
	author={Jialei Chen and Yuanbo Xu and Yiheng Jiang},
	booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
	year={2025},
	organization={ACM},
	doi = {10.1145/3711896.3737172}
}
```
