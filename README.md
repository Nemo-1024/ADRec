# ADRec

This is a PyTorch implementation for our ADRec paper.

![](asset/title.jpg)

## Requirements

The following environment packages need to be installed to set up the required dependencies.

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

Our code has been tested running under a Linux server with NVIDIA GeForce RTX 4090 GPU. 

## Usage

#### **First, navigate to the `src` directory.**

**We have provided pre-trained embedding weights, which can be directly used for subsequent backbone warm-up and full-parameter fine-tuning. You can directly run the below command for model training and evaluation.**

#### ADRec:

```
python main.py --dataset baby --model adrec
```

#### DiffuRec:

```
python main.py --dataset baby --model diffurec
```

#### DreamRec:

```
python main.py --dataset baby --model dreamrec
```

#### DreamRec:

```
python main.py --dataset baby --model sasrec
```

**We also provide a script that facilitates running multiple models across various datasets.**

```
bash baseline.bash
```

#### Pretrain embedding:

If you want to reproduce the pretrained weights, you can run the following code:

```
python main.py --dataset baby --model pretrain
```

#### ADRec with PCGrad:

```
python main.py --dataset baby --model adrec --pcgrad true
```



**If you wish to visualize the embeddings, you first need to run the model to obtain the weight files. After that, set the correct weight file name in `t-SNE.ipynb` and run the notebook.**

## Acknowledgements

[RecBole](https://recbole.io/), [DiffuRec](https://github.com/WHUIR/DiffuRec), [DreamRec](https://github.com/YangZhengyi98/DreamRec) and [SASRec+](https://github.com/antklen/sasrec-bert4rec-recsys23).

