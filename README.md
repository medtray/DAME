# DAME: Domain Adaptation for Matching Entities

This repository contains source code for the [`DAME` model](https://dl.acm.org/doi/abs/10.1145/3488560.3498486), a new domain adaptation-based method that transfers the task knowledge from multiple source domains to a target domain for entity matching (EM). `DAME` presents a new setting for EM where the objective is to capture the task-specific knowledge from pretraining our model using multiple source domains, then testing our model on a target domain. `DAME` presents a solution for the zero-shot learning case on the target domain in EM.

## Installation

First, install the conda environment `dame` with supporting libraries.

```bash
conda create --name dame python=3.7
conda activate dame
pip install -r requirements.txt
```

## Data

We use 12 datasets in our experiments. Datasets are collected from the entity resolution Benchmark datasets and the Magellan data repository. These datasets cover multiple domains including clothing, electronics, citation, restaurant, products, music, and software. Each dataset is composed of candidate pairs of records from two structured tables that have the same set of attributes. We obtained the datasets from [the Ditto repo](https://github.com/megagonlabs/ditto).

## Training with DAME

To train the matching model with `DAME`:

```bash
python train.py \
 --dataset_loc entity-matching-dataset
 --n_gpu 1
 --n_epochs 3
 --model_dir moe_dame
 --batch_size 16
 --lr 1e-5
```

where the flags are:
* ``--dataset_loc``: the data location
* ``--n_gpu``: set to 1 for using GPU
* ``--n_epochs``: number of epochs
* ``--model_dir``: the output folder where to store models
* ``--batch_size``: batch size for training and testing
* ``--lr``: learning rate for training

## Fine-tuning of DAME with different percentages of training data from target domain



## Fine-tuning of DAME with Active Learning (AL)



## Dropping expert models from source domains



## Reference

If you plan to use `DAME` in your project, please consider citing [our paper](https://dl.acm.org/doi/abs/10.1145/3488560.3498486):

```bash
@inproceedings{trabelsi22wsdm,
author = {Trabelsi, Mohamed and Heflin, Jeff and Cao, Jin},
title = {DAME: Domain Adaptation for Matching Entities},
year = {2022},
isbn = {9781450391320},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3488560.3498486},
doi = {10.1145/3488560.3498486},
pages = {1016–1024},
numpages = {9},
keywords = {domain adaptation, entity matching, transfer learning},
location = {Virtual Event, AZ, USA},
series = {WSDM '22}
}
```
 ## Contact
  
  if you have any questions, please contact Mohamed Trabelsi at mot218@lehigh.edu
