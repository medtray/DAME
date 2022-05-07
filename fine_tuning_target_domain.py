import argparse
import gc
import os
import random
from typing import AnyStr
from typing import List
import ipdb
import krippendorff
from collections import defaultdict
from pathlib import Path
import math
import pickle

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from datareader import MultiDomainEntityMatchingDataset
from datareader import collate_batch_transformer
from metrics import MultiDatasetClassificationEvaluator
from metrics import ClassificationEvaluator
from metrics import acc_f1

from metrics import plot_label_distribution
from model import MultiTransformerClassifier
from model import VanillaBert
from model import *
import pandas as pd
import copy


def train(
        model: torch.nn.Module,
        train_dls: List[DataLoader],
        optimizer: [torch.optim.Optimizer,torch.optim.Optimizer],
        scheduler: LambdaLR,
        validation_evaluator: ClassificationEvaluator,
        n_epochs: int,
        device: AnyStr,
        log_interval: int = 1,
        patience: int = 10,
        model_dir: str = "wandb_local",
        gradient_accumulation: int = 1,
        domain_name: str = ''
):
    #best_loss = float('inf')
    best_F1 = 0.0


    epoch_counter = 0
    total = sum(len(dl) for dl in train_dls)

    optG = optimizer[0]
    sotmax = nn.Softmax(dim=1)

    # Main loop
    while epoch_counter < n_epochs:
        dl_iters = [iter(dl) for dl in train_dls]
        dl_idx = list(range(len(dl_iters)))
        finished = [0] * len(dl_iters)
        i = 0
        with tqdm(total=total, desc="Training") as pbar:
            while sum(finished) < len(dl_iters):
                random.shuffle(dl_idx)
                for d in dl_idx:
                    domain_dl = dl_iters[d]
                    batches = []
                    try:
                        for j in range(gradient_accumulation):
                            batches.append(next(domain_dl))
                    except StopIteration:
                        finished[d] = 1
                        if len(batches) == 0:
                            continue
                    optG.zero_grad()
                    for batch in batches:
                        model.train()
                        batch = tuple(t.to(device) for t in batch)
                        input_ids = batch[0]
                        masks = batch[1]
                        labels = batch[2]
                        domains = batch[3]
                        loss, logits, alpha = model.forward_test(input_ids, attention_mask=masks, domains=domains, labels=labels, ret_alpha = True)
                        loss = loss.mean() / gradient_accumulation

                        loss.backward()
                        i += 1
                        pbar.update(1)

                    optG.step()
                    if scheduler is not None:
                        scheduler.step()

        gc.collect()

        (loss, acc, P, R, F1), plots, (labels, logits), votes = validation_evaluator.evaluate(
            model,
            plot_callbacks=[],
            return_labels_logits=True,
            return_votes=True
        )

        print('valid F1={}'.format(F1))

        if F1 > best_F1:
            best_F1 = F1
            torch.save(model.state_dict(), f'{model_dir}/files/model_{domain_name}.pth')


        gc.collect()
        epoch_counter += 1


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=False, type=str,
                        default='entity-matching-dataset')
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.8)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    parser.add_argument("--warmup_steps", help="Number of steps to warm up Adam", type=int, default=200)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=3)
    parser.add_argument("--pretrained_bert", help="Directory with weights to initialize the shared model with",
                        type=str, default=None)
    parser.add_argument("--pretrained_multi_xformer",help="Directory with trained models", type=str,default='moe_dame')
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training',
                        default=['Walmart-Amazon', 'Abt-Buy', 'Beer', 'DBLP-GoogleScholar', 'Amazon-Google', 'cameras_',
                                 'DBLP-ACM', 'Fodors-Zagats', 'iTunes-Amazon', 'shoes_', 'computers_', 'watches_'])

    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--tags", nargs='+', help='A list of tags for this run', default=[])
    parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", help="l2 reg", type=float, default=0.01)
    parser.add_argument("--n_heads", help="Number of transformer heads", default=6, type=int)
    parser.add_argument("--n_layers", help="Number of transformer layers", default=6, type=int)
    parser.add_argument("--d_model", help="Transformer model size", default=768, type=int)
    parser.add_argument("--ff_dim", help="Intermediate feedforward size", default=2048, type=int)
    parser.add_argument("--gradient_accumulation", help="Number of gradient accumulation steps", default=1, type=int)
    parser.add_argument("--model", help="Name of the model to run", default="VanillaBert")
    parser.add_argument("--supervision_layer", help="The layer at which to use domain adversarial supervision",
                        default=6, type=int)
    parser.add_argument("--indices_dir", help="If standard splits are being used", type=str, default=None)

    args = parser.parse_args()

    #number of trials and cumulative training data from the target domain
    trials = 5
    ratio = 0.05

    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # See if CUDA available
    device = torch.device("cpu")
    if args.n_gpu > 0 and torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # model configuration
    bert_model = 'distilbert-base-uncased'
    ##################
    # override for now
    batch_size = args.batch_size
    args.gradient_accumulation = 1
    ###############
    lr = args.lr
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    bert_config = DistilBertConfig.from_pretrained(bert_model, num_labels=2, output_hidden_states=True)

    # Create the dataset
    all_dsets = [MultiDomainEntityMatchingDataset(
        args.dataset_loc,
        [domain],
        DistilBertTokenizer.from_pretrained(bert_model)
    ) for domain in args.domains]

    # Create the model
    bert = DistilBertForSequenceClassification.from_pretrained(bert_model, config=bert_config).to(device)
    multi_xformer = MultiDistilBertClassifier(
        bert_model,
        bert_config,
        n_domains=len(all_dsets) - 1
    ).to(device)
    shared_bert = VanillaBert(bert).to(device)

    # MultiViewTransformerNetworkProbabilitiesAdversarial
    model = MultiViewTransformerNetworkDomainAdversarial(
        multi_xformer,
        shared_bert,
        supervision_layer=args.supervision_layer
    ).to(device)

    all_results={}

    for i in range(len(all_dsets)):
        domain = args.domains[i]
        test_dset = all_dsets[i]
        test_dset_supervised_training = copy.deepcopy(test_dset)
        valid_dset = copy.deepcopy(test_dset)
        test_indices = test_dset.split_indices['test']
        train_indices = test_dset.split_indices['train']
        valid_indices = test_dset.split_indices['valid']
        test_dset.dataset=pd.DataFrame(test_dset.test_data)

        train_data=test_dset.original_data[train_indices[0]:train_indices[1]]
        valid_data=test_dset.original_data[valid_indices[0]:valid_indices[1]]
        valid_dset.dataset=pd.DataFrame(valid_data)


        nb_samples = int(ratio * len(train_data))
        nb_runs = int(1 / (ratio))

        for trial in range(trials):

            random.shuffle(train_data)

            model.load_state_dict(torch.load(f"{args.pretrained_multi_xformer}/files/model_{domain}.pth"))

            evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
            (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
                model,
                plot_callbacks=[],
                return_labels_logits=True,
                return_votes=True
            )
            print('initial results for {}'.format(domain))
            print(f"{domain} F1: {F1}")
            print(f"{domain} Accuracy: {acc}")
            print()

            evaluator_valid = ClassificationEvaluator(valid_dset, device, use_domain=False)

            all_f1=[]
            all_sizes=[]
            all_acc=[]
            all_precision=[]
            all_recall=[]

            all_f1.append(F1)
            all_acc.append(acc)
            all_precision.append(P)
            all_recall.append(R)
            all_sizes.append(0)

            for i in range(0,nb_runs):

                if i == nb_runs - 1:
                    samples_to_use = train_data
                else:
                    samples_to_use = train_data[:nb_samples * (i + 1)]

                all_sizes.append(int(ratio*100*(i+1)))

                test_dset_supervised_training.dataset = pd.DataFrame(samples_to_use)
                test_dset_supervised_training.set_domain_id(0)
                test_dset.set_domain_id(1)
                # For test
                train_sizes = [int(len(test_dset_supervised_training) * args.train_pct)]
                val_sizes = [len(test_dset_supervised_training) - train_sizes[0]]

                subsets = [random_split(test_dset_supervised_training, [train_sizes[0], val_sizes[0]])]

                train_dls = [DataLoader(
                    subset[0],
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_batch_transformer
                ) for subset in subsets]

                model.load_state_dict(torch.load(f"{args.pretrained_multi_xformer}/model_{domain}.pth"))

                for param in model.multi_xformer.parameters():
                    param.requires_grad = False

                for param in model.multi_xformer_classifiers.parameters():
                    param.requires_grad = False

                for param in model.domain_classifier.parameters():
                    param.requires_grad = False


                # Create the optimizer
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                     'weight_decay': weight_decay},
                    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

                optimizer_grouped_parameters[0]['params']=optimizer_grouped_parameters[0]['params'][:-1]
                optimizer_grouped_parameters[1]['params'] = optimizer_grouped_parameters[1]['params'][:-1]

                optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    args.warmup_steps,
                    n_epochs * sum([len(train_dl) for train_dl in train_dls])
                )

                opt = [optimizer]

                # Train
                train(
                    model,
                    train_dls,
                    opt,
                    scheduler,
                    evaluator_valid,
                    n_epochs,
                    device,
                    args.log_interval,
                    model_dir=args.pretrained_multi_xformer,
                    gradient_accumulation=args.gradient_accumulation,
                    domain_name=domain
                )
                # Load the best weights
                model.load_state_dict(torch.load(f'{args.pretrained_multi_xformer}/files/model_{domain}.pth'))

                evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
                (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
                    model,
                    plot_callbacks=[],
                    return_labels_logits=True,
                    return_votes=True
                )
                print('{} samples for {}'.format(nb_samples*(i+1),domain))
                print(f"{domain} F1: {F1}")
                print(f"{domain} Accuracy: {acc}")
                print()

                all_f1.append(F1)
                all_acc.append(acc)
                all_precision.append(P)
                all_recall.append(R)


            print(all_sizes)
            print(all_f1)
            print(all_acc)
            print(all_precision)
            print(all_recall)

            if domain not in all_results:
                all_results[domain]={'sizes':all_sizes,'F1':[all_f1],'acc':[all_acc], 'precision':[all_precision], 'recall':[all_recall]}
            else:
                all_results[domain]['F1'].append(all_f1)
                all_results[domain]['acc'].append(all_acc)
                all_results[domain]['precision'].append(all_precision)
                all_results[domain]['recall'].append(all_recall)

            print(all_results)

        np.save('fine_tuning_results.npy',all_results)