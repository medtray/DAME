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
        validation_evaluator: MultiDatasetClassificationEvaluator,
        n_epochs: int,
        device: AnyStr,
        log_interval: int = 1,
        patience: int = 10,
        model_dir: str = "wandb_local",
        gradient_accumulation: int = 1,
        domain_name: str = ''
):
    #best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    epoch_counter = 0
    total = sum(len(dl) for dl in train_dls)

    optG = optimizer[0]
    optD = optimizer[1]
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

                    for batch in batches:
                        model.train()
                        batch = tuple(t.to(device) for t in batch)
                        input_ids = batch[0]
                        masks = batch[1]
                        labels = batch[2]
                        # Null the labels if its the test data
                        if d == len(train_dls) - 1:
                            labels = None
                        # Testing with random domains to see if any effect
                        #domains = torch.tensor(np.random.randint(0, 16, batch[3].shape)).to(device)
                        domains = batch[3]

                        outputs = model.shared_bert(input_ids, attention_mask=masks)


                        divisor = min(1, 2 * (len(outputs[1]) - model.supervision_layer))
                        domain_supervision_layer = outputs[1][model.supervision_layer][:, 0, :]
                        #adv_input = GradientReversal.apply(domain_supervision_layer)

                        adv_logits = model.domain_classifier(domain_supervision_layer)
                        probs_domain_classifier = sotmax(adv_logits)
                        #print(probs_domain_classifier)
                        #print(domains)
                        lossD = (1e-2 / divisor) * nn.CrossEntropyLoss()(adv_logits, domains)
                        optD.zero_grad()
                        lossD.backward()
                        optD.step()

                        loss, logits, alpha = model(input_ids, attention_mask=masks, domains=domains, labels=labels, ret_alpha = True)
                        loss = loss.mean() / gradient_accumulation
                        #print(loss.item())

                        optG.zero_grad()
                        loss.backward()
                        i += 1
                        pbar.update(1)

                    optG.step()
                    if scheduler is not None:
                        scheduler.step()

        gc.collect()

        # Inline evaluation
        (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
        print(f"Validation acc: {acc}")

        #torch.save(model.state_dict(), f'{model_dir}/files/model_{domain_name}.pth')

        # Saving the best model and early stopping
        #if val_loss < best_loss:
        if acc > best_acc:
            best_model = model.state_dict()
            #best_loss = val_loss
            best_acc = acc
            torch.save(model.state_dict(), f'{model_dir}/files/model_{domain_name}.pth')
            patience_counter = 0

        else:
            patience_counter += 1
            # Stop training once we have lost patience
            if patience_counter == patience:
                break

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
    parser.add_argument("--pretrained_bert", help="Directory with weights to initialize the shared model with", type=str, default=None)
    parser.add_argument("--pretrained_multi_xformer", help="Directory with weights to initialize the domain specific models", type=str, default=None)
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training', default=['Walmart-Amazon','Abt-Buy','Beer','DBLP-GoogleScholar','Amazon-Google','cameras_','DBLP-ACM','Fodors-Zagats','iTunes-Amazon','shoes_','computers_','watches_'])

    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--model_dir", help="Where to store the saved model",default="moe_dame", type=str)
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
    parser.add_argument("--supervision_layer", help="The layer at which to use domain adversarial supervision", default=6, type=int)
    parser.add_argument("--indices_dir", help="If standard splits are being used", type=str, default=None)

    args = parser.parse_args()

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
        device = torch.device("cuda:1")

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

    #Create save directory for model
    if not os.path.exists(f"{args.model_dir}/files"):
        os.makedirs(f"{args.model_dir}/files")

    # Create the dataset
    all_dsets = [MultiDomainEntityMatchingDataset(
        args.dataset_loc,
        [domain],
        DistilBertTokenizer.from_pretrained(bert_model)
    ) for domain in args.domains]
    train_sizes = [int(len(dset) * args.train_pct) for j, dset in enumerate(all_dsets)]
    val_sizes = [len(all_dsets[j]) - train_sizes[j] for j in range(len(train_sizes))]

    accs = []
    Ps = []
    Rs = []
    F1s = []
    # Store labels and logits for individual splits for micro F1
    labels_all = []
    logits_all = []

    for i in range(len(all_dsets)):
        domain = args.domains[i]
        test_dset = all_dsets[i]
        test_dset_unsupervised_training = copy.deepcopy(test_dset)
        test_indices = test_dset.split_indices['test']
        train_indices = test_dset.split_indices['train']
        valid_indices = test_dset.split_indices['valid']
        #test_dset.dataset = pd.DataFrame(test_dset.original_data[test_indices[0]:test_indices[1]])
        test_dset.dataset = pd.DataFrame(test_dset.test_data)
        test_dset_unsupervised_training.dataset = pd.DataFrame(
            test_dset.original_data[train_indices[0]:train_indices[1]] + test_dset.original_data[
                                                                         valid_indices[0]:valid_indices[1]])

        # Override the domain IDs
        k = 0
        for j in range(len(all_dsets)):
            if j != i:
                all_dsets[j].set_domain_id(k)
                k += 1
        test_dset.set_domain_id(k)
        test_dset_unsupervised_training.set_domain_id(k)
        # For test
        #all_dsets = [all_dsets[0], all_dsets[2]]

        # Split the data
        if args.indices_dir is None:
            subsets = [random_split(all_dsets[j], [train_sizes[j], val_sizes[j]])
                       for j in range(len(all_dsets)) if j != i]
        else:
            # load the indices
            dset_choices = [all_dsets[j] for j in range(len(all_dsets)) if j != i]
            subset_indices = defaultdict(lambda: [[], []])
            with open(f'{args.indices_dir}/train_idx_{domain}.txt') as f, \
                    open(f'{args.indices_dir}/val_idx_{domain}.txt') as g:
                for l in f:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][0].append(int(vals[1]))
                for l in g:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][1].append(int(vals[1]))
            subsets = [[Subset(dset_choices[d], subset_indices[d][0]), Subset(dset_choices[d], subset_indices[d][1])] for d in
                       subset_indices]

        train_dls = [DataLoader(
            subset[0],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch_transformer
        ) for subset in subsets]
        # Add test data for domain adversarial training
        train_dls += [DataLoader(
            test_dset_unsupervised_training,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch_transformer
        )]

        val_ds = [subset[1] for subset in subsets]
        # for vds in val_ds:
        #     print(vds.indices)
        validation_evaluator = MultiDatasetClassificationEvaluator(val_ds, device)

        # Create the model
        bert = DistilBertForSequenceClassification.from_pretrained(bert_model, config=bert_config).to(device)
        multi_xformer = MultiDistilBertClassifier(
            bert_model,
            bert_config,
            n_domains=len(train_dls) - 1
        ).to(device)
        if args.pretrained_multi_xformer is not None:
            multi_xformer.load_state_dict(torch.load(f"{args.pretrained_multi_xformer}/model_{domain}.pth"))
            (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(multi_xformer)
            print(f"Validation acc multi-xformer: {acc}")

        shared_bert = VanillaBert(bert).to(device)
        if args.pretrained_bert is not None:
            shared_bert.load_state_dict(torch.load(f"{args.pretrained_bert}/model_{domain}.pth"))
            (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(shared_bert)
            print(f"Validation acc shared bert: {acc}")

        #MultiViewTransformerNetworkProbabilitiesAdversarial
        model = MultiViewTransformerNetworkDomainAdversarial(
            multi_xformer,
            shared_bert,
            supervision_layer=args.supervision_layer
        ).to(device)
        # (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)
        # print(f"Validation acc starting: {acc}")

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

        #optimizer = Adam(model.bert.parameters(), lr=lr)
        optimizerD = Adam(model.domain_classifier.parameters(), lr=lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            args.warmup_steps,
            n_epochs * sum([len(train_dl) for train_dl in train_dls])
        )

        opt = [optimizer, optimizerD]

        # Train
        train(
            model,
            train_dls,
            opt,
            scheduler,
            validation_evaluator,
            n_epochs,
            device,
            args.log_interval,
            model_dir=args.model_dir,
            gradient_accumulation=args.gradient_accumulation,
            domain_name=domain
        )
        # Load the best weights
        model.load_state_dict(torch.load(f'{args.model_dir}/files/model_{domain}.pth'))

        evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
        (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
            model,
            plot_callbacks=[plot_label_distribution],
            return_labels_logits=True,
            return_votes=True
        )
        print(f"{domain} F1: {F1}")
        print(f"{domain} Accuracy: {acc}")
        print()

        Ps.append(P)
        Rs.append(R)
        F1s.append(F1)
        accs.append(acc)
        labels_all.extend(labels)
        logits_all.extend(logits)
        with open(f'{args.model_dir}/files/pred_lab.txt', 'a+') as f:
            for p, l in zip(np.argmax(logits, axis=-1), labels):
                f.write(f'{domain}\t{p}\t{l}\n')

        test_dset.dataset = pd.DataFrame(test_dset.original_data)

    acc, P, R, F1 = acc_f1(logits_all, labels_all)

