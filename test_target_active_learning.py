import sys
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
from datetime import datetime
from gurobipy import *

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
from sklearn.cluster import KMeans


def extract_embs(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_embs = []
        for batch in tqdm(dataloader, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            embs = model.return_embedding(input_ids, attention_mask=masks)
            all_embs.append(embs)

    all_embs = torch.cat(all_embs)

    return all_embs


def extract_confidences(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_confs = []
        for batch in tqdm(dataloader, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            confs = model.return_confidence(input_ids, attention_mask=masks)
            all_confs.append(confs)

    all_confs = torch.cat(all_confs)

    return all_confs

def extract_probs_dropout(model,dataloader,device,n_drop):

    model.train()
    dropout_confs=[]
    with torch.no_grad():
        for i in range(n_drop):
            all_embs=[]
            for batch in tqdm(dataloader, desc="Evaluation"):
                batch = tuple(t.to(device) for t in batch)
                input_ids = batch[0]
                masks = batch[1]
                confs = model.return_confidence(input_ids, attention_mask=masks)
                all_embs.append(confs)

            all_embs=torch.cat(all_embs)
            dropout_confs.append(all_embs)


    dropout_confs=torch.stack(dropout_confs,dim=1)
    dropout_confs=torch.sum(dropout_confs,dim=1)
    dropout_confs/=n_drop


    return dropout_confs

def least_confidence(model, dataloader_target, train_data, nb_samples, device):
    confs_target = extract_confidences(model, dataloader_target, device)
    U = confs_target.max(1)[0]
    idxs_unlabeled = np.arange(len(train_data))
    samples_to_query = idxs_unlabeled[U.sort()[1].cpu()[:nb_samples]]

    query_data = [train_data[jj] for jj in samples_to_query]

    return query_data

def random_samples(train_data,nb_samples):

    random.shuffle(train_data)
    query_data=train_data[:nb_samples]

    return query_data

def kmeans_sampling(model,dataloader_target,train_data,nb_samples,device):

    idxs_unlabeled = np.arange(len(train_data))
    embss_target = extract_embs(model, dataloader_target, device)
    embedding = embss_target.cpu().numpy()
    cluster_learner = KMeans(n_clusters=nb_samples)
    cluster_learner.fit(embedding)

    cluster_idxs = cluster_learner.predict(embedding)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embedding - centers) ** 2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(nb_samples)])

    samples_to_query=idxs_unlabeled[q_idxs]
    query_data = [train_data[jj] for jj in samples_to_query]

    return query_data

def entropy_sampling(model,dataloader_target,train_data,nb_samples,device):

    confs_target = extract_confidences(model, dataloader_target, device)
    log_probs = torch.log(confs_target)
    U = (confs_target * log_probs).sum(1)
    idxs_unlabeled = np.arange(len(train_data))
    samples_to_query = idxs_unlabeled[U.sort()[1].cpu()[:nb_samples]]

    query_data = [train_data[jj] for jj in samples_to_query]

    return query_data


def kcenters_sampling1(model,dataloader_target,train_data,nb_samples,device):

    NUM_INIT_LB=int(nb_samples/2)
    #NUM_INIT_LB = nb_samples
    nb_unlabeled=nb_samples-NUM_INIT_LB
    idxs_lb = np.zeros(len(train_data), dtype=bool)

    confs_target = extract_confidences(model, dataloader_target, device)
    U = confs_target.max(1)[0]
    idxs_unlabeled = np.arange(len(train_data))
    samples_to_query = idxs_unlabeled[U.sort(descending=False)[1].cpu()[:NUM_INIT_LB]]
    idxs_lb[samples_to_query] = True

    lb_flag = idxs_lb.copy()
    embss_target = extract_embs(model, dataloader_target, device)
    embedding = embss_target.cpu().numpy()

    from datetime import datetime

    #print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(train_data), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    #print(datetime.now() - t_start)

    mat = dist_mat[~lb_flag, :][:, lb_flag]

    for i in range(nb_unlabeled):
        #if i % 10 == 0:
        #    print('greedy solution {}/{}'.format(i, nb_samples))
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(len(train_data))[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

    samples_to_query=np.arange(len(train_data))[(idxs_lb | lb_flag)]
    query_data = [train_data[jj] for jj in samples_to_query]

    return query_data


def kcenters_sampling(model,dataloader_target,train_data,test_dset_supervised_training,optimizer,args,evaluator_valid,n_epochs,domain,test_dset,nb_samples,device):

    NUM_INIT_LB=int(nb_samples/2)
    #NUM_INIT_LB = nb_samples
    nb_unlabeled=nb_samples-NUM_INIT_LB
    idxs_lb = np.zeros(len(train_data), dtype=bool)

    # confs_target = extract_confidences(model, dataloader_target, device)
    # U = confs_target.max(1)[0]
    # idxs_unlabeled = np.arange(len(train_data))
    # samples_to_query = idxs_unlabeled[U.sort()[1].cpu()[:NUM_INIT_LB]]
    # idxs_lb[samples_to_query] = True

    indices=np.arange(len(train_data))
    np.random.shuffle(indices)
    samples_to_query=indices[:NUM_INIT_LB]
    idxs_lb[samples_to_query] = True

    query_data = [train_data[jj] for jj in samples_to_query]

    test_dset_supervised_training.dataset = pd.DataFrame(query_data)
    test_dset_supervised_training.set_domain_id(0)
    test_dset.set_domain_id(1)

    train_sizes = [int(len(test_dset_supervised_training) * args.train_pct)]
    val_sizes = [len(test_dset_supervised_training) - train_sizes[0]]
    subsets = [random_split(test_dset_supervised_training, [train_sizes[0], val_sizes[0]])]
    train_dls = [DataLoader(
        subset[0],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch_transformer
    ) for subset in subsets]

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
        model_dir=args.model_dir,
        gradient_accumulation=args.gradient_accumulation,
        domain_name=domain
    )
    model.load_state_dict(torch.load(f'{args.model_dir}/files/model_{domain}.pth'))
    evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
    (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
        model,
        plot_callbacks=[],
        return_labels_logits=True,
        return_votes=True
    )
    print('{} samples for {}'.format(NUM_INIT_LB, domain))
    print(f"{domain} F1: {F1}")
    print(f"{domain} Accuracy: {acc}")
    print()

    test_dset_supervised_training.dataset = pd.DataFrame(train_data)
    lb_flag = idxs_lb.copy()
    embss_target = extract_embs(model, dataloader_target, device)
    embedding = embss_target.cpu().numpy()

    from datetime import datetime

    #print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(train_data), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    #print(datetime.now() - t_start)

    mat = dist_mat[~lb_flag, :][:, lb_flag]

    for i in range(nb_unlabeled):
        #if i % 10 == 0:
        #    print('greedy solution {}/{}'.format(i, nb_unlabeled))
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(len(train_data))[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

    samples_to_query=np.arange(len(train_data))[(idxs_lb | lb_flag)]
    query_data = [train_data[jj] for jj in samples_to_query]

    return model,query_data

def solve_fac_loc(xx, yy, subset, n, budget):
    t_start = datetime.now()
    model = Model("k-center")
    x = {}
    y = {}
    z = {}

    print('gen z', datetime.now() - t_start)
    for i in range(n):
        # z_i: is a loss
        z[i] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(i))

    print('gen x y', datetime.now() - t_start)
    m = len(xx)
    for i in range(m):
        if i % 1000000 == 0:
            print('gen x y {}/{}'.format(i, m), datetime.now() - t_start)
        _x = xx[i]
        _y = yy[i]
        # y_i = 1 means i is facility, 0 means it is not
        if _y not in y:
            if _y in subset:
                y[_y] = model.addVar(obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
            else:
                y[_y] = model.addVar(obj=0, vtype="B", name="y_{}".format(_y))
        # if not _x == _y:
        x[_x, _y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x, _y))
    model.update()

    print('gen sum q', datetime.now() - t_start)
    coef = [1 for j in range(n)]
    var = [y[j] for j in range(n)]
    model.addConstr(LinExpr(coef, var), "=", rhs=budget + len(subset), name="k_center")

    print('gen <=', datetime.now() - t_start)
    for i in range(m):
        if i % 1000000 == 0:
            print('gen <= {}/{}'.format(i, m), datetime.now() - t_start)
        _x = xx[i]
        _y = yy[i]
        # if not _x == _y:
        model.addConstr(x[_x, _y], "<=", y[_y], name="Strong_{},{}".format(_x, _y))

    print('gen sum 1', datetime.now() - t_start)
    yyy = {}
    for v in range(m):
        if i % 1000000 == 0:
            print('gen sum 1 {}/{}'.format(i, m), datetime.now() - t_start)
        _x = xx[v]
        _y = yy[v]
        if _x not in yyy:
            yyy[_x] = []
        if _y not in yyy[_x]:
            yyy[_x].append(_y)

    for _x in yyy:
        coef = []
        var = []
        for _y in yyy[_x]:
            # if not _x==_y:
            coef.append(1)
            var.append(x[_x, _y])
        coef.append(1)
        var.append(z[_x])
        model.addConstr(LinExpr(coef, var), "=", 1, name="Assign{}".format(_x))

    print('ok', datetime.now() - t_start)
    model.__data = x, y, z
    return model

def core_set(model,dataloader_target,train_data,test_dset_supervised_training,optimizer,args,evaluator_valid,n_epochs,domain,test_dset,nb_samples,device):

    NUM_INIT_LB=int(nb_samples/2)
    #NUM_INIT_LB = nb_samples
    nb_unlabeled=nb_samples-NUM_INIT_LB
    idxs_lb = np.zeros(len(train_data), dtype=bool)

    # confs_target = extract_confidences(model, dataloader_target, device)
    # U = confs_target.max(1)[0]
    # idxs_unlabeled = np.arange(len(train_data))
    # samples_to_query = idxs_unlabeled[U.sort()[1].cpu()[:NUM_INIT_LB]]
    # idxs_lb[samples_to_query] = True

    indices=np.arange(len(train_data))
    np.random.shuffle(indices)
    samples_to_query=indices[:NUM_INIT_LB]
    idxs_lb[samples_to_query] = True

    query_data = [train_data[jj] for jj in samples_to_query]

    test_dset_supervised_training.dataset = pd.DataFrame(query_data)
    test_dset_supervised_training.set_domain_id(0)
    test_dset.set_domain_id(1)

    train_sizes = [int(len(test_dset_supervised_training) * args.train_pct)]
    val_sizes = [len(test_dset_supervised_training) - train_sizes[0]]
    subsets = [random_split(test_dset_supervised_training, [train_sizes[0], val_sizes[0]])]
    train_dls = [DataLoader(
        subset[0],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch_transformer
    ) for subset in subsets]

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
        model_dir=args.model_dir,
        gradient_accumulation=args.gradient_accumulation,
        domain_name=domain
    )
    model.load_state_dict(torch.load(f'{args.model_dir}/files/model_{domain}.pth'))
    evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
    (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
        model,
        plot_callbacks=[],
        return_labels_logits=True,
        return_votes=True
    )
    print('{} samples for {}'.format(NUM_INIT_LB, domain))
    print(f"{domain} F1: {F1}")
    print(f"{domain} Accuracy: {acc}")
    print()

    test_dset_supervised_training.dataset = pd.DataFrame(train_data)
    lb_flag = idxs_lb.copy()
    embss_target = extract_embs(model, dataloader_target, device)
    embedding = embss_target.cpu().numpy()

    from datetime import datetime

    #print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(train_data), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    #print(datetime.now() - t_start)

    mat = dist_mat[~lb_flag, :][:, lb_flag]

    for i in range(nb_unlabeled):
        if i % 10 == 0:
            print('greedy solution {}/{}'.format(i, nb_unlabeled))
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(len(train_data))[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

    print(datetime.now() - t_start)
    opt = mat.min(axis=1).max()

    bound_u = opt
    bound_l = opt / 2.0
    delta = opt

    xx, yy = np.where(dist_mat <= opt)
    dd = dist_mat[xx, yy]

    lb_flag_ = idxs_lb.copy()
    subset = np.where(lb_flag_ == True)[0].tolist()

    SEED = 5

    pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), nb_unlabeled, len(train_data)),
                open('mip{}.pkl'.format(SEED), 'wb'), 2)


    #ipdb.set_trace()

    r_name = 'mip{}.pkl'.format(SEED)
    w_name = 'sols{}.pkl'.format(SEED)
    print('load pickle {}'.format(r_name))
    xx, yy, dd, subset, max_dist, budget, n = pickle.load(open(r_name, 'rb'))
    print(len(subset), budget, n)

    t_start = datetime.now()

    print('start')
    ub = max_dist
    lb = ub / 2.0

    model_coreset = solve_fac_loc(xx, yy, subset, n, budget)

    # model.setParam( 'OutputFlag', False )
    x, y, z = model_coreset.__data
    tor = 1e-3
    sols = None
    while ub - lb > tor:
        cur_r = (ub + lb) / 2.0
        print("======[State]======", ub, lb, cur_r, ub - lb)

        # viol = numpy.where(_d>cur_r)
        viol = [i for i in range(len(dd)) if dd[i] > cur_r]
        # new_max_d = numpy.min(_d[_d>=cur_r])
        new_max_d = min([d for d in dd if d >= cur_r])
        # new_min_d = numpy.max(_d[_d<=cur_r])
        new_min_d = max([d for d in dd if d <= cur_r])
        print("If it succeeds, new max is:", new_max_d, new_min_d)
        for v in viol:
            x[xx[v], yy[v]].UB = 0

        model_coreset.update()
        r = model_coreset.optimize()
        if model_coreset.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
            failed = True
            print("======[Infeasible]======")
        elif sum([z[i].X for i in range(len(z))]) > 0:
            failed = True
            print("======[Failed]======Failed")
        else:
            failed = False
        if failed:
            lb = max(cur_r, new_max_d)
            # failed so put edges back
            for v in viol:
                x[xx[v], yy[v]].UB = 1
        else:
            print("======[Solution Founded]======", ub, lb, cur_r)
            ub = min(cur_r, new_min_d)
            sols = [v.varName for v in model_coreset.getVars() if v.varName.startswith('y') and v.x > 0]
            #break

    print('end', datetime.now() - t_start, ub, lb, max_dist)

    if sols is not None:
        sols = [int(v.split('_')[-1]) for v in sols]
    print('save pickle {}'.format(w_name))
    pickle.dump(sols, open(w_name, 'wb'), 2)

    sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))

    if sols is None:
        q_idxs = lb_flag
    else:
        lb_flag_[sols] = True
        q_idxs = lb_flag_
    print('sum q_idxs = {}'.format(q_idxs.sum()))


    samples_to_query = np.arange(len(train_data))[(idxs_lb | q_idxs)]
    query_data = [train_data[jj] for jj in samples_to_query]

    print('nb samples={}'.format(len(query_data)))

    return model,query_data

def USDE(model,dataloader_target,train_data,nb_samples,n_drop,device):

    confs_target = extract_probs_dropout(model, dataloader_target, device, n_drop)
    log_probs = torch.log(confs_target)
    U = (confs_target * log_probs).sum(1)
    idxs_unlabeled = np.arange(len(train_data))
    samples_to_query = idxs_unlabeled[U.sort()[1].cpu()[:nb_samples]]

    query_data = [train_data[jj] for jj in samples_to_query]

    return query_data

def extract_probs_split(model,dataloader,device,n_drop):

    model.train()
    dropout_confs=[]
    with torch.no_grad():
        for i in range(n_drop):
            all_embs=[]
            for batch in tqdm(dataloader, desc="Evaluation"):
                batch = tuple(t.to(device) for t in batch)
                input_ids = batch[0]
                masks = batch[1]
                confs = model.return_confidence(input_ids, attention_mask=masks)
                all_embs.append(confs)

            all_embs=torch.cat(all_embs)
            dropout_confs.append(all_embs)


    dropout_confs=torch.stack(dropout_confs,dim=0)

    return dropout_confs

def BALD(model,dataloader_target,train_data,nb_samples,n_drop,device):

    confs_target = extract_probs_split(model, dataloader_target, device, n_drop)
    pb = confs_target.mean(0)
    entropy1 = (-pb * torch.log(pb)).sum(1)
    entropy2 = (-confs_target * torch.log(confs_target)).sum(2).mean(0)
    U = entropy2 - entropy1
    idxs_unlabeled = np.arange(len(train_data))
    samples_to_query = idxs_unlabeled[U.sort()[1].cpu()[:nb_samples]]

    query_data = [train_data[jj] for jj in samples_to_query]

    return query_data

def USDE_BALD(model,dataloader_target,train_data,nb_samples,n_drop,device):

    confs_target = extract_probs_dropout(model, dataloader_target, device, n_drop)
    log_probs = torch.log(confs_target)
    U1 = (confs_target * log_probs).sum(1)

    confs_target = extract_probs_split(model, dataloader_target, device, n_drop)
    pb = confs_target.mean(0)
    entropy1 = (-pb * torch.log(pb)).sum(1)
    entropy2 = (-confs_target * torch.log(confs_target)).sum(2).mean(0)
    U = entropy2 - entropy1
    U=U+U1
    idxs_unlabeled = np.arange(len(train_data))
    samples_to_query = idxs_unlabeled[U.sort()[1].cpu()[:nb_samples]]

    query_data = [train_data[jj] for jj in samples_to_query]

    return query_data

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
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training',
                        default=['Walmart-Amazon', 'Abt-Buy', 'Beer', 'DBLP-GoogleScholar', 'Amazon-Google', 'cameras_',
                                 'DBLP-ACM', 'Fodors-Zagats', 'iTunes-Amazon', 'shoes_', 'computers_', 'watches_'])

    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--model_dir", help="Where to store the saved model", default="moe_dame", type=str)
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

    parser.add_argument("--active_learning", help="active learning strategies", default="random sampling", type=str)

    args = parser.parse_args()

    active_learning_strategy = args.active_learning

    valid_als=['random sampling','least confidence', 'entropy sampling', 'usde', 'bald', 'k-centers', 'k-means', 'core-set']

    if active_learning_strategy not in valid_als:
        print('choose AL strategy from the list {}'.format(valid_als))
        exit()

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
    batch_size = 8#args.batch_size
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

    all_results = {}

    for i in range(len(all_dsets)):
        domain = args.domains[i]
        test_dset = all_dsets[i]
        test_dset_supervised_training = copy.deepcopy(test_dset)
        valid_dset = copy.deepcopy(test_dset)
        test_indices = test_dset.split_indices['test']
        train_indices = test_dset.split_indices['train']
        valid_indices = test_dset.split_indices['valid']
        test_dset.dataset = pd.DataFrame(test_dset.test_data)

        train_data = test_dset.original_data[train_indices[0]:train_indices[1]]
        valid_data = test_dset.original_data[valid_indices[0]:valid_indices[1]]
        valid_dset.dataset = pd.DataFrame(valid_data)

        trials = 2
        ratio = [0.05, 0.25]

        for trial in range(trials):

            random.shuffle(train_data)

            model.load_state_dict(torch.load(f"{args.model_dir}/files/model_{domain}.pth"))

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

            all_f1 = []
            all_sizes = []
            all_acc = []
            all_precision = []
            all_recall = []

            all_f1.append(F1)
            all_acc.append(acc)
            all_precision.append(P)
            all_recall.append(R)
            all_sizes.append(0)

            for i in range(len(ratio)):

                model.load_state_dict(torch.load(f"{args.model_dir}/files/model_{domain}.pth"))

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
                    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0}
                ]

                optimizer_grouped_parameters[0]['params'] = optimizer_grouped_parameters[0]['params'][:-1]
                optimizer_grouped_parameters[1]['params'] = optimizer_grouped_parameters[1]['params'][:-1]

                optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

                nb_samples = int(ratio[i] * len(train_data))

                all_sizes.append(nb_samples)

                test_dset_supervised_training.dataset = pd.DataFrame(train_data)

                dataloader_target = DataLoader(
                    test_dset_supervised_training,
                    batch_size=8,
                    shuffle=False,
                    collate_fn=collate_batch_transformer
                )


                if active_learning_strategy=='random sampling':
                    query_data = random_samples(train_data, nb_samples)

                elif active_learning_strategy=='least confidence':
                    query_data = least_confidence(model, dataloader_target, train_data,nb_samples, device)

                elif active_learning_strategy == 'entropy sampling':
                    query_data = entropy_sampling(model, dataloader_target, train_data, nb_samples, device)

                elif active_learning_strategy == 'k-means':
                    query_data = kmeans_sampling(model, dataloader_target, train_data, nb_samples, device)

                elif active_learning_strategy == 'k-centers':
                    #query_data = kcenters_sampling1(model, dataloader_target, train_data, nb_samples, device)
                    model, query_data = kcenters_sampling(model, dataloader_target, train_data,test_dset_supervised_training, optimizer,args,evaluator_valid,n_epochs,domain,test_dset, nb_samples, device)

                elif active_learning_strategy == 'core-set':
                    #query_data = core_set(model, dataloader_target, train_data, nb_samples, device)
                    model, query_data = core_set(model, dataloader_target, train_data, test_dset_supervised_training, optimizer, args, evaluator_valid, n_epochs, domain, test_dset, nb_samples, device)

                elif active_learning_strategy == 'usde':
                    query_data = USDE(model, dataloader_target, train_data, nb_samples, 5, device)

                elif active_learning_strategy == 'bald':
                    query_data = BALD(model, dataloader_target, train_data, nb_samples, 10, device)

                else:

                    exit()

                #query_data = USDE_BALD(model, dataloader_target, train_data, nb_samples, 10, device)

                test_dset_supervised_training.dataset = pd.DataFrame(query_data)

                test_dset_supervised_training.set_domain_id(0)
                test_dset.set_domain_id(1)
                train_sizes = [int(len(test_dset_supervised_training) * args.train_pct)]
                val_sizes = [len(test_dset_supervised_training) - train_sizes[0]]

                subsets = [random_split(test_dset_supervised_training, [train_sizes[0], val_sizes[0]])]

                train_dls = [DataLoader(
                    subset[0],
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_batch_transformer
                ) for subset in subsets]



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
                    model_dir=args.model_dir,
                    gradient_accumulation=args.gradient_accumulation,
                    domain_name=domain
                )
                # Load the best weights
                model.load_state_dict(torch.load(f'{args.model_dir}/files/model_{domain}.pth'))

                evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
                (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
                    model,
                    plot_callbacks=[],
                    return_labels_logits=True,
                    return_votes=True
                )
                print('{} samples for {}'.format(nb_samples,domain))
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

        np.save('active_learning_results.npy',all_results)
