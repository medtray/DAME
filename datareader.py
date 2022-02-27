from xml.dom import minidom
from typing import AnyStr
from typing import List
from typing import Tuple
import unicodedata
import pandas as pd
import json
import glob
import ipdb

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import random
import numpy as np

entity_matching_domain_map={
    'Abt-Buy':0,
    'Amazon-Google':1,
    'Beer':2,
    'cameras_':3,
    'computers_':4,
    'DBLP-ACM':5,
    'DBLP-GoogleScholar':6,
    'Fodors-Zagats':7,
    'iTunes-Amazon':8,
    'shoes_':9,
    'Walmart-Amazon':10,
    'watches_':11,
}

balance_dataset=False

def text_to_batch_transformer(text: List, tokenizer: PreTrainedTokenizer, text_pair: AnyStr = None) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :param: text_pair: An optional second string (for multiple sentence sequences)
    :return: A list of IDs and a mask
    """
    if text_pair is None:
        input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=tokenizer.max_len) for t in text]
    else:
        input_ids = [tokenizer.encode(t, text_pair=p, add_special_tokens=True, max_length=tokenizer.max_len) for t,p in zip(text, text_pair)]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def collate_batch_transformer(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]
    labels = [i[2] for i in input_data]
    domains = [i[3] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [0] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks), torch.tensor(labels), torch.tensor(domains)


def collate_batch_transformer_with_index(input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List]:
    return collate_batch_transformer(input_data) + ([i[-1] for i in input_data],)



def remove_attributes(entry,shuffle_cols):

    #print('initial entry')
    #print(entry)

    inter=entry.split('COL ')[1:]
    #print('after col')
    #print(inter)
    attributes = []
    values=[]

    for item in inter:
        inter2=item.split('VAL ')
        #print('col')
        #print(inter2)
        att=inter2[0].strip()
        value=inter2[1].strip()
        attributes.append(att)
        values.append(value)

    if shuffle_cols:
        indices = np.arange(len(attributes))
        np.random.shuffle(indices)
        attributes = [attributes[i] for i in indices]
        values = [values[i] for i in indices]
        seq = []
        for i, att in enumerate(attributes):
            seq.append('COL')
            seq.append(att)
            seq.append('VAL')
            seq.append(values[i])

    record=' '.join(seq)

    return record

def read_text(dir: AnyStr, domain: AnyStr, split: AnyStr = 'train'):

    pairs=[]
    remove_att=False
    shuffle_cols=False
    #print(domain)
    #if domain=='cameras_':
    #    print('here')
    with open(f'{dir}/{domain}/{split}.txt', encoding='utf8', errors='ignore') as f:
        for index,line in enumerate(f):
            #if index>3:
            #    break
            items = line.strip().split('\t')
            if remove_att:
                for index,item in enumerate(items[:2]):
                    items[index]=remove_attributes(item,shuffle_cols)

            pairs.append({'text': items[0] + ' [SEP] ' + items[1], 'label': int(items[2]), 'domain': entity_matching_domain_map[domain]})

    return pairs

def return_balanced_split(split):
    pos_instances = []
    neg_instances = []
    for item in split:
        if item['label'] == 0:
            neg_instances.append(item)
        else:
            pos_instances.append(item)

    random.shuffle(neg_instances)
    neg_instances = neg_instances[0:len(pos_instances)]
    new_split = pos_instances + neg_instances

    return new_split

class MultiDomainEntityMatchingDataset(Dataset):
    """
    Implements a dataset for the multidomain sentiment analysis dataset
    """
    def __init__(
            self,
            dataset_dir: AnyStr,
            domains: List,
            tokenizer: PreTrainedTokenizer,
            domain_ids: List = None
    ):
        """

        :param dataset_dir: The base directory for the dataset
        :param domains: The set of domains to load data for
        :param: tokenizer: The tokenizer to use
        :param: domain_ids: A list of ids to override the default domain IDs
        """
        super(MultiDomainEntityMatchingDataset, self).__init__()
        data = []
        split_indices={}
        for domain in domains:

            train_split=read_text(dataset_dir, domain, 'train')
            if balance_dataset:
                train_split=return_balanced_split(train_split)
            split_indices['train']=[0,len(train_split)]
            data.extend(train_split)

            valid_split=read_text(dataset_dir, domain, 'valid')
            valid_balanced_data=True
            if balance_dataset:
                valid_split=return_balanced_split(valid_split)
            split_indices['valid'] = [len(train_split),len(train_split)+len(valid_split)]
            data.extend(valid_split)


            test_split=read_text(dataset_dir, domain, 'test')
            test_data =test_split.copy()
            if balance_dataset:
                test_split = return_balanced_split(test_split)
            split_indices['test'] = [len(train_split)+len(valid_split),len(train_split)+len(valid_split)+len(test_split)]
            data.extend(test_split)

        self.dataset = pd.DataFrame(data)
        if domain_ids is not None:
            for i in range(len(domain_ids)):
                data[data['domain'] == entity_matching_domain_map[domains[i]]][2] = domain_ids[i]
        self.tokenizer = tokenizer
        self.split_indices=split_indices
        self.original_data=data
        self.test_data=test_data

    def set_domain_id(self, domain_id):
        """
        Overrides the domain ID for all data
        :param domain_id:
        :return:
        """
        self.dataset['domain'] = domain_id

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.values[item]
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        domain = row[2]
        return input_ids, mask, label, domain, item


