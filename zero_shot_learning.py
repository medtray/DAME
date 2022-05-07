import argparse
import gc
import os
import random
from tqdm import tqdm
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer

from datareader import MultiDomainEntityMatchingDataset
from metrics import ClassificationEvaluator
from metrics import acc_f1

from model import VanillaBert
from model import *
import pandas as pd
import copy


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_loc", help="Root directory of the dataset", required=False, type=str,
                        default='entity-matching-dataset')
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=1)
    parser.add_argument("--pretrained_multi_xformer",help="Directory with trained models", type=str,default='moe_dame')
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training', default=['Walmart-Amazon','Abt-Buy','Beer','DBLP-GoogleScholar','Amazon-Google','cameras_','DBLP-ACM','Fodors-Zagats','iTunes-Amazon','shoes_','computers_','watches_'])
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
    parser.add_argument("--model", help="Name of the model to run", default="VanillaBert")

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
        device = torch.device("cuda:0")

    # model configuration
    bert_model = 'distilbert-base-uncased'
    ##################
    # override for now
    batch_size = args.batch_size
    ###############
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
        valid_dset = copy.deepcopy(test_dset)
        test_indices = test_dset.split_indices['test']
        train_indices = test_dset.split_indices['train']
        valid_indices = test_dset.split_indices['valid']
        test_dset.dataset=pd.DataFrame(test_dset.test_data)

        train_data=test_dset.original_data[train_indices[0]:train_indices[1]]
        valid_data=test_dset.original_data[valid_indices[0]:valid_indices[1]]
        valid_dset.dataset=pd.DataFrame(valid_data)

        model.load_state_dict(torch.load(f"{args.pretrained_multi_xformer}/files/model_{domain}.pth"))

        evaluator = ClassificationEvaluator(test_dset, device, use_domain=False)
        (loss, acc, P, R, F1), plots, (labels, logits), votes = evaluator.evaluate(
            model,
            plot_callbacks=[],
            return_labels_logits=True,
            return_votes=True
        )
        print('zero-shot learning results for {}'.format(domain))
        print(f"{domain} F1: {F1}")
        print(f"{domain} Accuracy: {acc}")
        print()


        all_results[domain]={'F1':F1,'acc':acc, 'precision':P, 'recall':R}

    print(all_results)
