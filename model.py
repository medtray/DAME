import os
from tqdm import tqdm
import torch
from copy import deepcopy
from typing import List
from torch import nn
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertModel
import ipdb
import numpy as np
from torch.nn import functional as F
from transformers import PreTrainedTokenizer
from torch.nn import init
from argparse import Namespace
import math

device = torch.device("cuda:1")


class GradientReversal(torch.autograd.Function):
    """
    Basic layer for doing gradient reversal
    """
    lambd = 1.0
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReversal.lambd * grad_output.neg()

class VanillaBert(nn.Module):
    """
    A really basic wrapper around BERT
    """
    def __init__(self, bert: BertForSequenceClassification, **kwargs):
        super(VanillaBert, self).__init__()

        self.bert = bert

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None):

        return self.bert(input_ids, attention_mask=attention_mask, labels=labels)



class MultiTransformer(nn.Module):
    """
        Multiple transformers for different domains
        """

    def __init__(
            self,
            bert_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_domains: int = 2,
            n_layers: int = 6,
            n_heads: int = 6,
    ):
        super(MultiTransformer, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model
        self.n_domains = n_domains

        self.bert_embeddings = bert_embeddings

        self.xformer = nn.ModuleList([nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=ff_dim
            ),
            n_layers
        ) for d in range(n_domains)])

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None
    ):
        embs = self.bert_embeddings(input_ids=input_ids)

        # Sequence length first
        inputs = embs.permute(1, 0, 2)
        # Flags the 0s instead of 1s
        masks = attention_mask == 0

        if domains is not None:
            domain = domains[0]
            output = self.xformer[domain](inputs, src_key_padding_mask=masks)
            pooled_output = output[0]
            return pooled_output

        else:
            pooled_outputs = []
            for d in range(self.n_domains):
                output = self.xformer[d](inputs, src_key_padding_mask=masks)
                pooled_outputs.append(output[0])
            return pooled_outputs


class MultiTransformerClassifier(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            bert_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_domains: int = 2,
            n_layers: int = 6,
            n_classes: int = 2,
            n_heads: int = 6,
    ):
        super(MultiTransformerClassifier, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model
        self.n_domains = n_domains

        self.multi_xformer = MultiTransformer(
            bert_embeddings,
            ff_dim=ff_dim,
            d_model=d_model,
            n_domains=n_domains,
            n_layers=n_layers,
            n_heads=n_heads
        )

        # final classifier layers (d_model x n_classes)
        self.classifier = nn.ModuleList([nn.Linear(d_model, n_classes) for d in range(n_domains)])

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):

        if domains is not None:
            domain = domains[0]
            pooled_output = self.multi_xformer(input_ids, attention_mask, domains)
            logits = self.classifier[domain](pooled_output)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, logits)

        else:
            logits_all = []
            pooled_outputs = self.multi_xformer(input_ids, attention_mask)
            for d,po in enumerate(pooled_outputs):
                logits_all.append(self.classifier[d](po))

            loss_fn = nn.CrossEntropyLoss()
            loss = torch.mean(torch.stack([loss_fn(logits, labels) for logits in logits_all]))
            # b x ndom x 2
            votes = torch.stack(logits_all, dim=1)
            self.votes = votes
            # Normalize with softmax
            logits_all = torch.nn.Softmax(dim=-1)(votes)
            logits = torch.mean(logits_all, dim=1)

            return loss, logits



class MultiDistilBert(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            model_name,
            config,
            n_domains: int = 2,
            init_weights: List = None
    ):
        super(MultiDistilBert, self).__init__()
        self.models = nn.ModuleList([DistilBertModel.from_pretrained(model_name, config=config) for d in range(n_domains)])
        if init_weights is not None:
            if 'distilbert' in list(init_weights.keys())[0]:
                init_weights = {k[11:]: v for k,v in init_weights.items()}
            for m in self.models:
                model_dict = m.state_dict()
                model_dict.update(deepcopy(init_weights))
                m.load_state_dict(model_dict)
        self.n_domains = n_domains
        self.d_model = config.hidden_size

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):

        if domains is not None:
            domain = domains[0]
            outputs = self.models[domain](input_ids, attention_mask=attention_mask)
            return outputs[0][:,0,:]

        else:
            pooled_outputs = []
            for d in range(self.n_domains):
                output = self.models[d](input_ids, attention_mask=attention_mask)
                pooled_outputs.append(output[0][:,0,:])
            return pooled_outputs


class MultiDistilBertClassifier(nn.Module):
    """
    Multiple transformers for different domains
    """

    def __init__(
            self,
            model_name,
            config,
            n_domains: int = 2,
            n_classes: int = 2,
            init_weights: List = None
    ):
        super(MultiDistilBertClassifier, self).__init__()
        self.multi_xformer = MultiDistilBert(model_name, config, n_domains=n_domains, init_weights=init_weights)
        self.n_domains = n_domains
        self.d_model = config.hidden_size
        # final classifier layers (d_model x n_classes)
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, n_classes) for d in range(n_domains)])
        #base_classifier=nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),nn.ReLU(),nn.Dropout(0.2),nn.Linear(config.hidden_size, n_classes))
        #self.classifier=nn.ModuleList([base_classifier for d in range(n_domains)])

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None
    ):

        if domains is not None:
            domain = domains[0]
            output = self.multi_xformer(input_ids, attention_mask=attention_mask, domains=domains)
            logits = self.classifier[domain](output)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, logits)

        else:
            logits_all = []
            pooled_outputs = self.multi_xformer(input_ids, attention_mask=attention_mask)
            for d,po in enumerate(pooled_outputs):
                logits_all.append(self.classifier[d](po))

            loss_fn = nn.CrossEntropyLoss()
            loss = torch.mean(torch.stack([loss_fn(logits, labels) for logits in logits_all]))
            # b x ndom x 2
            votes = torch.stack(logits_all, dim=1)
            self.votes = votes
            # Normalize with softmax
            logits_all = torch.nn.Softmax(dim=-1)(votes)
            logits = torch.mean(logits_all, dim=1)

            return loss, logits


class MultiViewTransformerNetworkDomainAdversarial(nn.Module):
    """
    Multi-view transformer network for domain adaptation
    """
    def __init__(self, multi_xformer: MultiTransformerClassifier, shared_bert: VanillaBert, n_classes: int = 2, n_domains: int = 3, supervision_layer: int = 12):
        super(MultiViewTransformerNetworkDomainAdversarial, self).__init__()

        self.multi_xformer = multi_xformer.multi_xformer
        #self.shared_bert = shared_bert.bert.bert
        self.shared_bert = shared_bert.bert
        self.multi_xformer_classifiers = multi_xformer.classifier

        self.d_model = multi_xformer.d_model
        self.dim_param = nn.Parameter(torch.FloatTensor([multi_xformer.d_model]), requires_grad=False)
        self.n_xformers = multi_xformer.n_domains
        self.n_classes = n_classes
        self.n_domains = multi_xformer.n_domains
        self.supervision_layer = supervision_layer
        #self.MultiHeadAttention = MultiHeadAttention(1, self.d_model)

        # Query matrix
        self.Q = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Key matrix
        self.K = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)
        # Value matrix
        self.V = nn.Parameter(torch.randn((multi_xformer.d_model, multi_xformer.d_model)), requires_grad=True)

        # Main classifier
        self.domain_classifier = nn.Linear(self.d_model, self.n_domains+1)
        #self.domain_classifier=nn.Sequential(nn.Linear(self.d_model, self.d_model),nn.ReLU(),nn.Dropout(0.2),nn.Linear(self.d_model, n_domains+1))
        self.task_classifier = nn.Linear(2*multi_xformer.d_model, n_classes)
        # TODO: Introduce aux tasks if needed

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        logits_shared = outputs[0]
        shared_output = outputs[1][-1][:,0,:]
        divisor = min(1, 2 * (len(outputs[1]) - self.supervision_layer))

        if domains is not None and self.training:
            pooled_outputs_to_use = [l for j, l in enumerate(pooled_outputs) if j != domains[0]]
        else:
            pooled_outputs_to_use = pooled_outputs

        #pooled_outputs_to_use = pooled_outputs

        #nb_sources=len(pooled_outputs_to_use)+1
        nb_sources = len(pooled_outputs_to_use)

        # Values b x n_domain + 1 x dim
        #v = torch.stack(pooled_outputs_to_use + [shared_output], dim=1)
        v = torch.stack(pooled_outputs_to_use, dim=1)
        #yy = self.MultiHeadAttention(v, v, v)
        #o = torch.mean(yy, dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, nb_sources, self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)
        v = v.view(-1, self.d_model) @ self.V
        v = v.view(-1, nb_sources, self.d_model)
        # Attend to the values b x dim
        o = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        feat=torch.cat([o,shared_output],dim=1)

        # Classifier
        #logits = self.task_classifier(o)
        logits = self.task_classifier(feat)

        # Domain adversarial bit
        domain_supervision_layer = outputs[1][self.supervision_layer][:,0,:]
        #adv_input = GradientReversal.apply(domain_supervision_layer)
        adv_logits = self.domain_classifier(domain_supervision_layer)

        outputs = (logits,)

        #loss_fn = nn.CrossEntropyLoss()
        # if domains is not None:
        #     # Scale the adversarial loss depending on how deep in the network it is
        #     loss = (1e-3 / divisor) * loss_fn(adv_logits, domains)
        #     if labels is not None:
        #         loss += loss_fn(logits, labels)
        #     outputs = (loss,) + outputs
        # elif labels is not None:
        #     loss = loss_fn(logits, labels)
        #     outputs = (loss,) + outputs

        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.CrossEntropyLoss()
            loss = (1.0) * loss_fn(logits, labels)+ (1.0)*loss_fn(logits_shared, labels)
            # Strong supervision on in domain
            if domains is not None:
                domain = domains[0]
                domain_logits = logits_private[domain]
                xent = nn.CrossEntropyLoss()
                loss += 1.0 * xent(domain_logits, labels)
                # Scale the adversarial loss depending on how deep in the network it is
                #print(domains)
                #print(adv_logits)
                loss += -(1e-3 / divisor) * xent(adv_logits, domains)

            outputs = (loss,) + outputs

        elif domains is not None:
            domain = domains[0]
            xent = nn.CrossEntropyLoss()
            # Scale the adversarial loss depending on how deep in the network it is
            loss = -(1e-3 / divisor) * xent(adv_logits, domains)
            outputs = (loss,) + outputs

        if ret_alpha:
            outputs += (attn,)
            #outputs += ([],)

        #outputs += (o,)

        return outputs

    def forward_test(self,input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        shared_output = outputs[1][-1][:,0,:]

        pooled_outputs_to_use = pooled_outputs

        #nb_sources=len(pooled_outputs_to_use)+1
        nb_sources = len(pooled_outputs_to_use)

        # Values b x n_domain + 1 x dim
        #v = torch.stack(pooled_outputs_to_use + [shared_output], dim=1)
        v = torch.stack(pooled_outputs_to_use, dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, nb_sources, self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)
        v = v.view(-1, self.d_model) @ self.V
        v = v.view(-1, nb_sources, self.d_model)
        # Attend to the values b x dim
        o = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        # Classifier
        logits = self.task_classifier(o)

        outputs = (logits,)

        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.CrossEntropyLoss()
            loss = (1) * loss_fn(logits, labels)

            outputs = (loss,) + outputs

        if ret_alpha:
            outputs += (attn,)

        return outputs

    def forward_test_drop_experts(self,input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            domains: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            ret_alpha: bool = False,
            list_experts: np.array=np.array([0,1,2])
    ):
        # Get all the pooled outputs
        # (n_domain) b x dim
        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        logits_private = [self.multi_xformer_classifiers[d](pooled_outputs[d]) for d in range(self.n_domains)]
        # b x dim
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        shared_output = outputs[1][-1][:,0,:]

        pooled_outputs_to_use = pooled_outputs


        pooled_outputs_to_use=[pooled_outputs_to_use[ii] for ii in list_experts]

        nb_sources=len(pooled_outputs_to_use)+0

        # Values b x n_domain + 1 x dim
        #v = torch.stack(pooled_outputs_to_use + [shared_output], dim=1)
        v = torch.stack(pooled_outputs_to_use, dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, nb_sources, self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)
        v = v.view(-1, self.d_model) @ self.V
        v = v.view(-1, nb_sources, self.d_model)
        # Attend to the values b x dim
        o = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        # Classifier
        logits = self.task_classifier(o)

        outputs = (logits,)

        if labels is not None:
            # LogSoftmax + NLLLoss
            loss_fn = nn.CrossEntropyLoss()
            loss = (1) * loss_fn(logits, labels)

            outputs = (loss,) + outputs

        if ret_alpha:
            outputs += (attn,)

        return outputs

    def return_embedding(self,input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor):

        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        shared_output = outputs[1][-1][:, 0, :]

        pooled_outputs_to_use = pooled_outputs

        nb_sources = len(pooled_outputs_to_use) + 1

        # Values b x n_domain + 1 x dim
        v = torch.stack(pooled_outputs_to_use + [shared_output], dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, nb_sources, self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)
        v = v.view(-1, self.d_model) @ self.V
        v = v.view(-1, nb_sources, self.d_model)
        # Attend to the values b x dim
        o = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        return o

    def return_confidence(self,input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor):

        pooled_outputs = self.multi_xformer(input_ids, attention_mask)
        outputs = self.shared_bert(input_ids, attention_mask=attention_mask)
        shared_output = outputs[1][-1][:, 0, :]

        pooled_outputs_to_use = pooled_outputs

        nb_sources = len(pooled_outputs_to_use) + 1

        # Values b x n_domain + 1 x dim
        v = torch.stack(pooled_outputs_to_use + [shared_output], dim=1)
        # Queries b x dim
        q = shared_output @ self.Q
        # Keys b*(n_domain + 1) x dim
        k = v.view(-1, self.d_model) @ self.K
        # Attention (scaled dot product) b x (n_domain + 1)
        attn = torch.sum(k.view(-1, nb_sources, self.d_model) * q.unsqueeze(1), dim=-1) / torch.sqrt(self.dim_param)
        attn = nn.Softmax(dim=-1)(attn)
        v = v.view(-1, self.d_model) @ self.V
        v = v.view(-1, nb_sources, self.d_model)
        # Attend to the values b x dim
        o = torch.sum(attn.unsqueeze(-1) * v, dim=1)

        logits = self.task_classifier(o)
        softmax=nn.Softmax(dim=1)
        probs=softmax(logits)

        return probs