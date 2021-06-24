# -*- coding: utf-8 -*-
from typing import Dict
import torch
from torch.utils.data import Dataset
import random


class EncoderDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, cnt):
        super(EncoderDataset, self).__init__()
        self.data = []
        self.extra_negative_samples = []
        self.clusters = {}
        self.cnt = cnt
        for each in data:
            text, tid = each[0], each[1]
            if tid == 0:
                self.extra_negative_samples.append(text)
                continue
            self.data.append(each)
            if tid not in self.clusters:
                self.clusters[tid] = [text]
            else:
                self.clusters[tid].append(text)
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        CLS, SEP = self.tokenizer.cls_token, self.tokenizer.sep_token
        query, tid = self.data[idx]
        query = [CLS] + self.tokenizer.tokenize(query)[:self.max_length - 2] + [SEP]
        query = self.tokenizer.convert_tokens_to_ids(query)

        responses = []
        positive_response = random.choice(self.clusters[tid])
        positive_response = [CLS] + self.tokenizer.tokenize(positive_response)[:self.max_length-2] + [SEP]
        positive_response = self.tokenizer.convert_tokens_to_ids(positive_response)
        responses.append(positive_response)
        cnt = self.cnt - 1
        if len(self.extra_negative_samples) > 0:
            cnt -= 1
            neg_response = random.choice(self.extra_negative_samples)
            neg_response = [CLS] + self.tokenizer.tokenize(neg_response)[:self.max_length-2] + [SEP]
            neg_response = self.tokenizer.convert_tokens_to_ids(neg_response)
            responses.append(neg_response)

        for _ in range(cnt):
            neg_response = get_negative_sample(self.clusters, tid)
            neg_response = [CLS] + self.tokenizer.tokenize(neg_response)[:self.max_length-2] + [SEP]
            neg_response = self.tokenizer.convert_tokens_to_ids(neg_response)
            responses.append(neg_response)
        return query, responses

    def collate_fn(self, data):
        queries, responses = zip(*data)
        pad_id = self.pad_id
        max_len = max(len(x) for x in queries)
        q_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in queries], dtype=torch.long)
        queries = torch.tensor([x + [pad_id] * (max_len - len(x)) for x in queries], dtype=torch.long)

        max_len = 0
        for res in responses:
            max_len = max(max_len, max(len(x) for x in res))
        r_mask = torch.tensor([
            [[1] * len(x) + [0] * (max_len - len(x)) for x in res] for res in responses], dtype=torch.long)
        responses = torch.tensor([
            [x + [pad_id] * (max_len - len(x)) for x in res] for res in responses], dtype=torch.long)
        return queries, q_mask, responses, r_mask


def get_negative_sample(clusters: Dict, tid: int):
    tids = set(list(clusters.keys()))
    tids.remove(tid)
    tids = list(tids)
    # assert len(tids) > 0, "no negative clusters"
    neg_tid = random.choice(tids)
    return random.choice(clusters[neg_tid])


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, with_label=False):
        super(TestDataset, self).__init__()
        self.data = data
        self.with_label = with_label
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        CLS, SEP = self.tokenizer.cls_token, self.tokenizer.sep_token
        if self.with_label:
            query, label = self.data[idx]
        else:
            query, label = self.data[idx], 0
        query = [CLS] + self.tokenizer.tokenize(query)[:self.max_length - 2] + [SEP]
        query = self.tokenizer.convert_tokens_to_ids(query)
        return query, label
    
    def collate_fn(self, data):
        queries, labels = zip(*data)
        pad_id = self.pad_id
        max_len = max(len(x) for x in queries)
        q_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in queries], dtype=torch.long)
        queries = torch.tensor([x + [pad_id] * (max_len - len(x)) for x in queries], dtype=torch.long)
        return queries, q_mask, labels
