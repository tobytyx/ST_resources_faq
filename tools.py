# -*- coding: utf-8 -*-
from transformers import BertTokenizer
from model import BiEncoder, PolyEncoder
from dataset import TestDataset
from torch.utils.data import DataLoader
import torch


def create_model(args, device):
    tokenizer = BertTokenizer.from_pretrained(args["pre_model"])
    if args["model_name"] == "bi":
        model = BiEncoder(args)
    else:
        model = PolyEncoder(args, device)
    return model, tokenizer
    

def create_data_vec(data_list, model, tokenizer, max_length, device, mode, with_label):
    dataset = TestDataset(data_list, tokenizer, max_length, with_label)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=dataset.collate_fn)
    data_vecs, labels = [], []
    for batch in dataloader:
        queries, q_mask, batch_labels = batch
        queries, q_mask = queries.to(device), q_mask.to(device)
        if mode == "query":
            batch_vecs = model.encode_query(queries, q_mask)
        else:
            batch_vecs = model.encode_response(queries, q_mask)
        batch_vecs = batch_vecs.detach().cpu()
        data_vecs.append(batch_vecs)
        labels.extend(batch_labels)
    data_vecs = torch.cat(data_vecs, dim=0)
    assert len(data_vecs) == len(labels)
    return data_vecs, labels


def get_tsv_data(data_file):
    data = []
    with open(data_file) as f:
        for line in f:
            each = line.strip().split("\t")
            try:
                each[-1] = int(each[-1])
            except ValueError:
                raise
            data.append(each)
    return data
