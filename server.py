# -*- coding: utf-8 -*-
from tools import create_model
import torch
import pickle
import os
from fastapi import FastAPI

args = {
    "patent": {
        "pre_model": "./dependences/roberta_wwm_ext",
        "model_name": "bi",
        "model_dir": "./output/bi_patent"
    },
    "expert": {
        "pre_model": "./dependences/roberta_wwm_ext",
        "model_name": "bi",
        "model_dir": "./output/bi_expert"
    },
    "equipment": {
        "pre_model": "./dependences/roberta_wwm_ext",
        "model_name": "bi",
        "model_dir": "./output/bi_equipment"
    }
}
max_length = 64
device = torch.device("cpu")
ERROR_LABEL = 0
MAX_LIST_NUM = 3
ENSURE_THRESHOLD = 0.92


def get_state_map():
    state_map = {}
    for domain, param in args.items():
        # documents
        with open(os.path.join(param["model_dir"], "total.pkl"), mode="rb") as f:
            data = pickle.load(f)
        vecs, labels = data["vecs"], data["labels"]
        print("documents for domain {} ready, total: {}".format(domain, len(labels)))
        # model
        model, tokenizer = create_model(param, device)
        model.load_state_dict(
            torch.load(os.path.join(param["model_dir"], "model.bin"), map_location=device)
        )
        model = model.eval()
        state_map[domain] = {
            "model": model,
            "tokenizer": tokenizer,
            "vecs": vecs,
            "labels": labels
        }
    return state_map


state_map = get_state_map()


def rank_single_text(domain, text, return_type="list"):
    if domain not in state_map:
        return ERROR_LABEL
    tokenizer, model = state_map[domain]["tokenizer"], state_map[domain]["model"]
    vecs, labels = state_map[domain]["vecs"], state_map[domain]["labels"]

    tokens = tokenizer.tokenize(text)[:max_length-2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        query_vec = model.encode_query(input_ids, None).squeeze(0)
        scores = model.evaluate(query_vec, vecs)
    if return_type == "list":
        sort_scores, indics = torch.sort(scores, descending=True)
        return_labels = []
        if sort_scores[0].item() > ENSURE_THRESHOLD:
            return_labels.append(labels[indics[0].item()])
        else:
            cnt = 0
            for index in indics.tolist():
                if labels[index] not in return_labels:
                    return_labels.append(labels[index])
                    cnt += 1
                if cnt >= MAX_LIST_NUM:
                    break
        return return_labels
    max_index = torch.argmax(scores).item()
    return labels[max_index]

app = FastAPI()

@app.get("/{domain}/{text}")
def read_item(domain: str, text: str):
    label = [ERROR_LABEL]
    try:
        text = str(text)
        assert len(text) > 0
        label = rank_single_text(domain, text, "list")
    except Exception as e:
        print(e)
        label = [ERROR_LABEL]
    return {"label": label}
