# -*- coding: utf-8 -*-
from tools import create_model
import torch
import json
import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel
from mysql_utils import (
    get_mysql_connect, insert_model_record, delete_model_record, update_model_record, init_model_record,
    STATE_ERROR_NUMBER, STATE_READY_NUMBER, STATE_TRAINING_NUMBER, STATE_USING_NUMBER)

SUCCESS_CODE = 0
FAIL_CODE = -1

max_length = 64
device = torch.device("cpu")
ERROR_LABEL = 0
MAX_LIST_NUM = 3
ENSURE_THRESHOLD = 0.92


class Item(BaseModel):
    record_id: int=None
    name: str=None
    domain: str=None
    data_path: str=None
    category_num: int=None


def load_model(model_dir):
    with open(os.path.join(model_dir, "args.json"), mode="r") as f:
        args = json.load(f)
    with open(os.path.join(model_dir, "total.pkl"), mode="rb") as f:
        data = pickle.load(f)
    vecs, labels = data["vecs"], data["labels"]
    # model
    model, tokenizer = create_model(args, device)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "model.bin"), map_location=device)
    )
    model = model.eval()
    return {
        "record_id": args["record_id"],
        "name": args["name"],
        "model": model,
        "tokenizer": tokenizer,
        "vecs": vecs,
        "labels": labels
    }


state_map = {
    "patent": {},
    "expert": {},
    "equipment": {}
}


def rank_single_text(domain, text, return_type="list"):
    if domain not in state_map or "tokenizer" not in state_map[domain]:
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

init_status = init_model_record()
assert init_status == 0, "{}".format(init_status)

app = FastAPI()


@app.get("/faq/")
async def faq_service(domain: str, text: str):
    label = [ERROR_LABEL]
    state = SUCCESS_CODE
    try:
        text = str(text)
        assert len(text) > 0
        label = rank_single_text(domain, text, "list")
    except Exception as e:
        print(e)
        label = [ERROR_LABEL]
        state = FAIL_CODE
    return {"label": label, "state": state}


@app.post("/create/")
async def create_model_service(item: Item):
    state = SUCCESS_CODE
    record_id = -1
    try:
        name = item.name
        domain = item.domain
        data_path = item.data_path
        category_num = item.category_num
        conn = get_mysql_connect()
        record_id = insert_model_record(conn, name, domain, STATE_TRAINING_NUMBER, data_path, category_num, "")
        conn.close()
        os.system("sh auto_train.sh {} {} {} {}".format(record_id, name, domain, data_path))
    except:
        state = FAIL_CODE
    return {"record_id": record_id, "state": state}


@app.post("/update/")
async def update_model_service(item: Item):
    state = SUCCESS_CODE
    try:
        record_id = item.record_id
        name = item.name
        domain = item.domain
        global state_map
        old_record_id = state_map[domain].get("record_id", -1)
        model_dir = "./output/{}/{}_{}".format(domain, name, record_id)
        state_map[domain]["model"] = None
        state_map[domain] = load_model(model_dir)
        conn = get_mysql_connect()
        if old_record_id != -1:
            update_model_record(conn, old_record_id, STATE_READY_NUMBER)
        update_model_record(conn, record_id, STATE_USING_NUMBER)
        conn.close()
    except:
        state = FAIL_CODE
    return {"state": state}


@app.post("/delete/")
async def delete_model_service(item: Item):
    state = SUCCESS_CODE
    model_state = STATE_READY_NUMBER
    try:
        record_id = item.record_id
        conn = get_mysql_connect()
        res = delete_model_record(conn, record_id)
        if res != STATE_READY_NUMBER:
            state = FAIL_CODE
        model_state = res
    except:
        state = FAIL_CODE
        model_state = STATE_ERROR_NUMBER
    return {"state": state, "model_state": model_state}
