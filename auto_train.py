# -*- coding: utf-8 -*-
import os
import traceback
import torch
import json
import pickle
import time
from shutil import copyfile
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from transformers import AdamW
import random
import numpy as np
from dataset import EncoderDataset
from tools import create_model, get_tsv_data, create_data_vec
from log import create_logger
from mysql_utils import get_mysql_connect, update_model_record, STATE_ERROR_NUMBER, STATE_READY_NUMBER



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_id", type=int, required=True)
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--data_path", type=str, default="init")
    parser.add_argument("--domain", type=str, default="patent", choices=["patent", "expert", "equipment"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="bi", choices=["bi", "poly"])
    parser.add_argument("--sim", type=str, default="cosine", choices=["cosine", "normal"])
    parser.add_argument("--cosine_scale", type=float, default=5.,
                        help="scale to help cosine result with cross_entropy")
    parser.add_argument("--poly_m", type=int, default=32)
    parser.add_argument("--cnt", type=int, default=5)
    parser.add_argument("--pre_model", type=str, default="./dependences/roberta_wwm_ext")
    parser.add_argument("--max_oom", type=int, default=5)
    args = parser.parse_known_args()[0]
    args = vars(args)
    return args


def main(args):
    args["output_dir"] = "./output/{}/{}_{}".format(args["domain"], args["name"], args["record_id"])
    data_path = os.path.join(args["output_dir"], "data.tsv")
    logger = create_logger(log_path=os.path.join(args["output_dir"], "train.log"))
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"], exist_ok=True)
        try:
            copyfile(args["data_path"], data_path)
        except IOError as e:
            traceback.print_exc()
            logger.info("No source file")
            return -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args["output_dir"], "args.json"), mode="w") as f:
        json.dump(args, f, ensure_ascii=False, indent=2)
    logger.info(args)
    key = False
    oom_count = 0
    model, tokenizer = create_model(args, device)
    data_list = get_tsv_data(data_path)
    # 开始训练
    train_dataset = EncoderDataset(data_list, tokenizer, args["max_length"], args["cnt"])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True,
        num_workers=args["num_workers"], collate_fn=train_dataset.collate_fn)
    while not key and oom_count < args["max_oom"]:
        try:
            model = model.to(device)
            optimizer = AdamW(model.parameters(), lr=args["lr"], correct_bias=True)
            logger.info('starting training, each epoch step {}'.format(len(train_dataloader)))
            for _ in range(1, args["epochs"]+1):
                model.train()
                for batch in train_dataloader:
                    queries, q_masks, responses, r_masks = batch
                    queries, q_masks = queries.to(device), q_masks.to(device)
                    responses, r_masks = responses.to(device), r_masks.to(device)
                    # B * cnt
                    outputs = model(queries, q_masks, responses, r_masks)
                    bsz = outputs.size(0)
                    if args["sim"] == "cosine" and args["cosine_scale"] > 1:
                        outputs = outputs * args["cosine_scale"]
                    labels = torch.zeros(bsz, dtype=torch.long, device=device)
                    loss = F.cross_entropy(outputs, labels, reduction="mean")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()
            model.eval()
            model_path = os.path.join(args["output_dir"], "model.bin")
            torch.save(model.state_dict(), model_path)
            logger.info('training finished')
            # 开始生成预制文件
            doc_vecs, doc_labels = create_data_vec(data_list, model, tokenizer, args["max_length"], device, "response", True)
            saved_data = {"vecs": doc_vecs, "labels": doc_labels}
            with open(os.path.join(args["output_dir"], "total.pkl"), mode="wb") as f:
                pickle.dump(saved_data, f)
            logger.info("finish generate corpus vecs to " + os.path.join(args["output_dir"], "total.pkl"))
            key = True
        except RuntimeError as e:
            traceback.print_exc()
            if oom_count >= args["max_oom"]:
                break
            oom_count += 1
            logger.info("OOM count: {}, sleep 30s".format(oom_count))
            time.sleep(30)
    return 0 if key else -1


if __name__ == '__main__':
    setup_seed(20)
    main_args = get_args()
    res = main(args=main_args)
    conn = get_mysql_connect()
    if conn is not None:
        if res == 0:
            update_model_record(conn, main_args["record_id"], STATE_READY_NUMBER)
        else:
            update_model_record(conn, main_args["record_id"], STATE_ERROR_NUMBER)
        conn.close()
