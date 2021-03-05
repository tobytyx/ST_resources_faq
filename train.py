# -*- coding: utf-8 -*-
import os
import torch
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import EncoderDataset
import random
import numpy as np
from tools import create_model, get_tsv_data
from log import create_logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--domain", type=str, default="patent", choices=["patent", "expert", "equipment"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--log_step", type=int, default=6)
    parser.add_argument("--eval_step", type=int, default=6)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="bi", choices=["bi", "poly"])
    parser.add_argument("--sim", type=str, default="cosine", choices=["cosine", "normal"])
    parser.add_argument("--cosine_scale", type=float, default=5.,
                        help="scale to help cosine result with cross_entropy")
    parser.add_argument("--poly_m", type=int, default=32)
    parser.add_argument("--cnt", type=int, default=5)
    parser.add_argument("--pre_model", type=str, default="./dependences/roberta_wwm_ext")
    args = parser.parse_known_args()[0]
    args = vars(args)
    return args


def train(model, device, train_list, val_list, args, tokenizer, logger):
    train_dataset = EncoderDataset(train_list, tokenizer, args["max_length"], args["cnt"])
    val_dataset = EncoderDataset(val_list, tokenizer, args["max_length"], args["cnt"])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True,
        num_workers=args["num_workers"], collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args["batch_size"], shuffle=True,
        num_workers=args["num_workers"], collate_fn=val_dataset.collate_fn)
    total_steps = int(len(train_dataloader) * args["epochs"])
    optimizer = AdamW(model.parameters(), lr=args["lr"], correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=total_steps)
    best_acc, acc = 0, 0
    logger.info('starting training, each epoch step {}'.format(len(train_dataloader)))
    cur_step = 1
    for epoch in range(1, args["epochs"]+1):
        total_loss = 0
        model.train()
        for batch_step, batch in enumerate(train_dataloader):
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
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            scheduler.step()
            if (batch_step+1) % args["log_step"] == 0:
                total_loss /= args["log_step"]
                logger.info("[{}:{}], loss {:.4f}".format(epoch, batch_step+1, total_loss))
                total_loss = 0

            if cur_step % args["eval_step"] == 0:
                acc, bad_cases = evaluate(model, device, val_dataloader)
                update_best_model = False
                if  acc > best_acc:
                    best_acc = acc
                    update_best_model = True
                    model_path = os.path.join(args["output_dir"], "model.bin")
                    with open(os.path.join(args["output_dir"], "bad_cases.txt"), mode="w") as f:
                        for bad_case in bad_cases:
                            query, wrong, gt = bad_case
                            while len(query) > 0 and query[-1] == tokenizer.pad_token_id:
                                query.pop()
                            query = tokenizer.convert_ids_to_tokens(query)
                            while len(wrong) > 0 and wrong[-1] == tokenizer.pad_token_id:
                                wrong.pop()
                            wrong = tokenizer.convert_ids_to_tokens(wrong)
                            while len(gt) > 0 and gt[-1] == tokenizer.pad_token_id:
                                gt.pop()
                            gt = tokenizer.convert_ids_to_tokens(gt)
                            f.write("{}\t{}\t{}\n".format(" ".join(query), " ".join(wrong), " ".join(gt)))
                    torch.save(model.state_dict(), model_path)
                log_info = "[{}:{}][total:{}], Acc: {:.2f}%".format(epoch, batch_step+1, cur_step, acc*100)
                if update_best_model:
                    log_info = log_info + " updated best model"
                logger.info(log_info)
            cur_step += 1
    logger.info('training finished')


def evaluate(model, device, val_dataloader):
    bad_cases = []
    model.eval()
    with torch.no_grad():
        total_num = 0
        correct = 0
        for batch in val_dataloader:
            queries, q_masks, responses, r_masks = batch
            queries, q_masks = queries.to(device), q_masks.to(device)
            responses, r_masks = responses.to(device), r_masks.to(device)
            # B * cnt
            outputs = model(queries, q_masks, responses, r_masks)
            labels = torch.zeros(outputs.size(0), dtype=torch.long, device=device)
            pred_labels = outputs.argmax(-1).detach()
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            bsz = labels.size()[0]
            for i in range(bsz):
                pred = pred_labels[i].item()
                label = labels[i].item()
                if pred != label:
                    bad_cases.append(
                        [queries[i].detach().cpu().tolist(),
                            responses[i][pred].detach().cpu().tolist(),
                            responses[i][label].detach().cpu().tolist()])
            total_num += bsz
    acc = correct / total_num
    return acc, bad_cases


def main():
    setup_seed(20)
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args["output_dir"] = os.path.join("./output", args["domain"], args["name"])
    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"], exist_ok=True)
    logger = create_logger()
    with open(os.path.join(args["output_dir"], "args.json"), mode="w") as f:
        json.dump(args, f, ensure_ascii=False, indent=2)
    logger.info(args)
    model, tokenizer = create_model(args, device)
    model = model.to(device)
    train_list = get_tsv_data(os.path.join("./data", args["domain"], "train.tsv"))
    val_list = get_tsv_data(os.path.join("./data", args["domain"], "test.tsv"))
    # 开始训练
    train(model, device, train_list, val_list, args, tokenizer, logger)


if __name__ == '__main__':
    main()
