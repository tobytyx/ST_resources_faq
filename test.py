# -*- coding: utf-8 -*-
import argparse
import torch
from tools import create_model, create_data_vec, get_tsv_data
import os
import json
import pickle


def one_stage_test(model, device, max_length, tokenizer, dataset, output_dir):
    document_list = get_tsv_data("./data/{}/train.tsv".format(dataset))
    infer_list = get_tsv_data("./data/{}/test.tsv".format(dataset))
    print("total document len {}, infer len {}".format(len(document_list), len(infer_list)))
    correct_results, fail_results = [], []
    doc_vecs, doc_labels = create_data_vec(document_list, model, tokenizer, max_length, device, "response", True)
    infer_vecs, infer_labels = create_data_vec(infer_list, model, tokenizer, max_length, device, "query", True)
    infer_len = len(infer_list)
    for i in range(infer_len):
        infer_vec, infer_label = infer_vecs[i], infer_labels[i]
        scores = model.evaluate(infer_vec, doc_vecs)
        doc_index = torch.argmax(scores).item()
        pred_label = doc_labels[doc_index]
        if infer_label == pred_label:
            correct_results.append([infer_list[i], document_list[doc_index], scores[doc_index].item()])
        else:
            fail_results.append([infer_list[i], document_list[doc_index], scores[doc_index].item()])
    acc = len(correct_results) / infer_len
    print("Acc@1: {:.4f}".format(acc))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, "corrects.txt"), mode="w") as f:
        for query, doc, score in correct_results:
            f.write("{}\t{}\t{:.4f}\n".format(query, doc, score))
    with open(os.path.join(output_dir, "fails.txt"), mode="w") as f:
        for query, doc, score in fail_results:
            f.write("{}\t{}\t{:.4f}\n".format(query, doc, score))


def generate_corpus(model, device, max_length, tokenizer, dataset, output_dir):
    total_list = get_tsv_data("./data/{}/total.tsv".format(dataset))
    print("total len: ", len(total_list))
    doc_vecs, doc_labels = create_data_vec(total_list, model, tokenizer, max_length, device, "response", True)
    saved_data = {"vecs": doc_vecs, "labels": doc_labels}
    with open(os.path.join(output_dir, "total.pkl"), mode="wb") as f:
        pickle.dump(saved_data, f)
    print("finish generate corpus vecs to ", os.path.join(output_dir, "total.pkl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--task", type=str, default="test", choices=["test", "corpus"])
    parser.add_argument("--name", type=str, default="bi_patent")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    main_args = parser.parse_known_args()[0]
    main_args = vars(main_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join("./output", main_args["name"], "args.json")) as f:
        args = json.load(f)
        args.update(main_args)
    print(args)
    output_dir = os.path.join("./output", main_args["name"])
    model, tokenizer = create_model(args, device)
    model.load_state_dict(torch.load("./output/{}/model.bin".format(args["name"]),
        map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    with torch.no_grad():
        if args["task"] == "test":
            one_stage_test(model, device, args["max_length"], tokenizer, args["dataset"], output_dir)
        else:
            generate_corpus(model, device, args["max_length"], tokenizer, args["dataset"], output_dir)


if __name__ == '__main__':
    main()
