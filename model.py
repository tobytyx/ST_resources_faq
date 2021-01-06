# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn
from transformers import BertModel


def dot_attention(q, k, v):
    """

    :param q: B * cnt * hidden
    :param k: B * len * hidden
    :param v: B * len * hidden
    :return:
    """
    attn_weights = torch.matmul(q, k.transpose(2, 1))  # B * cnt * len
    attn_weights = torch.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, v)  # B * cnt * hidden
    return output


class BiEncoder(nn.Module):
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(args["pre_model"])
        self.sim = "normal"
        if "sim" in args:
            self.sim = args["sim"]

    def forward(self, query_input_ids, query_input_masks, response_input_ids, response_input_masks):
        """

        :param query_input_ids: B * len
        :param query_input_masks: B * len
        :param response_input_ids: B * cnt * len
        :param response_input_masks: B * cnt * len
        :return:
        """
        query_vec = self.encoder(query_input_ids, query_input_masks)[0][:, 0, :]  # B * dim
        query_vec = query_vec.unsqueeze(1)  # B * 1 * dim
        batch_size, cnt, length = response_input_ids.size()
        response_input_ids = response_input_ids.view(batch_size * cnt, length)
        response_input_masks = response_input_masks.view(batch_size * cnt, length)
        responses_vec = self.encoder(response_input_ids, response_input_masks)[0][:, 0, :]  # B x cnt * dim
        responses_vec = responses_vec.view(batch_size, cnt, -1)
        if self.sim == "normal":
            output = torch.matmul(query_vec, responses_vec.permute(0, 2, 1)).squeeze(1)
        else:
            query_vec = query_vec.expand(-1, cnt, -1)
            output = torch.cosine_similarity(query_vec, responses_vec, dim=-1)
        return output

    def encode_query(self, input_ids, masks):
        vec = self.encoder(input_ids, masks)[0][:, 0, :]
        return vec

    def encode_response(self, input_ids, masks):
        return self.encode_query(input_ids, masks)

    @staticmethod
    def evaluate(query_vec, response_vecs):
        """

        :param query_vec: hidden_size
        :param response_vecs: cnt * hidden_size
        :return: cnt
        """
        query_vec = query_vec.unsqueeze(0).expand(response_vecs.size(0), -1)
        output = torch.cosine_similarity(query_vec, response_vecs, dim=-1)
        return output


class PolyEncoder(nn.Module):
    def __init__(self, args: Dict, device):
        super(PolyEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(args["pre_model"])
        self.device = device
        args["hidden_size"] = self.encoder.config.hidden_size
        self.args = args
        self.poly_m = args["poly_m"]
        self.poly_code_embeddings = nn.Embedding(self.poly_m, args["hidden_size"])
        self.sim = args.get("sim", "normal")
        torch.nn.init.normal_(self.poly_code_embeddings.weight, args["hidden_size"] ** -0.5)

    def forward(self, query_input_ids, query_input_masks, responses_input_ids, responses_input_masks):
        """

        :param query_input_ids: B * len
        :param query_input_masks: B * len
        :param responses_input_ids: B * cnt * len
        :param responses_input_masks: B * cnt * len
        :return:
        """
        device = self.device
        batch_size, cnt, seq_length = responses_input_ids.size()

        query_out = self.encoder(query_input_ids, query_input_masks)[0]  # B * len * hidden
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long, device=device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, -1)
        poly_codes = self.poly_code_embeddings(poly_code_ids)  # [bs, poly_m, dim]
        query_vec = dot_attention(poly_codes, query_out, query_out)  # [bs, poly_m, dim]

        responses_input_ids = responses_input_ids.view(batch_size * cnt, seq_length)
        responses_input_masks = responses_input_masks.view(batch_size * cnt, seq_length)
        response_out = self.encoder(responses_input_ids, responses_input_masks)[0][:, 0, :]  # [bs*cnt, dim]
        response_vec = response_out.view(batch_size, cnt, -1)  # [bs, cnt, dim]
        ctx_emb = dot_attention(response_vec, query_vec, query_vec)  # [bs, cnt, dim]
        if self.sim == "normal":
            output = torch.sum(ctx_emb * response_vec, dim=-1)  # [bs, cnt]
        else:  # cosine
            output = torch.cosine_similarity(ctx_emb, response_vec, dim=-1)
        return output

    def encode_query(self, input_ids, masks):
        batch_size = input_ids.size(0)
        encoder_output = self.encoder(input_ids, masks)  # B * len * hidden
        query_out = encoder_output[0]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long, device=self.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, -1)
        poly_codes = self.poly_code_embeddings(poly_code_ids)  # B * poly_m * hidden
        vec = dot_attention(poly_codes, query_out, query_out)  # B * poly_m * hidden
        return vec

    def encode_response(self, input_ids, masks):
        encoder_output = self.encoder(input_ids, masks)
        vec = encoder_output[0][:, 0, :]
        return vec

    @staticmethod
    def evaluate(query_vec, response_vecs):
        """

        :param query_vec: poly_m * hidden
        :param response_vecs: cnt * hidden
        :return: B * cnt
        """
        query_vec = query_vec.unsqueeze(0)
        response_vecs = response_vecs.unsqueeze(0)
        ctx_emb = dot_attention(response_vecs, query_vec, query_vec)
        ctx_emb = ctx_emb.squeeze()  # cnt * hidden
        output = torch.cosine_similarity(ctx_emb, response_vecs, dim=-1)  # res_cnt
        return output
