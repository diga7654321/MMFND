import math
from random import random
import torch
import torch.nn as nn
import random
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from transformers import BertTokenizer
from transformers import  BertConfig, BertModel
import copy
from cn_clip.clip import load_from_name
from pre_train_models.CLIP_BERT import clip


# class UnimodalDetection(nn.Module):
#     def __init__(self, shared_dim=512, prime_dim=256):
#         super(UnimodalDetection, self).__init__()
#
#         self.text_uni = nn.Sequential(
#             nn.Linear(1280, shared_dim),
#             nn.BatchNorm1d(shared_dim),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(shared_dim, prime_dim),
#             nn.BatchNorm1d(prime_dim),
#             nn.ReLU())
#
#         self.image_uni = nn.Sequential(
#             nn.Linear(1512, shared_dim),
#             nn.BatchNorm1d(shared_dim),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(shared_dim, prime_dim),
#             nn.BatchNorm1d(prime_dim),
#             nn.ReLU())

    # def forward(self, text_encoding, image_encoding):
    #     text_prime = self.text_uni(text_encoding)
    #     image_prime = self.image_uni(image_encoding)
    #     return text_prime, image_prime


def load_clip_model(device):#引入英文clip
    model, preprocess = clip.load('ViT-B/32', device)
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess

def load_chinese_clip_model(device):#引入中文clip
    model, preprocess = load_from_name("ViT-H-14", device=device, download_root='pre_train_models/Chinese_CLIP/')
    model.eval()
    return model,preprocess

def pretrain_bert_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").cuda()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

# def pretrain_bert_models():
#     tokenizer = BertTokenizer.from_pretrained("pretrain_models/roberta_wwm")
#     model = BertModel.from_pretrained("pretrain_models/roberta_wwm").cuda()
#     for param in model.parameters():
#         param.requires_grad = False
#     return model, tokenizer

# def bert_process(txt, model, token):
#     data = token.batch_encode_plus(batch_text_or_text_pairs=txt, truncation=True, padding='max_length', max_length=300,
#                                    return_tensors='pt', return_length=True)

#     # Prepare input data for the model
#     input_ids = data['input_ids'].cuda()
#     attention_mask = data['attention_mask'].cuda()
#     token_type_ids = data['token_type_ids'].cuda()

#     BERT_feature = model(input_ids=input_ids,
#                          attention_mask=attention_mask,
#                          token_type_ids=token_type_ids)

#     last_hidden_states = BERT_feature['last_hidden_state']

#     return last_hidden_states.cuda()
