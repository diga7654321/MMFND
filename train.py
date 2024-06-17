import os
from torch.utils.data import DataLoader
from mynetwork_ple import MultiModal
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import logging, BertConfig, BertModel
from torch.autograd import Variable
from tqdm import tqdm
from dataset import *
from models import *
import cn_clip.clip as clip_cn

# Set logging verbosity for transformers library
logging.set_verbosity_warning()
logging.set_verbosity_error()



def prepre_weibo():
    batch_size = 16
    lr = 0.0005
    l2 = 0
    # Set CUDA device if available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BERT model
    model_name = 'bert-base-chinese'
    config = BertConfig.from_pretrained(model_name, num_labels=3)
    config.output_hidden_states = True
    BERT = BertModel.from_pretrained(model_name, config=config).cuda()

    # Freeze the parameters of the BERT model
    for param in BERT.parameters():
        param.requires_grad = False

    # create dataset
    train_set = weibo_dataset(is_train=True)
    test_set = weibo_dataset(is_train=False)

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate_fn_weibo,
        shuffle=True)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate_fn_weibo,
        shuffle=False)

    # Initialize the MultiModal network
    rumor_module = MultiModal()
    rumor_module.to(device)

    # 加载检查点
    # path = ''
    # rumor_module.load_state_dict(torch.load(path))

    # Define the optimizer
    optim_task = torch.optim.Adam(
        rumor_module.parameters(), lr=lr, weight_decay=l2)

    # Define the loss function for rumor classification
    loss_f_rumor = torch.nn.CrossEntropyLoss()

    clipmodel, preprocess = load_chinese_clip_model(device)
    # clipmodel, preprocess = pre_clip()
    # Load BERT tokenizer
    token = BertTokenizer.from_pretrained('bert-base-chinese')

    return device,train_set,test_set,train_loader,test_loader,rumor_module,optim_task,loss_f_rumor,BERT,clipmodel,token

def train():
    device, train_set, test_set, train_loader, test_loader, rumor_module, optim_task, loss_f_rumor, BERT, clipmodel, token = prepre_weibo()
    # print(test_loader)
    # Training loop
    for epoch in range(25):
        rumor_module.train()
        corrects_pre_rumor = 0
        loss_total = 0
        rumor_count = 0
        for i, (sents, image, imageclip, labels) in tqdm(enumerate(train_loader)):
            image, imageclip, labels = (to_var(image), to_var(imageclip), to_var(labels))
            text_pre = clip_cn.tokenize(sents).to(device)
            with torch.no_grad():
                data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                               truncation=True,
                                               padding='max_length',
                                               max_length=300,
                                               return_tensors='pt',
                                               return_length=True
                                               )
                # Prepare input data for the model
                input_ids = data['input_ids'].cuda()
                attention_mask = data['attention_mask'].cuda()
                token_type_ids = data['token_type_ids'].cuda()
                BERT_feature = BERT(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

                last_hidden_states = BERT_feature['last_hidden_state']
                all_hidden_states = BERT_feature['hidden_states']

                # Encode image and text using CLIP model
                image_clip = clipmodel.encode_image(imageclip)  # Chinese-clip模型
                text_clip = clipmodel.encode_text(text_pre)

            # Forward pass through the MultiModal network
            pre_rumor = rumor_module(input_ids, all_hidden_states, image, image_clip, text_clip)
            loss_rumor = loss_f_rumor(pre_rumor, labels)

            optim_task.zero_grad()
            loss_rumor.backward()
            optim_task.step()
            pre_label_rumor = pre_rumor.argmax(1)
            corrects_pre_rumor += pre_label_rumor.eq(labels.view_as(pre_label_rumor)).sum().item()


            loss_total += loss_rumor.item() * last_hidden_states.shape[0]
            rumor_count += last_hidden_states.shape[0]

        loss_rumor_train = loss_total / rumor_count
        acc_rumor_train = corrects_pre_rumor / rumor_count

        acc_rumor_test, loss_rumor_test, conf_rumor = test(rumor_module, test_loader, BERT, token, clipmodel, epoch + 1)
        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f ||  loss_rumor_train = %.3f || loss_rumor_test = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test, loss_rumor_train, loss_rumor_test))
        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def test(rumor_module, test_loader, BERT, token, clipmodel, epoch):
    rumor_module.eval()
    loss_f_rumor = torch.nn.CrossEntropyLoss()
    rumor_count = 0
    loss_total = 0
    rumor_label_all = []
    rumor_pre_label_all = []
    save_path = 'checkpoint/epoch_{}.pt'.format(epoch)
    with torch.no_grad():
        for i, (sents, image, imageclip, labels) in enumerate(test_loader):
            image, imageclip, labels = (to_var(image), to_var(imageclip), to_var(labels))
            text_pre = clip_cn.tokenize(sents).to(device)
            # textclip = clip.tokenize(textclip, truncate=True).cuda()
            data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=300,
                                           return_tensors='pt',
                                           return_length=True)
            # Prepare input data for the model
            input_ids = data['input_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()
            token_type_ids = data['token_type_ids'].cuda()
            BERT_feature = BERT(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            last_hidden_states = BERT_feature['last_hidden_state']
            all_hidden_states = BERT_feature['hidden_states']

            # Encode image and text using CLIP model
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(text_pre)

            # Forward pass through the MultiModal network
            pre_rumor = rumor_module(input_ids, all_hidden_states, image, image_clip, text_clip)
            loss_rumor = loss_f_rumor(pre_rumor, labels)

            pre_label_rumor = pre_rumor.argmax(1)
            loss_total += loss_rumor.item() * last_hidden_states.shape[0]
            rumor_count += last_hidden_states.shape[0]

            # Store predicted and true labels for evaluation
            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(labels.detach().cpu().numpy())

        #torch.save(rumor_module.state_dict(), save_path)
        # Calculate accuracy and confusion matrix
        loss_rumor_test = loss_total / rumor_count
        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)
        acc_rumor_test = accuracy_score(rumor_pre_label_all, rumor_label_all)
        conf_rumor = confusion_matrix(rumor_pre_label_all, rumor_label_all)

    return acc_rumor_test, loss_rumor_test, conf_rumor


if __name__ == "__main__":
    train()
