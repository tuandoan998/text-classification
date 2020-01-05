import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from transformers import DistilBertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from multiprocessing import Pool, cpu_count
import json
import numpy as np
from utils import *
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(train_examples_len):
    num_train_optimization_steps = int(train_examples_len / config["BATCH_SIZE"] / config["GRADIENT_ACCUMULATION_STEPS"]) * config["NUM_TRAIN_EPOCHS"]
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=config["LEARNING_RATE"],
                     warmup=config["WARMUP_PROPORTION"],
                     t_total=num_train_optimization_steps)
    return optimizer

def train(model, train_dataloader, optimizer, num_labels):
    print('... Fine tuning ...')
    model.train()
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    for _ in trange(int(config["NUM_TRAIN_EPOCHS"]), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask, labels=None)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            if config["GRADIENT_ACCUMULATION_STEPS"] > 1:
                loss = loss / config["GRADIENT_ACCUMULATION_STEPS"]

            loss.backward()
            print("\r%f" % loss, end='')

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % config["GRADIENT_ACCUMULATION_STEPS"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
    print('Fine tuned!')

if __name__=='__main__':
    train_dataloader, label_list, train_examples_len = load_data(data_set='train')
    num_labels=len(label_list)
    
    print('... Loading BERT BASE model ...')
    if config["NAME_BERT_MODEL"]=='bert-base-cased':
        model = BertForSequenceClassification.from_pretrained(config["NAME_BERT_MODEL"], cache_dir = config["CACHE_DIR"], num_labels=num_labels)
    else:
        model = DistilBertForSequenceClassification.from_pretrained(config["NAME_BERT_MODEL"], cache_dir = config["CACHE_DIR"], num_labels=num_labels)

    print('Model loaded!')
    model.to(device)
    
    optimizer = get_optimizer(train_examples_len)
    
    train(model, train_dataloader, optimizer, num_labels)
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(config["OUTPUT_DIR"], config["WEIGHTS_NAME"])
    output_config_file = os.path.join(config["OUTPUT_DIR"], config["CONFIG_NAME"])

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    
    
