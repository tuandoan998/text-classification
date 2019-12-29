import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from utils import *
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    cf = confusion_matrix(labels, preds)
    return mcc, cf

def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)

def evaluate(model, test_dataloader, num_labels):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds, all_label_ids = [], []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        all_label_ids+=label_ids.tolist()

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        pred_batch = np.argmax(logits.detach().cpu().numpy(), axis=1)
        print(pred_batch)
        preds.append(pred_batch.tolist())

    eval_loss = eval_loss / nb_eval_steps
    preds = [item for sublist in preds for item in sublist]
    result = compute_metrics(np.array(all_label_ids), preds)
    
    return result

if __name__=='__main__':    
    test_dataloader, label_list, _ = load_data(data_set='test')
    num_labels=len(label_list)

    model = BertForSequenceClassification.from_pretrained(config["CACHE_DIR"] + config["FINE_TUNED_MODEL"], cache_dir=config["CACHE_DIR"], num_labels=num_labels)
    model.to(device)
    
    mcc, cf = evaluate(model, test_dataloader, num_labels)
    print('mmc: ', mcc)
    print('confusion matrix: ', cf)
    