import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from pytorch_pretrained_bert import BertTokenizer
from transformers import DistilBertTokenizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
from utils import *

with open('config.json') as json_file:
        config = json.load(json_file)

def get_features(examples, label_map):
    if config["NAME_BERT_MODEL"]=='bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained(config["NAME_BERT_MODEL"], do_lower_case=False)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(config["NAME_BERT_MODEL"], do_lower_case=False)
    examples_len = len(examples)
    examples_for_processing = [(example, label_map, config["MAX_SEQ_LENGTH"], tokenizer) for example in examples]
    print(f'... Preparing to convert {examples_len} examples ...')
    process_count = cpu_count() - 1
    with Pool(process_count) as p:
        features = list(tqdm(p.imap(convert_example_to_feature, examples_for_processing), total=examples_len))
    return features

def load_data(data_set='train'):
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    processor = NewsClassificationProcessor(config["DATA_DIR"])
    label_list = processor.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    
    if data_set=='train':
        examples = processor.get_train_examples()
    elif data_set=='test':
        examples = processor.get_test_examples()
        
    features = get_features(examples, label_map)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    sampler = SequentialSampler(data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=config["BATCH_SIZE"])
    
    return data_loader, label_list, len(features)

if __name__=='__main__':
    data_loader, _, num_examples = load_data(data_set='test')
    print(num_examples)