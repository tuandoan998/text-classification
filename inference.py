import numpy as np
import torch
import json
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inference:
    def __init__(self, config_file):
        with open(config_file) as json_file:
            self.config = json.load(json_file)
        self.label_list = pd.read_csv(os.path.join(self.config["DATA_DIR"], "train.tsv"), header=None, sep='\t', usecols=[1]).T.squeeze().unique()
        # self.label_list = ['business', 'entertainment', 'politics', 'sport', 'tech']
        print('... Loading model ...')
        self.model = BertForSequenceClassification.from_pretrained(self.config["CACHE_DIR"] + self.config["FINE_TUNED_MODEL"], cache_dir=self.config["CACHE_DIR"], num_labels=len(self.label_list))
        print('Model loaded!')
        self.model.to(device)
        self.tokenizer = BertTokenizer.from_pretrained(self.config["BERT_BASE_CASED_MODEL"], do_lower_case=False)
        
    def predict(self, text):
        self.model.eval()
        input_example = InputExample(guid=None, text_a = text, text_b = None, label = None)
        input_feature = convert_example_to_feature((input_example, self.label_list, self.config["MAX_SEQ_LENGTH"], self.tokenizer), to_test=True)

        input_ids = torch.tensor(input_feature.input_ids, dtype=torch.long).unsqueeze(0).to(device)
        input_mask = torch.tensor(input_feature.input_mask, dtype=torch.long).unsqueeze(0).to(device)
        segment_ids = torch.tensor(input_feature.segment_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = self.model(input_ids, segment_ids, input_mask, labels=None)
        
        int2label = {i: label for i, label in enumerate(self.label_list)}

        return input_ids, int2label[np.argmax(logit.squeeze().detach().cpu().numpy())]


if __name__=='__main__':    
    news_model = Inference('config.json')
    
    text = "England will play a friendly against Italy at Wembley in March as part of their preparations for Euro 2020. Italy are ranked 13th in the world, nine places below England, and finished qualification for the European Championships with a 100% record. The meeting on Friday 27 March will be the first of four friendlies for Gareth Southgate's side ahead of the Euros. England also play Denmark at Wembley on 31 March, Austria in Vienna on 2 June and Romania at home on 7 June. The Italy fixture will also be designated as 'The Heads Up International' in support of the Football Association's partnership with The Duke of Cambridge's Heads Together mental health initiative. It will be the 28th meeting between England and four-time World Cup winners Italy, with the most recent a 1-1 draw at Wembley in 2018."
    ids, catogery = news_model.predict(text)
    print("Catogery: ", catogery)