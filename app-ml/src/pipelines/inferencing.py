from pathlib import Path
import sys
import os
import torch

root_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path / 'common'))

from common.utils import tfidf_load_model
from common.utils import bert_load_model
from common.utils import bert_load_tokenizer

class Inferencing:
    def __init__(self, config):
        self.config = config
    
    def run(self, df, model_type='both', model_loaded = False, tfidf_model = None, bert_model = False, bert_tokenizer = None):
        tfidf_preds = None
        bert_preds = None
        if model_loaded is False:
            if model_type == 'both' or model_type == 'tfidf':
                tfidf_model = tfidf_load_model(path = os.path.join(self.config['model']['model_path'], self.config['model']['model_file']))
                tfidf_preds = tfidf_model.predict_proba(df['comment_text'])[:, 1].reshape(-1, 1)

            if model_type == 'both' or model_type == 'bert':
                bert_model = bert_load_model(path = os.path.join(self.config['model']['bert_model_path'], self.config['model']['bert_model_file']))
                bert_tokenizer = bert_load_tokenizer()

                encoded = bert_tokenizer(list(df['comment_text']), padding=True, truncation=True, max_length=128, return_tensors='pt')
                with torch.no_grad():
                    outputs = bert_model(**encoded)
                    if outputs.logits.shape[1] == 1:  # Binary regression
                        bert_preds = torch.sigmoid(outputs.logits).numpy()
                    else:  # Multi-class classification
                        bert_preds = torch.softmax(outputs.logits, dim=1).numpy()
        
        else:
            if tfidf_model is not None:
                tfidf_preds = tfidf_model.predict_proba(df['comment_text'])[:, 1].reshape(-1, 1)
            if bert_model is not None:
                encoded = bert_tokenizer(list(df['comment_text']), padding=True, truncation=True, max_length=128, return_tensors='pt')
                with torch.no_grad():
                    outputs = bert_model(**encoded)
                    if outputs.logits.shape[1] == 1:  # Binary regression
                        bert_preds = torch.sigmoid(outputs.logits).numpy()
                    else:  # Multi-class classification
                        bert_preds = torch.softmax(outputs.logits, dim=1).numpy()
        
        return tfidf_preds, bert_preds