from pathlib import Path
import sys
import os

root_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(root_path)
sys.path.append(str(root_path / 'common'))

from common.utils import save_tfidf_model
from common.utils import save_bert_model


class PostProcessing:
    def __init__(self, config):
        self.config = config

    def run_train(self, tfidf_model, bert_model):
        save_tfidf_model(tfidf_model, path = os.path.join(self.config['model']['model_path'], self.config['model']['model_file']))
        save_bert_model(bert_model, path = os.path.join(self.config['model']['bert_model_path'], self.config['model']['bert_model_file']))
        
    def run_inference(self, tfidf_preds, bert_preds):
        return tfidf_preds, bert_preds