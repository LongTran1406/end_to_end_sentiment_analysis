import yaml
import joblib
import torch
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer

def read_config(config_path):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error, {config_path} not found")
    
    return config

def save_tfidf_model(model, path):
    try:
        joblib.dump(model, path)
        print(f"TFIDF model saved successfully to {path}")
    except Exception as e:
        print(f"Error saving TFIDF model to {path}: {e}")

def save_bert_model(model, path):
    try:
        torch.save(model.state_dict(), path)
        print(f"BERT model saved successfully to {path}")
    except Exception as e:
        print(f"Error saving BERT model to {path}: {e}")

def tfidf_load_model(path):
    try:
        model = joblib.load(path)
        print(f"Model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        print(f"Error: File {path} not found")
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
    return None


def bert_load_model(path, num_labels=1):
    try:
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"BERT model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        print(f"Error: File {path} not found")
    except Exception as e:
        print(f"Error loading BERT model from {path}: {e}")
    return None

def bert_load_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer
