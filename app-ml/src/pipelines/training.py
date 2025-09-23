import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import mlflow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class TextDataset:
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class Training:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def encode_text(self, df):
        MAX_LEN = 128

        encoded_text = self.tokenizer(text = list(df['comment_text']),
                                padding = 'max_length',
                                truncation = True,
                                max_length=MAX_LEN,
                                return_tensors='pt')
        
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        labels = torch.from_numpy(df['toxicity'].to_numpy(dtype = np.float32))
        return input_ids, attention_mask, labels
    
    def create_dataset(self, input_ids, attention_mask, labels):
        dataset = TextDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle= True)
        return dataloader
    
    def train_bert(self, dataloader, lr, epochs):
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model.train()

        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        with mlflow.start_run(run_name = "distilbert-base-uncased"):
            mlflow.log_param("lr", lr)
            mlflow.log_param("epochs", epochs)

            for epoch in range(epochs):
                total_loss = 0

                for batch_idx, data in enumerate(dataloader):
                    optimizer.zero_grad()
                    
                    input_ids = data['input_ids']
                    attention_mask = data['attention_mask']
                    labels = data['labels']

                    outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)

                    loss = outputs.loss

                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    mlflow.log_metric("batch_loss", loss.item(), step=batch_idx + epoch * len(dataloader))
                    print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch_idx, loss))

                mlflow.log_metric("avg_loss", total_loss / len(dataloader), step=epoch + 1)
                print('Epoch: {}, Avg Loss: {}'.format(epoch + 1, total_loss/ len(dataloader)))
            
            mlflow.pytorch.log_model(model, "distilbert_model")
            
        return model
    
    def run_bert(self, df):
        self.load_tokenizer()
        input_ids, attention_mask, labels = self.encode_text(df)
        dataloader = self.create_dataset(input_ids, attention_mask, labels)
        bert_model = self.train_bert(dataloader, lr=2e-5, epochs = 2)
        return bert_model
    
    def run_tfidf(self, df, threshold = 0.3):
        df['labelsBinary'] = (df['toxicity'] > threshold).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['labelsBinary'], stratify = df['labelsBinary'])

        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('mnb', MultinomialNB())
        ])

        svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('svm', SVC(probability=True))
        ])

        svm_params = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf']
        }

        svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=5, scoring = 'f1_weighted', n_jobs = -1, verbose = 1)

        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        with mlflow.start_run(run_name="NaiveBayes_TFIDF"):
            # mlflow.log_params()
            nb_pipeline.fit(X_train, y_train)
            y_pred = nb_pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("Accuracy", acc)

            report = classification_report(y_test, y_pred, output_dict = True)
            mlflow.log_metrics({
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"]
            })

            mlflow.sklearn.log_model(nb_pipeline)
           
        
        with mlflow.start_run(run_name="SVM_TFIDF"):
            # mlflow.log_params()
            svm_grid.fit(X_train, y_train)
            y_pred = svm_grid.predict(X_test)
            mlflow.log_params(svm_grid.best_params_)

            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("Accuracy", acc)

            report = classification_report(y_test, y_pred, output_dict = True)
            mlflow.log_metrics({
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"]
            })

            mlflow.sklearn.log_model(svm_grid.best_estimator_)
        
        return svm_grid.best_estimator_
        
    def run(self, df):
        bert_model = self.run_bert(df)
        tfidf_model = self.run_tfidf(df)
        return tfidf_model, bert_model
        