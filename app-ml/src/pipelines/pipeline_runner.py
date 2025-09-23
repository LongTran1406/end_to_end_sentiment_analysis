from src.pipelines.preprocessing import Preprocessing
from src.pipelines.feature_engineer import FeatureEngineer
from src.pipelines.training import Training
from src.pipelines.postprocessing import PostProcessing
from src.pipelines.inferencing import Inferencing
import os


class PipelineRunner:
    def __init__(self, config):
        self.config = config
        self.preprocess = Preprocessing(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.training = Training(self.config)
        self.postprocessing = PostProcessing(self.config)
        self.inference = Inferencing(self.config)

    def run_train(self, df):
        df = df.sample(frac = 0.1, random_state = 42)
        df = self.preprocess.run(df)

        df.to_csv(os.path.join(self.config['data']['cleaned_folder'], self.config['data']['cleaned_file']))

        df = self.feature_engineer.run(df)
        tfidf_model, bert_model = self.training.run(df)
        self.postprocessing.run_train(tfidf_model, bert_model)
    
    def run_inference(self, df, model_type = 'both', model_loaded = False, tfidf_model = None, bert_model = False, bert_tokenizer = None):
        df = self.preprocess.run(df)
        df = self.feature_engineer.run(df)
        tfidf_preds, bert_preds = self.inference.run(df, model_type, model_loaded, tfidf_model, bert_model, bert_tokenizer)
        tfidf_preds, bert_preds = self.postprocessing.run_inference(tfidf_preds, bert_preds)
        return tfidf_preds, bert_preds
