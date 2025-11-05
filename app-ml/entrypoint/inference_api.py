from pathlib import Path
import os
import sys
from flask import Flask, request, jsonify
import pandas as pd

file_path = Path(__file__)
root_path = file_path.resolve().parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / 'app-ml'))

from common.utils import read_config, tfidf_load_model, bert_load_model, bert_load_tokenizer
from common.data_manager import DataManager
from src.pipelines.pipeline_runner import PipelineRunner
file_path = Path(__file__)
root_path = file_path.resolve().parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / 'app-ml'))

config = read_config(config_path = str(root_path / 'config/config.yaml'))
runner = PipelineRunner(config)
tfidf_model = tfidf_load_model(path = os.path.join(config['model']['model_path'], config['model']['model_file']))
bert_model = bert_load_model(path = os.path.join(config['model']['bert_model_path'], config['model']['bert_model_file']))
bert_tokenizer = bert_load_tokenizer()
app = Flask(__name__)



@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data.get("texts", [])
    models = data.get("models", [])
    df = pd.DataFrame(texts, columns=['comment_text'])
    print(df)
    print(texts, models)
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    if not models:
        tfidf, bert = runner.run_inference(df, 'both', model_loaded=True, tfidf_model=tfidf_model, bert_model=bert_model, bert_tokenizer=bert_tokenizer)
        tfidf = tfidf.flatten().tolist()
        bert = bert.flatten().tolist()
        return jsonify({'tfidf': tfidf, 'bert': bert})

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
