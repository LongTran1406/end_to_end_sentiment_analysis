from pathlib import Path
import os
import sys

file_path = Path(__file__)
root_path = file_path.resolve().parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / 'app-ml'))

from common.utils import read_config
from common.data_manager import DataManager
from src.pipelines.pipeline_runner import PipelineRunner


if __name__ == '__main__':
    config = read_config(config_path = str(root_path / 'config/config.yaml'))
    
    data_manager = DataManager(config = config)

    df = data_manager.read_data(path = os.path.join(config['data']['real_time_folder'], config['data']['real_time_file']))

    pipeline_runner = PipelineRunner(config)

    tfidf_preds, bert_preds = pipeline_runner.run_inference(df, model_type = 'both')

    print(tfidf_preds)
    print(bert_preds)

