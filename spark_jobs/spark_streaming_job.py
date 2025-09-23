from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
import json
import os
from pathlib import Path
import sys
from kafka import KafkaProducer
import time
# =====================
# Setup paths & imports
# =====================
file_path = Path(__file__)
root_path = file_path.resolve().parent.parent

sys.path.append(str(root_path))           # for src.pipelines
sys.path.append(str(root_path / 'app-ml')) 
sys.path.append(str(root_path / 'common'))

from common.utils import read_config, tfidf_load_model, bert_load_model, bert_load_tokenizer
from src.pipelines.pipeline_runner import PipelineRunner

# =====================
# Load config & models
# =====================
config = read_config(config_path=str(root_path / 'config/config.yaml'))

tfidf_model = tfidf_load_model(os.path.join(config['model']['model_path'], config['model']['model_file']))
bert_model = bert_load_model(os.path.join(config['model']['bert_model_path'], config['model']['bert_model_file']))
bert_tokenizer = bert_load_tokenizer()

pipeline_runner = PipelineRunner(config)

# =====================
# Initialize Spark
# =====================
spark = SparkSession.builder \
    .appName("SentimentStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# =====================
# Kafka parameters (Docker container names)
# =====================
kafka_bootstrap = "kafka:9092"  # container name of Kafka broker
input_topic = "sentiment-input"
output_topic = "sentiment-output"

# producer = KafkaProducer(
#     bootstrap_servers=kafka_bootstrap,
#     value_serializer=lambda v: json.dumps(v).encode("utf-8")
# )

while True:
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print("Connected to Kafka!")
        break
    except Exception as e:
        print(f"Kafka not ready yet ({e}), retrying in 5s...")
        time.sleep(5)

# =====================
# Kafka schema
# =====================
schema = StructType([
    StructField("text", StringType()),
    StructField("client_id", StringType())
])

# =====================
# Read streaming data from Kafka
# =====================
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap) \
    .option("subscribe", input_topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

df = df.selectExpr("CAST(value AS STRING) as json_str") \
       .select(from_json(col("json_str"), schema).alias("data")) \
       .select("data.*")

# =====================
# Batch inference function
# =====================
def process_batch(batch_df, batch_id):
    if batch_df.count() == 0:
        return
    
    pdf = batch_df.toPandas()
    pdf = pdf.rename(columns={"text": "comment_text"})  # Keep client_id intact

    # Run your ML inference
    tfidf_preds, bert_preds = pipeline_runner.run_inference(
        pdf,
        model_type="both",
        model_loaded=True,
        tfidf_model=tfidf_model,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer
    )

    print(tfidf_preds, bert_preds)

    for idx, (tfidf_pred, bert_pred) in enumerate(zip(tfidf_preds, bert_preds)):
        row = pdf.iloc[idx]
        value = {
            "client_id": row["client_id"],               # important!
            "text": row["comment_text"],
            "tfidf_prediction": float(tfidf_pred[0]) if tfidf_pred is not None else None,
            "bert_prediction": float(bert_pred[0]) if bert_pred is not None else None
        }
        producer.send(output_topic, value=value)

    producer.flush()  # Ensure messages are sent

# =====================
# Start streaming with foreachBatch
# =====================
query = df.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", str(root_path / "tmp/checkpoints")) \
    .start()

print(f"Streaming inference started: {input_topic} -> {output_topic}")
query.awaitTermination()
