from flask import Flask, request, jsonify
from kafka import KafkaProducer, KafkaConsumer
import threading
import json
import uuid
from collections import defaultdict, deque
import time 
import os

app = Flask(__name__)

# -----------------------------
# Kafka configuration (local)
# -----------------------------
KAFKA_BOOTSTRAP = "kafka:9092"
INPUT_TOPIC = "sentiment-input"
OUTPUT_TOPIC = "sentiment-output"


while True:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print("Connected to Kafka!")
        break
    except Exception as e:
        print(f"Kafka not ready yet ({e}), retrying in 5s...")
        time.sleep(5)

# Thread-safe storage for results per client
RESULTS_BUFFER = defaultdict(lambda: deque(maxlen=100))  # client_id -> deque of results

# -----------------------------
# Consumer for ML results from Spark
# -----------------------------
while True:
    try:
        GROUP_ID = f"flask-results-{os.getpid()}"

        consumer_results = KafkaConsumer(
            OUTPUT_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            auto_offset_reset='earliest',
            enable_auto_commit=False,   # don’t auto-commit so it always starts fresh
            group_id=GROUP_ID,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        print("Consumer connected to Kafka!")
        break
    except Exception as e:
        print(f"Kafka not ready yet ({e}), retrying in 5s...")
        time.sleep(5)


def results_worker():
    for message in consumer_results:
        data = message.value
        client_id = data.get("client_id")
        if not client_id:
            continue
        RESULTS_BUFFER[client_id].append(data)
        print(f"[Results Worker] Stored result for client {client_id}: {data}")

threading.Thread(target=results_worker, daemon=True).start()

# -----------------------------
# Routes
# -----------------------------
@app.route("/start_session", methods=["GET"])
def start_session():
    client_id = str(uuid.uuid4())
    return jsonify({"client_id": client_id})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    client_id = data.get("client_id")
    texts = data.get("texts", [])

    if not client_id:
        return jsonify({"error": "client_id required"}), 400
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    for text in texts:
        producer.send(INPUT_TOPIC, {"client_id": client_id, "text": text})
    producer.flush()
    print(f"[Predict] Sent {len(texts)} texts for client {client_id} to Kafka")

    return jsonify({"message": f"{len(texts)} texts sent to Kafka for client {client_id}"}), 200

@app.route("/results", methods=["GET"])
def get_results():
    client_id = request.args.get("client_id")
    if not client_id:
        return jsonify({"error": "client_id required"}), 400

    results = []
    client_queue = RESULTS_BUFFER[client_id]
    while client_queue:
        results.append(client_queue.popleft())  # remove as we read

    return jsonify(results)

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
