from flask import Flask, jsonify, request
from kafka import KafkaConsumer
import json
import threading
from collections import deque

app = Flask(__name__)

# Thread-safe buffer for Kafka messages
RESULTS_BUFFER = deque(maxlen=100)

# Track last index per client (shared with producer)
client_pointers = {}

# Kafka consumer
consumer = KafkaConsumer(
    'sentiment-output',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='flask-consumer',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def consume_messages():
    for message in consumer:
        RESULTS_BUFFER.append(message.value)

threading.Thread(target=consume_messages, daemon=True).start()

@app.route("/results", methods=["GET"])
def get_results():
    client_id = request.args.get("client_id")
    if not client_id:
        return jsonify({"error": "client_id required"}), 400

    if client_id not in client_pointers:
        return jsonify({"error": "Invalid client_id"}), 400

    # Filter only messages for this client
    client_messages = [msg for msg in list(RESULTS_BUFFER) if msg.get("client_id") == client_id]
    
    # Only return new messages for this client
    last_index = client_pointers[client_id]
    new_messages = client_messages[last_index:]
    client_pointers[client_id] = last_index + len(new_messages)

    return jsonify(new_messages)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
