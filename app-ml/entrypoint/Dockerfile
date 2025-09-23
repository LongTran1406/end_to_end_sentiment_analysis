FROM python:3.10
WORKDIR /app

COPY requirements.txt requirements.txt
COPY app-ml/entrypoint/inference_api.py app-ml/entrypoint/inference_api.py
COPY common common
COPY app-ml/src app-ml/src
COPY config config
COPY model model

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app-ml/entrypoint/inference_api.py"]