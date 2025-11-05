# -----------------------------
# 1. Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# 2. Set environment variables
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# -----------------------------
# 3. Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 4. Copy requirements first (for caching)
# -----------------------------
COPY requirements.txt .

# -----------------------------
# 5. Install dependencies
# -----------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 6. Copy project files
# -----------------------------
COPY common/ common/
COPY app-ml/ app-ml/
COPY config/ config/
COPY model/ model/

# -----------------------------
# 7. Expose the port
# -----------------------------
EXPOSE 8000

# -----------------------------
# 8. Run the app
# -----------------------------
CMD ["python", "app-ml/entrypoint/inference_api.py"]
