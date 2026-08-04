FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY config.py .
COPY model_loader.py .
COPY inference.py .
COPY database.py .
COPY schemas.py .
COPY main.py .

RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
