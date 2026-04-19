FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1JqrQB23cZVjXMnxEQGntzLhHzEGxXWL0', 'model.h5', quiet=False)"

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1", "--max-requests", "1000", "--max-requests-jitter", "50"]