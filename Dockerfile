# Base image
FROM python:3.9-slim

# Env vars
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy source code
COPY ./src /app/src

COPY ./models ./models

# Expose port for Flask app (sẽ được map bởi Docker Compose)
EXPOSE 8000

# Default command (sẽ bị override bởi docker-compose cho service training)
# Chạy Flask app bằng waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8000", "src.app:app"]