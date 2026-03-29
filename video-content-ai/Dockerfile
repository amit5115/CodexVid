FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY app/ app/

ENV VCAI_HOST=0.0.0.0
ENV VCAI_PORT=8501
ENV VCAI_RELOAD=false
ENV VCAI_DATA_DIR=/app/data

EXPOSE 8501

CMD ["python", "-m", "app.main"]
