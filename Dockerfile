FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN grep -vE "^torch|^torchvision" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Copy application files
COPY config/   ./config/
COPY checkpoints/ ./checkpoints/
COPY src/      ./src/

EXPOSE 8000

WORKDIR /app/src

CMD ["python", "main.py"]
