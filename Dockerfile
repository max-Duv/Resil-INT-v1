FROM python:3.11-slim

# Pluto / IIO deps
RUN apt-get update && apt-get install -y \
    libiio-dev libxml2-dev libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY resilint_rf_rx.py .

CMD ["python", "resilint_rf_rx.py"]
