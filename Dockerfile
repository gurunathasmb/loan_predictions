FROM python:3.10-slim

WORKDIR /app

# Install only system dependencies needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY . .

# Streamlit will look for this
ENV PYTHONUNBUFFERED=1

# Expose Streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "Scripts/app.py"]
