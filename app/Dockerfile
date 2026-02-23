# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run training (if data exists) and then launch the app
# This ensures models are up-to-date with the code
CMD python train.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0
