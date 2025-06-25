# Use a Python version compatible with transformers/tokenizers
FROM python:3.10-slim

# Set working directory to root of your app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    rustc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the main Python file
COPY Chatbot.py .

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "Chatbot.py", "--server.port=8501", "--server.enableCORS=false"]
