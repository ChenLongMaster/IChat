# Use official Python image
FROM python:3.12-slim
# Set workdir
WORKDIR /dockerApp

# Install system dependencies (optional but helpful for some libs)
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libxslt1-dev \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Copy files
COPY app ./app

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port FastAPI runs on
EXPOSE 8000

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
