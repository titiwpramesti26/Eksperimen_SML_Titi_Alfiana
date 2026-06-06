FROM python:3.10-slim

WORKDIR /app

# Copy dependency manifest first to maximize Docker layer caching
COPY MLProject_Folder/requirements-docker.txt /app/requirements-docker.txt

# Install project dependencies (keep pip cache to speed up rebuilds)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-docker.txt

# Copy entire repo so relative paths resolved from modelling.py work
COPY . /app

WORKDIR /app/MLProject_Folder

# Run training script
CMD ["python", "modelling.py"]


