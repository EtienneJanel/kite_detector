# syntax=docker/dockerfile:1.4

### STAGE 1: Build base image
FROM python:3.12-slim as builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# Copy only necessary files for poetry install
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev)
RUN poetry config virtualenvs.create false \
 && poetry install --without dev --no-root

# install torch CPU
RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Copy rest of the application
COPY . .

### STAGE 2: Runtime container
FROM python:3.12-slim

WORKDIR /app

# Reinstall necessary system libs in the runtime image
RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY --from=builder /app /app

# Expose port and run
EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
