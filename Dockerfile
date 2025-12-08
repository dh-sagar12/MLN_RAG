# Dockerfile for Starlette Dashboard Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ----------------------------------------------------------------------
# System dependencies
# ----------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    wget \
    gnupg \
    gosu \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libxshmfence1 \
    libglib2.0-0 \
    libx11-xcb1 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libxcb1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# Copy requirements
# ----------------------------------------------------------------------
COPY requirements.txt .

# ----------------------------------------------------------------------
# Install Python deps
# GPU packages COMMENTED OUT
# ----------------------------------------------------------------------

# ---- GPU SECTION (for future use) ------------------------------------
# Uncomment this block in future when GPU is needed
# RUN pip install --no-cache-dir \
#     torch==2.9.1 \
#     triton==3.5.1 \
#     nvidia-cublas-cu12 \
#     nvidia-cuda-cupti-cu12 \
#     nvidia-cuda-nvrtc-cu12 \
#     nvidia-cuda-runtime-cu12 \
#     nvidia-cudnn-cu12 \
#     nvidia-cufft-cu12 \
#     nvidia-cufile-cu12 \
#     nvidia-curand-cu12 \
#     nvidia-cusolver-cu12 \
#     nvidia-cusparse-cu12 \
#     nvidia-cusparselt-cu12 \
#     nvidia-nccl-cu12 \
#     nvidia-nvjitlink-cu12 \
#     nvidia-nvshmem-cu12 \
#     nvidia-nvtx-cu12
# ----------------------------------------------------------------------

# Install CPU-only PyTorch instead of GPU build
RUN pip install --no-cache-dir \
    torch==2.9.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------
# Create non-root user
# ----------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /home/app/.cache && \
    chown -R app:app /home/app /app

# ----------------------------------------------------------------------
# Playwright install (under non-root)
# ----------------------------------------------------------------------
USER app
ENV HOME=/home/app
RUN python -m playwright install chromium

# ----------------------------------------------------------------------
# Switch back to root & copy app files
# ----------------------------------------------------------------------
USER root

COPY --chown=app:app app/ ./app/
COPY --chown=app:app *.py ./
COPY --chown=app:app app/templates/ ./app/templates/
COPY --chown=app:app app/static/ ./app/static/

RUN mkdir -p /app/logs/performance /app/storage/uploads && \
    chown -R app:app /app

# ----------------------------------------------------------------------
# Entrypoint & healthcheck
# ----------------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
