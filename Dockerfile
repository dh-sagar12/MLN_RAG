# Dockerfile for Starlette Dashboard Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (includes Chromium dependencies)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    wget \
    gnupg \
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

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /home/app/.cache && \
    chown -R app:app /home/app /app

# Switch to app user and set HOME environment variable
USER app
ENV HOME=/home/app

# Install Playwright browsers as 'app' user (without --with-deps since system deps already installed)
RUN python -m playwright install chromium

# Copy application code
COPY --chown=app:app app/ ./app/
COPY --chown=app:app *.py ./
COPY --chown=app:app app/templates/ ./app/templates/
COPY --chown=app:app app/static/ ./app/static/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the Starlette application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]