# Docker image for Lambda reranker function
# Build: docker build -t reranker-lambda .
# Test locally: docker run -p 9000:8080 reranker-lambda

FROM public.ecr.aws/lambda/python:3.11

# Create model directory in the task root (read-only at runtime, but bundled with image)
ENV MODEL_DIR=${LAMBDA_TASK_ROOT}/models

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch CPU version first (smaller size, sufficient for inference)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies with pinned versions that have pre-built wheels
COPY requirements_lambda.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements_lambda.txt

# Pre-download the model during build to the model directory
# Set cache locations for build time
RUN mkdir -p ${MODEL_DIR} && \
    HF_HOME=${MODEL_DIR} \
    TRANSFORMERS_CACHE=${MODEL_DIR} \
    SENTENCE_TRANSFORMERS_HOME=${MODEL_DIR} \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-large', max_length=512)"

# Copy handler
COPY handler.py ${LAMBDA_TASK_ROOT}

# Set runtime environment variables
# HF_HOME points to the bundled model, but we use /tmp for any runtime writes
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface  
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface
ENV BUNDLED_MODEL_DIR=${LAMBDA_TASK_ROOT}/models

# Set the handler
CMD ["handler.lambda_handler"]
