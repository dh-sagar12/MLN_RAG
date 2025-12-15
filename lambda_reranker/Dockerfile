# Docker image for Lambda reranker function
# Build: docker build -t reranker-lambda .
# Test locally: docker run -p 9000:8080 reranker-lambda

FROM public.ecr.aws/lambda/python:3.11

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch CPU version first (smaller size, sufficient for inference)
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies with pinned versions that have pre-built wheels
COPY requirements_lambda.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements_lambda.txt

# Pre-download the model during build to speed up cold starts
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-large', max_length=512)"

# Copy handler
COPY handler.py ${LAMBDA_TASK_ROOT}

# Set the handler
CMD ["handler.lambda_handler"]
