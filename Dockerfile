FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-core curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN curl -fsS https://dotenvx.sh | sh
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.9.1+cpu torchvision==0.24.1+cpu
COPY docker-constraints.txt /tmp/docker-constraints.txt
RUN pip install --no-cache-dir -c /tmp/docker-constraints.txt sentence-transformers==5.2.0 FlagEmbedding==1.3.5 semchunk==2.2.2 docling==2.68.0 docling-core==2.59.0 docling-ibm-models==3.10.3 docling-parse==4.7.3 pdftext==0.6.3


ARG VERSION=unknown
ARG COMMIT_SHA=unknown
ARG BUILD_AT=unknown

ENV RAG_VERSION=${VERSION}
ENV RAG_COMMIT_SHA=${COMMIT_SHA}
ENV RAG_BUILD_AT=${BUILD_AT}

WORKDIR /app
COPY . .
RUN if [ -f ./raganything/pyproject.toml ] || [ -f ./raganything/setup.py ]; then \
      pip install --no-cache-dir -c /tmp/docker-constraints.txt -e ./raganything; \
    else \
      pip install --no-cache-dir -c /tmp/docker-constraints.txt "raganything @ https://github.com/gptcompany/raganything/archive/refs/heads/main.zip"; \
    fi

# Models NOT included — mount ~/.cache/huggingface as volume
EXPOSE 8767
CMD ["dotenvx", "run", "-f", ".env", "--", "python", "scripts/raganything_service.py"]
