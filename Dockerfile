FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-core curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsS https://dotenvx.sh | sh

WORKDIR /app
COPY . .
RUN if [ -f ./raganything/pyproject.toml ] || [ -f ./raganything/setup.py ]; then \
      pip install --no-cache-dir -e ./raganything; \
    else \
      pip install --no-cache-dir "raganything @ https://github.com/gptcompany/raganything/archive/refs/heads/main.zip"; \
    fi

# Models NOT included — mount ~/.cache/huggingface as volume
EXPOSE 8767
CMD ["dotenvx", "run", "-f", ".env", "--", "python", "scripts/raganything_service.py"]
