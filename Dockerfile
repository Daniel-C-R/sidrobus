# --- Builder stage: install production deps with Poetry via pip ---
FROM python:3.13-slim AS builder

# 1. Configure Poetry to install into the global env (no venv)
ENV POETRY_VERSION=2.0.0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR="/var/cache/pypoetry"

# 2. Install build tools + Poetry, then remove build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install "poetry==${POETRY_VERSION}" \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy lockfiles only to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# 4. Install only main (non-dev) dependencies into global site-packages
#    --no-root skips installing your project as a package
RUN poetry install --only main --no-root --no-interaction --no-ansi \
    && rm -rf /root/.cache/pip /var/cache/pypoetry

# --- Final stage: minimal runtime image ---
FROM python:3.13-slim AS final

# 5. Ensure unbuffered logs and no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 6. Install only runtime system libraries (e.g. PostgreSQL client)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 7. Copy installed Python packages & CLI entrypoints from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 8. Copy application code
COPY . .

# 9. Create a non-root user with a valid home and switch to it
RUN useradd --system --create-home --home-dir /home/app --shell /bin/bash app \
    && chown -R app:app /app /home/app
USER app
ENV HOME=/home/app

# 10. Expose Streamlit default port
EXPOSE 8501

# 11. Add /app to PYTHONPATH so sidrobus package is importable
ENV PYTHONPATH="/app"

# 12. Launch the Streamlit app
CMD ["streamlit", "run", "web_app/Sidrobus.py", "--server.port=8501", "--server.address=0.0.0.0"]