# Start from a minimal Python environment (version 3.10.12)
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app

# Install Poetry
# NOTE: We also disable virtual environment creation in Poetry so that
# it installs packages into the system environment.
RUN pip install --no-cache-dir poetry==1.5.1
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/usr/local"

# Copy poetry files first (for efficient Docker build caching)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies with Poetry
# By default, "poetry install" will try to install dev dependencies as well.
# Use "--no-dev" if you only want production dependencies.
RUN poetry install --no-dev

# Now copy the rest of the application code
COPY . /app

# Expose the default Dash/Flask port
EXPOSE 8050

# Recommended: Configure your Dash app to listen on 0.0.0.0 in app.py
# e.g., app.run_server(debug=True, host="0.0.0.0", port=8050)

# Set the command to run your Dash app using Poetry
CMD ["poetry", "run", "python", "app.py"]
