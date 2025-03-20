FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Copy project files
COPY . /app/

# Configure Poetry to create virtual environment in the project directory
RUN poetry config virtualenvs.in-project true

# Install dependencies
RUN poetry install --with dev

# Set entrypoint to run the examples
ENTRYPOINT ["poetry", "run", "python"]
CMD ["src/ember/examples/basic/minimal_example.py"]