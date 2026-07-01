# Use the exact slim version matching your local environment
FROM python:3.12-slim

# Prevent Python from writing .pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd -m -r apiuser

# Set the working directory
WORKDIR /app

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies cleanly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and utilities
COPY ./api ./api
COPY ./src ./src

# Transfer ownership of the app directory to the non-root user
RUN chown -R apiuser:apiuser /app

# Switch to the non-root user
USER apiuser

# Expose the port FastAPI will run on
EXPOSE 8000

# Execute the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]