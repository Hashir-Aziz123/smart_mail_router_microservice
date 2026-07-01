FROM python:3.12-slim

# Prevent Python from writing .pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Created a non-root user for security
RUN useradd -m -r apiuser

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install dependencies cleanly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model artifact
COPY ./api ./api
COPY ./artifacts ./artifacts

# Transfer ownership of the app directory to the non-root user
RUN chown -R apiuser:apiuser /app

# Switch to the non-root user
USER apiuser

# Expose the port FastAPI will run on
EXPOSE 8000

# Execute the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]