# Smart Mail Router Microservice

A production-grade machine learning microservice that automatically classifies and routes customer support tickets using a fine-tuned Transformer model.

Unlike traditional ML deployments that bundle the full PyTorch stack into production, this project separates training from inference. The production service runs only ONNX Runtime, making it lightweight, fast, and suitable for memory-constrained environments.

---

## Architecture

The system consists of four independent stages:

### Training Pipeline
- Downloads datasets from Hugging Face
- Fine-tunes `distilbert-base-uncased`
- Exports the trained model to ONNX
- Uploads model artifacts to the Hugging Face Hub

### Model Registry
The ONNX model, tokenizer, and configuration files are versioned in a private Hugging Face repository, keeping model artifacts separate from the source code.

### CI/CD Pipeline
Every push triggers a GitHub Actions workflow that:

- Downloads the latest model
- Runs API and integration tests
- Evaluates the model against a golden dataset
- Prevents deployment if performance falls below a predefined threshold

### Runtime Service
The FastAPI application downloads the latest ONNX model during startup and performs inference using ONNX Runtime and the Hugging Face Rust tokenizer.

---

## Project Structure

```text
.
├── .github/workflows/
├── api/
│   ├── config.py
│   └── main.py
├── src/
│   ├── pipeline/
│   └── utils/
├── tests/
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```

---

## Tech Stack

- FastAPI
- Transformers
- PyTorch
- ONNX Runtime
- Hugging Face Hub
- GitHub Actions
- Docker

---

## Setup

### Training

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Set your Hugging Face token:

```bash
# Windows
set HF_TOKEN=your_token

# Linux/macOS
export HF_TOKEN=your_token
```

Run the training pipeline:

```bash
python -m src.pipeline.train_transformer
```

---

### Run the API

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn api.main:app --reload
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

## Testing

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Deployment

The application is containerized with Docker and deployed through GitHub Actions. Successful builds are published to GitHub Container Registry after passing automated integration and model validation tests.