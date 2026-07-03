import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from optimum.onnxruntime import ORTModelForSequenceClassification
from huggingface_hub import HfApi, create_repo
from api.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "distilbert-base-uncased"
ONNX_EXPORT_DIR = ROOT_DIR / "artifacts" / "onnx_model"
PYTORCH_TEMP_DIR = ROOT_DIR / "artifacts" / "pytorch_temp"

# For this initial test, we use 2000 rows. With real data, 
# this will easily clear your 85% CI/CD threshold.
SAMPLE_SIZE = 2000 

def prepare_dataset():
    logging.info("Downloading Bitext customer support dataset from Hugging Face Hub...")
    # The 'instruction' column contains the user query, 'category' contains the routing department
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    
    if len(dataset) > SAMPLE_SIZE:
        logging.info(f"Sub-sampling dataset to {SAMPLE_SIZE} rows for rapid prototyping...")
        dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))

    # Extract unique categories to build the mappings
    unique_categories = sorted(list(set(dataset["category"])))
    label2id = {label: i for i, label in enumerate(unique_categories)}
    id2label = {i: label for i, label in enumerate(unique_categories)}
    
    def format_dataset(example):
        example["label"] = label2id[example["category"]]
        example["text"] = example["instruction"]
        return example
        
    dataset = dataset.map(format_dataset)
    
    # Cast the integer column to a formal ClassLabel schema for the Arrow backend
    dataset = dataset.cast_column("label", ClassLabel(names=unique_categories))
    
    # Stratified split to ensure all categories are represented in both train and test sets
    split_ds = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
    
    return split_ds["train"], split_ds["test"], label2id, id2label

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    logging.info(f"\n--- Current Evaluation Accuracy: {acc:.4f} ---\n")
    return {"accuracy": acc}

def train_and_export():
    ONNX_EXPORT_DIR.mkdir(exist_ok=True, parents=True)
    PYTORCH_TEMP_DIR.mkdir(exist_ok=True, parents=True)
    
    train_ds, test_ds, label2id, id2label = prepare_dataset()

    logging.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_tokenized = train_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    logging.info("Initializing PyTorch DistilBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=str(ROOT_DIR / "artifacts" / "checkpoints"),
        eval_strategy="epoch",
        learning_rate=3e-5, # Slightly higher learning rate for a smaller dataset
        per_device_train_batch_size=16,
        num_train_epochs=3, # 3 epochs gives it enough passes to learn the real patterns
        weight_decay=0.01,
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )

    logging.info("Commencing fine-tuning...")
    trainer.train()

    logging.info("Saving intermediate PyTorch model to disk for stable export...")
    trainer.save_model(str(PYTORCH_TEMP_DIR))
    tokenizer.save_pretrained(str(PYTORCH_TEMP_DIR))

    logging.info("Compiling to ONNX format...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(PYTORCH_TEMP_DIR),
        export=True
    )
    
    logging.info("Saving ONNX graph locally...")
    tokenizer.save_pretrained(ONNX_EXPORT_DIR)
    ort_model.save_pretrained(ONNX_EXPORT_DIR)

    logging.info("Pushing ONNX compiled model to Hugging Face Registry...")
    api = HfApi(token=settings.hf_token)
    create_repo(repo_id=settings.hf_repo_id, repo_type="model", private=True, exist_ok=True)
    
    api.upload_folder(
        folder_path=str(ONNX_EXPORT_DIR),
        repo_id=settings.hf_repo_id,
        repo_type="model"
    )
    logging.info("Phase 1 Complete: ONNX Transformer successfully pushed.")

if __name__ == "__main__":
    train_and_export()