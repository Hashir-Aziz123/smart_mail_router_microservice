import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define architectural paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_FILE_PATH = DATA_DIR / "customer_support_tickets_200k.csv"
MODEL_PATH = ARTIFACTS_DIR / "router_model.joblib"

def load_ticket_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure the CSV is in the data/ directory.")
    
    logging.info(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, usecols=['issue_description', 'category'])
    df.dropna(subset=['issue_description', 'category'], inplace=True)
    return df

def train_routing_model():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    df = load_ticket_data(DATA_FILE_PATH)

    logging.info("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['issue_description'], 
        df['category'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['category'] 
    )

    logging.info("Initializing modeling pipeline...")
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=10000)),
        ('classifier', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    logging.info("Training the routing model...")
    model_pipeline.fit(X_train, y_train)

    logging.info("Evaluating model performance...")
    predictions = model_pipeline.predict(X_test)
    report = classification_report(y_test, predictions)
    print(f"\n--- Classification Report ---\n{report}")

    logging.info(f"Saving temporary local artifact to {MODEL_PATH}")
    joblib.dump(model_pipeline, MODEL_PATH)

    logging.info("Connecting to Hugging Face Registry...")
    api = HfApi()
    
    # Dynamically fetch the authenticated username
    user_info = api.whoami()
    hf_username = user_info["name"]
    repo_id = f"{hf_username}/smart-mail-router"

    logging.info(f"Ensuring private repository {repo_id} exists...")
    create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    logging.info(f"Uploading model artifact to the registry...")
    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo="router_model.joblib",
        repo_id=repo_id,
        repo_type="model"
    )
    
    logging.info("Phase 1 Complete: Artifact successfully pushed to the remote registry.")

if __name__ == "__main__":
    train_routing_model()