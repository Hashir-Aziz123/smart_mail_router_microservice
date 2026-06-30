import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_FILE_PATH = DATA_DIR / "customer_support_tickets_200k.csv"
MODEL_PATH = ARTIFACTS_DIR / "router_model.joblib"

def load_ticket_data(file_path: Path) -> pd.DataFrame:
    """Loads and sanitizes the customer support ticket dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. Ensure the CSV is in the data/ directory.")
    
    logging.info(f"Loading dataset from {file_path}...")
    
    # Read only necessary columns to optimize memory usage
    df = pd.read_csv(file_path, usecols=['issue_description', 'category'])
    
    # Dropped rows with missing text or labels
    initial_row_count = df.shape[0]
    df.dropna(subset=['issue_description', 'category'], inplace=True)
    dropped_rows = initial_row_count - df.shape[0]
    
    if dropped_rows > 0:
        logging.warning(f"Data Validation: Dropped {dropped_rows} rows containing missing values.")
        
    return df

def train_routing_model():
    """Executes the NLP training pipeline for ticket routing."""
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    df = load_ticket_data(DATA_FILE_PATH)

    logging.info("Splitting dataset into training and testing sets...")
    # Using stratify to ensure minority categories are proportionally represented in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        df['issue_description'], 
        df['category'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['category'] 
    )

    logging.info("Initializing modeling pipeline...")
    # Bundle the vectorizer and classifier into a single artifact
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

    logging.info(f"Saving serialized model artifact to {MODEL_PATH}")
    joblib.dump(model_pipeline, MODEL_PATH)
    logging.info("Training pipeline execution complete.")

if __name__ == "__main__":
    train_routing_model()