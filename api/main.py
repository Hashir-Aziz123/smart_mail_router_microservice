import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from src.utils.model_loader import fetch_and_load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global state for the model
routing_pipeline: Pipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global routing_pipeline
    try:
        logging.info("Initializing remote model fetch sequence...")
        routing_pipeline = fetch_and_load_model()
        logging.info("Routing model loaded into memory successfully.")
    except Exception as startup_error:
        logging.critical(f"Failed to load model during startup: {str(startup_error)}")
        # We allow the server to start so the /health endpoint can report the degraded status
    yield
    routing_pipeline = None

app = FastAPI(title="Smart Mail Router API", lifespan=lifespan)

class TicketRequest(BaseModel):
    issue_description: str = Field(..., min_length=10, description="The raw text of the customer support ticket.")

class RoutingPrediction(BaseModel):
    department: str
    confidence_score: float | None = None

@app.get("/health")
async def health_check():
    status = "healthy" if routing_pipeline is not None else "degraded"
    return {"status": status, "model_loaded": routing_pipeline is not None}

@app.post("/predict", response_model=RoutingPrediction)
async def predict_routing(ticket: TicketRequest):
    if routing_pipeline is None:
        raise HTTPException(status_code=503, detail="Routing model is currently unavailable.")
    
    try:
        prediction = routing_pipeline.predict([ticket.issue_description])[0]
        
        # LogisticRegression supports predict_proba, allowing confidence scoring
        probabilities = routing_pipeline.predict_proba([ticket.issue_description])[0]
        max_confidence = max(probabilities)

        return RoutingPrediction(
            department=str(prediction),
            confidence_score=round(float(max_confidence), 4)
        )
    except Exception as exception_instance:
        logging.error(f"Prediction inference failed: {str(exception_instance)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")