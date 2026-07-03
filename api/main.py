import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.utils.model_loader import fetch_and_load_model, ONNXRoutingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

routing_pipeline: ONNXRoutingPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global routing_pipeline
    try:
        logging.info("Initializing remote ONNX model fetch sequence...")
        routing_pipeline = fetch_and_load_model()
        logging.info("ONNX engine loaded into memory successfully.")
    except Exception as startup_error:
        logging.critical(f"Failed to load ONNX model: {str(startup_error)}")
    yield
    routing_pipeline = None

app = FastAPI(title="Smart Mail Router API", lifespan=lifespan)

class TicketRequest(BaseModel):
    issue_description: str = Field(..., min_length=10)

class RoutingPrediction(BaseModel):
    department: str

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
        return RoutingPrediction(department=prediction)
        
    except Exception as exception_instance:
        logging.error(f"Prediction inference failed: {str(exception_instance)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")