from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import os
import io
from datetime import datetime, timedelta
import uuid
import uvicorn

from makarspace.anomaly_detection.detector import AnomalyDetector
from makarspace.anomaly_detection.simulator.radiation_generator import RadiationAnomalyGenerator

# Create API app
app = FastAPI(
    title="MakarSpace Anomaly Detection API",
    description="API for spacecraft anomaly detection and predictive maintenance",
    version="0.1.0"
)

# Store active models in memory
models = {}

# Model input and output schemas
class TelemetryPoint(BaseModel):
    timestamp: str
    temperature: float = Field(..., description="Temperature in °C")
    radiation: float = Field(..., description="Radiation level in rads")
    voltage: float = Field(..., description="Voltage in V")
    current: float = Field(..., description="Current in A")
    pressure: float = Field(..., description="Pressure in kPa")
    
class TelemetryData(BaseModel):
    telemetry: List[TelemetryPoint]
    
class ModelConfig(BaseModel):
    model_type: str = Field("lstm", description="Model type: 'lstm', 'transformer', or 'hybrid'")
    input_features: Optional[List[str]] = Field(None, description="Features to use for anomaly detection")
    sequence_length: Optional[int] = Field(None, description="Sequence length for model")
    threshold: Optional[float] = Field(None, description="Anomaly threshold")
    
class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    input_features: List[str]
    sequence_length: int
    normal_ranges: Dict[str, List[float]]
    created_at: str
    status: str
    
class DetectionRequest(BaseModel):
    model_id: str
    explain: bool = Field(False, description="Whether to generate explanations")
    
class SimulationConfig(BaseModel):
    duration: str = Field("1d", description="Duration to simulate (e.g., '1d', '12h')")
    sampling_interval: str = Field("1m", description="Sampling interval (e.g., '1m', '10s')")
    base_temperature: float = Field(20.0, description="Baseline temperature in °C")
    base_radiation: float = Field(10.0, description="Baseline radiation in rads")
    include_anomalies: bool = Field(True, description="Whether to include anomalies")
    anomaly_rate: float = Field(0.05, description="Rate of anomalies to include")
    include_mission_phases: bool = Field(True, description="Whether to include mission phases")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")

# Background task for model training
def train_model_task(model_id: str, data_path: str, config: Dict[str, Any]):
    try:
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrames
        telemetry = pd.DataFrame(data['telemetry'])
        telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'])
        telemetry.set_index('timestamp', inplace=True)
        
        if 'anomalies' in data:
            anomalies = pd.DataFrame(data['anomalies'])
            anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
            anomalies.set_index('timestamp', inplace=True)
            data_dict = {'telemetry': telemetry, 'anomalies': anomalies}
        else:
            data_dict = {'telemetry': telemetry}
        
        # Create and train model
        detector = AnomalyDetector(
            model_type=config.get('model_type', 'lstm'),
            input_features=config.get('input_features'),
            sequence_length=config.get('sequence_length'),
            threshold=config.get('threshold')
        )
        
        # Train the model
        results = detector.train(data_dict)
        
        # Save model
        os.makedirs(f"models/{model_id}", exist_ok=True)
        detector.save(f"models/{model_id}")
        
        # Update model status
        models[model_id]['status'] = 'ready'
        models[model_id]['training_results'] = results
        models[model_id]['model'] = detector
        
    except Exception as e:
        # Update model status on failure
        models[model_id]['status'] = 'failed'
        models[model_id]['error'] = str(e)

@app.get("/", response_class=HTMLResponse)
async def root():
    """API root showing basic information."""
    return """
    <html>
        <head>
            <title>MakarSpace Anomaly Detection API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #2c3e50; }
                .container { max-width: 800px; margin: 0 auto; }
                code { background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; }
                .endpoints { background-color: #f5f5f5; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MakarSpace Anomaly Detection API</h1>
                <p>Welcome to the MakarSpace Anomaly Detection API for spacecraft telemetry monitoring.</p>
                
                <h2>Available Endpoints:</h2>
                <div class="endpoints">
                    <p><code>GET /docs</code> - Interactive API documentation</p>
                    <p><code>GET /models</code> - List all available models</p>
                    <p><code>POST /models/train</code> - Train a new model</p>
                    <p><code>POST /detect</code> - Detect anomalies in telemetry data</p>
                    <p><code>GET /simulate</code> - Generate synthetic telemetry data</p>
                </div>
                
                <p>For detailed documentation, visit <a href="/docs">/docs</a>.</p>
            </div>
        </body>
    </html>
    """

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available anomaly detection models."""
    # Convert models dict to list of ModelInfo
    model_list = []
    for model_id, model_data in models.items():
        normal_ranges = {}
        for k, v in model_data.get('normal_ranges', {}).items():
            normal_ranges[k] = [v[0], v[1]]
            
        model_list.append(ModelInfo(
            model_id=model_id,
            model_type=model_data['model_type'],
            input_features=model_data['input_features'] or [],
            sequence_length=model_data['sequence_length'] or 0,
            normal_ranges=normal_ranges,
            created_at=model_data['created_at'],
            status=model_data['status']
        ))
    
    return model_list

@app.post("/models/train", response_model=ModelInfo)
async def train_model(
    background_tasks: BackgroundTasks,
    config: ModelConfig = None,
    file: UploadFile = File(...)
):
    """
    Train a new anomaly detection model.
    
    Uploads telemetry data and trains a new model with the specified configuration.
    Training happens asynchronously, and the model status can be checked with /models endpoint.
    """
    # Generate model ID
    model_id = str(uuid.uuid4())
    
    # Save uploaded data
    os.makedirs("data", exist_ok=True)
    data_path = f"data/{model_id}.json"
    
    # Read and validate data
    contents = await file.read()
    try:
        data = json.loads(contents)
        # Basic validation
        if 'telemetry' not in data:
            raise HTTPException(status_code=400, detail="Data must contain 'telemetry' key")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    
    # Save data
    with open(data_path, 'w') as f:
        json.dump(data, f)
    
    # Create model config
    config_dict = config.dict() if config else {}
    
    # Register model
    models[model_id] = {
        'model_id': model_id,
        'model_type': config_dict.get('model_type', 'lstm'),
        'input_features': config_dict.get('input_features'),
        'sequence_length': config_dict.get('sequence_length'),
        'threshold': config_dict.get('threshold'),
        'status': 'training',
        'created_at': datetime.now().isoformat(),
        'normal_ranges': {}
    }
    
    # Start training in background
    background_tasks.add_task(train_model_task, model_id, data_path, config_dict)
    
    # Return model info
    return ModelInfo(
        model_id=model_id,
        model_type=models[model_id]['model_type'],
        input_features=models[model_id]['input_features'] or [],
        sequence_length=models[model_id]['sequence_length'] or 0,
        normal_ranges={},
        created_at=models[model_id]['created_at'],
        status=models[model_id]['status']
    )

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    model_data = models[model_id]
    
    # Convert normal ranges format
    normal_ranges = {}
    for k, v in model_data.get('normal_ranges', {}).items():
        normal_ranges[k] = [v[0], v[1]]
    
    response = {
        'model_id': model_id,
        'model_type': model_data['model_type'],
        'input_features': model_data['input_features'],
        'sequence_length': model_data['sequence_length'],
        'normal_ranges': normal_ranges,
        'created_at': model_data['created_at'],
        'status': model_data['status']
    }
    
    # Include training results if available
    if 'training_results' in model_data:
        response['training_results'] = model_data['training_results']
    
    return response

@app.post("/detect")
async def detect_anomalies(detection_request: DetectionRequest, telemetry_data: TelemetryData):
    """
    Detect anomalies in telemetry data.
    
    Uses a trained model to detect anomalies in the provided telemetry data.
    Can optionally generate explanations for detected anomalies.
    """
    model_id = detection_request.model_id
    explain = detection_request.explain
    
    # Check if model exists and is ready
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    if models[model_id]['status'] != 'ready':
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model_id} is not ready (status: {models[model_id]['status']})"
        )
    
    # Convert input data to DataFrame
    telemetry = pd.DataFrame([t.dict() for t in telemetry_data.telemetry])
    telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'])
    telemetry.set_index('timestamp', inplace=True)
    
    # Get detector from model registry
    detector = models[model_id]['model']
    
    # Detect anomalies
    results = detector.detect(telemetry, explain=explain)
    
    # Convert results to JSON-serializable format
    response = {
        'model_id': model_id,
        'data_points': results['data_points'],
        'anomaly_count': results['anomaly_count'],
        'anomaly_timestamps': [ts.isoformat() for ts in results['anomaly_timestamps']],
        'predictions': results['predictions'].reset_index().to_dict(orient='records')
    }
    
    # Include explanations if requested
    if explain and 'explanations' in results:
        explanations_dict = {}
        
        for ts, exp in results['explanations'].items():
            # Convert non-serializable parts
            exp_dict = {
                'top_features': [(f, float(s)) for f, s in exp['top_features']],
                'explanation_text': exp['explanation_text']
            }
            
            # Add deviations
            exp_dict['deviations'] = {}
            for f, d in exp['deviations'].items():
                exp_dict['deviations'][f] = {
                    'direction': d['direction'],
                    'normal_range': d['normal_range'],
                    'actual_value': float(d['actual_value']),
                    'percent_deviation': float(d['percent_deviation'])
                }
                
            explanations_dict[ts.isoformat()] = exp_dict
            
        response['explanations'] = explanations_dict
    
    return response

@app.get("/simulate")
async def simulate_telemetry(config: SimulationConfig = Depends()):
    """
    Generate synthetic spacecraft telemetry data.
    
    Creates realistic spacecraft telemetry data with optional anomalies,
    useful for testing and demonstration purposes.
    """
    # Create generator
    generator = RadiationAnomalyGenerator(
        base_temperature=config.base_temperature,
        base_radiation=config.base_radiation,
        radiation_spike_threshold=500,  # Default for spacecraft
        random_seed=config.random_seed
    )
    
    # Generate data
    data = generator.generate(
        duration=config.duration,
        sampling_interval=config.sampling_interval,
        include_anomalies=config.include_anomalies,
        anomaly_rate=config.anomaly_rate,
        include_mission_phases=config.include_mission_phases
    )
    
    # Convert to serializable format
    telemetry_records = data['telemetry'].reset_index().to_dict(orient='records')
    anomaly_records = data['anomalies'].reset_index().to_dict(orient='records')
    
    # Format timestamps
    for record in telemetry_records:
        record['timestamp'] = record['timestamp'].isoformat()
        
    for record in anomaly_records:
        record['timestamp'] = record['timestamp'].isoformat()
    
    return {
        'config': config.dict(),
        'telemetry': telemetry_records,
        'anomalies': anomaly_records
    }

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from the registry."""
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Remove from registry
    del models[model_id]
    
    # Delete files
    import shutil
    model_path = f"models/{model_id}"
    data_path = f"data/{model_id}.json"
    
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        
    if os.path.exists(data_path):
        os.remove(data_path)
    
    return {"message": f"Model {model_id} deleted successfully"}

# Script to run the API server
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
