from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from control_engine import ControlEngine

app = FastAPI(title="ControlPlot Studio API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ControlEngine()

class TransferFunctionInput(BaseModel):
    numerator: str
    denominator: str

class BuilderInput(BaseModel):
    gain: float
    zeros: List[str]
    poles: List[str]

class FrequencyRange(BaseModel):
    omega_min: float = 0.01
    omega_max: float = 10000
    num_points: int = 1000

@app.get("/")
def read_root():
    return {"message": "ControlPlot Studio API", "version": "1.0.0"}

@app.post("/parse")
def parse_transfer_function(tf: TransferFunctionInput):
    """Parse transfer function from string input"""
    success = engine.parse_transfer_function(tf.numerator, tf.denominator)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid transfer function format")
    
    return {
        "message": "Transfer function parsed successfully",
        "num": engine.num,
        "den": engine.den
    }

@app.post("/builder")
def parse_from_builder(builder: BuilderInput):
    """Parse transfer function from builder interface"""
    success = engine.parse_from_builder(builder.gain, builder.zeros, builder.poles)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid builder parameters")
    
    return {
        "message": "Transfer function built successfully",
        "num": engine.num,
        "den": engine.den
    }

@app.post("/bode")
def get_bode_data():
    """Get Bode plot data"""
    if not engine.num or not engine.den:
        raise HTTPException(status_code=400, detail="No transfer function loaded")
    
    return engine.get_bode_data()

@app.post("/frequency_response")
def get_frequency_response(freq_range: FrequencyRange):
    """Get frequency response data"""
    if not engine.num or not engine.den:
        raise HTTPException(status_code=400, detail="No transfer function loaded")
    
    return engine.get_frequency_response(
        freq_range.omega_min,
        freq_range.omega_max,
        freq_range.num_points
    )

@app.post("/pole_zero")
def get_pole_zero_data():
    """Get pole-zero data"""
    if not engine.num or not engine.den:
        raise HTTPException(status_code=400, detail="No transfer function loaded")
    
    return engine.get_pole_zero_data()

@app.post("/stability")
def get_stability():
    """Get stability information"""
    if not engine.num or not engine.den:
        raise HTTPException(status_code=400, detail="No transfer function loaded")
    
    return engine.get_stability_info()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
