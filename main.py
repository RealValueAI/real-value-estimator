import uvicorn
from fastapi import FastAPI

from src.inference.inference import run_inference_pipeline
from src.io_tools.io import refresh_data
from src.learning.fit_model import run_training_pipeline
from src.models.pydantic_models import FitParams

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Price ML Service"}


@app.post("/learning/")
async def run_full_matching_task(fit_params: FitParams):
    if fit_params.is_refresh:
        refresh_data(epoch='learning',output_file='data.parquet')
    metrics = run_training_pipeline()
    return {"message": f"success", "metrics": metrics}

@app.post("/inference/")
async def run_full_matching_task(fit_params: FitParams):
    if fit_params.is_refresh:
        refresh_data(epoch='inference',output_file='x_inference.parquet')
    run_inference_pipeline()
    return {"message": f"success"}

if __name__ == "__main__":
    uvicorn.run("main:app", port=8105, log_level="info")