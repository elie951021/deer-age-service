from fastapi import FastAPI
from app.routers import estimate

app = FastAPI(
    title="Deer Age Estimation API",
    description="Estimates deer age from jaw/teeth images using LLM vision.",
    version="0.1.0",
)

app.include_router(estimate.router, prefix="/api/v1")
