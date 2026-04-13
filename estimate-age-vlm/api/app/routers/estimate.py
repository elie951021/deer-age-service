from fastapi import APIRouter, Query, UploadFile, File
from app.schemas import EstimationResponse
from app.services.estimator import estimate_deer_age, estimate_deer_age_two_step

router = APIRouter()


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(
    image: UploadFile = File(...),
    mode: str = Query(
        default="single",
        description="Pipeline mode: 'single' (one-shot observe+classify) or 'two_step' (observe then classify separately)",
    ),
):
    image_bytes = await image.read()
    if mode == "two_step":
        return await estimate_deer_age_two_step(image_bytes)
    return await estimate_deer_age(image_bytes)
