import base64
import json
from pathlib import Path

from app.config import settings
from app.core.llm import LLMClient
from app.loggers.log_session import RequestSession
from app.loggers.logger import logger
from app.prompts.classify import DEER_CLASSIFY_PROMPT
from app.prompts.estimate import DEER_AGE_ESTIMATION_PROMPT
from app.prompts.observe import DEER_OBSERVATION_PROMPT
from app.schemas import EstimationResponse
from app.utils.image import normalize_image

llm = LLMClient(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    model_kwargs={"response_format": {"type": "json_object"}},
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _parse_json_response(response, step: str) -> dict:
    content = response.content
    if not isinstance(content, str):
        raise ValueError(f"[{step}] Unexpected LLM response type: {type(content)}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("[%s] Failed to parse JSON: %s | raw: %s", step, str(e), content)
        raise


async def estimate_deer_age(image_bytes: bytes) -> EstimationResponse:
    """Single-step pipeline: observe + classify in one LLM call."""
    session = RequestSession()
    try:
        image_bytes = normalize_image(image_bytes)
        session.save_image(image_bytes)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        BASE_DIR = Path(__file__).resolve().parent
        few_shot_base64 = encode_image(BASE_DIR / "few_shot.jpg")

        messages = DEER_AGE_ESTIMATION_PROMPT.format_messages(
            image_data=base64_image,
            few_shot_base64=few_shot_base64,
        )
        session.save_prompt(messages)

        response = llm.invoke(messages)
        logger.debug("RAW CONTENT: %s", response.content)

        data = _parse_json_response(response, "single-step")
        data["cost"] = getattr(response, "cost", None)
        session.save_response(data)

        return EstimationResponse(**data)

    except Exception as e:
        logger.error("ERROR in estimate_deer_age: %s", str(e))
        raise
    finally:
        session.close()


async def estimate_deer_age_two_step(image_bytes: bytes) -> EstimationResponse:
    """Two-step pipeline: Step 1 = structured observation, Step 2 = rule-based classification."""
    session = RequestSession()
    total_cost = 0.0
    try:
        image_bytes = normalize_image(image_bytes)
        session.save_image(image_bytes)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # --- Step 1: Observe ---
        observe_messages = DEER_OBSERVATION_PROMPT.format_messages(image_data=base64_image)
        session.save_prompt(observe_messages)

        observe_response = llm.invoke(observe_messages)
        logger.debug("OBSERVE RAW: %s", observe_response.content)
        total_cost += getattr(observe_response, "cost", 0.0) or 0.0

        observation = _parse_json_response(observe_response, "observe")
        logger.debug("OBSERVATION: %s", json.dumps(observation, indent=2))

        # --- Step 2: Classify ---
        classify_messages = DEER_CLASSIFY_PROMPT.format_messages(
            observation_json=json.dumps(observation, indent=2)
        )

        classify_response = llm.invoke(classify_messages)
        logger.debug("CLASSIFY RAW: %s", classify_response.content)
        total_cost += getattr(classify_response, "cost", 0.0) or 0.0

        data = _parse_json_response(classify_response, "classify")
        data["cost"] = total_cost
        data["observation"] = observation
        session.save_response(data)

        return EstimationResponse(**data)

    except Exception as e:
        logger.error("ERROR in estimate_deer_age_two_step: %s", str(e))
        raise
    finally:
        session.close()
