from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.model_service import ModelService
from app.schemas import GenerationRequest, GenerationResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str | None = None
    device_map: str = "auto"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = ModelService(
        base_model=settings.base_model,
        adapter_path=settings.adapter_path,
        device_map=settings.device_map,
    )
    yield
    service = None


app = FastAPI(
    title="LLM Fine-Tuning Studio API",
    version="0.1.0",
    description="Inference API for a LoRA or QLoRA fine-tuned language model.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": service is not None,
        "base_model": settings.base_model,
    }


@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest) -> GenerationResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        text = service.generate(
            instruction=request.instruction,
            context=request.context,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Generation failed") from exc

    return GenerationResponse(
        text=text,
        base_model=settings.base_model,
        adapter_path=settings.adapter_path,
    )
