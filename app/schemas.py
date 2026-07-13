from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=5000)
    context: str = Field(default="", max_length=10000)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class GenerationResponse(BaseModel):
    text: str
    base_model: str
    adapter_path: str | None
