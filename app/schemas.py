"""
Pydantic models for the /chat API contract. These are deliberately kept
exactly aligned to the assignment's specified schema -- see the docstring on
ChatResponse for the exact contract text this mirrors.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """
    Mirrors the assignment's required response shape exactly:

        {
          "reply": "...",
          "recommendations": [{"name": ..., "url": ..., "test_type": ...}],
          "end_of_conversation": false
        }

    `recommendations` must be an empty list while clarifying/refusing, and
    contain 1-10 items once the agent has committed to a shortlist.
    """

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def _validate_recommendation_count(cls, v):
        if len(v) > 10:
            raise ValueError("recommendations must contain at most 10 items")
        return v


class HealthResponse(BaseModel):
    status: str = "ok"
