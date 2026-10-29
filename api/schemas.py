from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=200
    )

    n: int = Field(
        default=10,
        ge=1,
        le=50
    )

    gender: Optional[str] = None

    include_unisex: bool = True


class RecommendRequest(BaseModel):
    product_id: str = Field(
        ...,
        min_length=1
    )

    n: int = Field(
        default=5,
        ge=1,
        le=50
    )

    gender: Optional[str] = None

    category: Optional[str] = None

    include_unisex: bool = True