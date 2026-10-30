from typing import Literal, Optional

from pydantic import BaseModel, Field


SortOption = Literal[
    "recommended",
    "price-asc",
    "price-desc",
    "newest",
    "rating",
]


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    limit: int = Field(
        default=24,
        ge=1,
        le=50,
    )

    offset: int = Field(
        default=0,
        ge=0,
    )

    gender: Optional[str] = None

    include_unisex: bool = True

    category: Optional[str] = None

    size: Optional[str] = None

    color: Optional[str] = None

    price_min: Optional[float] = Field(default=None, ge=0)

    price_max: Optional[float] = Field(default=None, ge=0)

    stock: Optional[str] = Field(
        default=None,
        pattern=r"^(in|low|out)$",
    )

    sort: SortOption = "recommended"


class RecommendRequest(BaseModel):
    product_id: str = Field(
        ...,
        min_length=1,
    )

    n: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    gender: Optional[str] = None

    category: Optional[str] = None

    include_unisex: bool = True
