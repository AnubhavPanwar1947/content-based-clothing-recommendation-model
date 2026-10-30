from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from .model_service import (
    get_catalog_filter_options,
    list_catalog_products,
    products_df,
    recommend_products,
    search_products,
)
from .schemas import (
    RecommendRequest,
    SearchRequest,
    SortOption,
)


app = FastAPI(
    title="Clothing Recommendation API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PUBLIC_PRODUCT_COLUMNS = [
    "id",
    "name",
    "imageUrl",
    "category",
    "gender",
    "colors",
    "style",
    "material",
    "season",
    "price",
    "rating",
    "popularity",
    "sizes",
    "stock",
]


def _catalog_query_params(
    gender: Optional[str] = None,
    include_unisex: bool = True,
    category: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    price_min: Optional[float] = Query(default=None, ge=0),
    price_max: Optional[float] = Query(default=None, ge=0),
    stock: Optional[str] = Query(
        default=None,
        pattern=r"^(in|low|out)$",
    ),
    sort: SortOption = "recommended",
):
    return {
        "gender": gender,
        "include_unisex": include_unisex,
        "category": category,
        "size": size,
        "color": color,
        "price_min": price_min,
        "price_max": price_max,
        "stock": stock,
        "sort": sort,
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Recommendation API is running",
    }


@app.get("/products/filter-options")
def product_filter_options():
    return get_catalog_filter_options()


@app.get("/products")
def get_products(
    limit: int = Query(default=24, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    gender: Optional[str] = None,
    include_unisex: bool = True,
    category: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    price_min: Optional[float] = Query(default=None, ge=0),
    price_max: Optional[float] = Query(default=None, ge=0),
    stock: Optional[str] = Query(
        default=None,
        pattern=r"^(in|low|out)$",
    ),
    sort: SortOption = "recommended",
):
    filter_params = _catalog_query_params(
        gender=gender,
        include_unisex=include_unisex,
        category=category,
        size=size,
        color=color,
        price_min=price_min,
        price_max=price_max,
        stock=stock,
        sort=sort,
    )

    catalog_page, total_matches = list_catalog_products(
        limit=limit,
        offset=offset,
        **filter_params,
    )

    page_columns = [
        column
        for column in PUBLIC_PRODUCT_COLUMNS
        if column in catalog_page.columns
    ]

    return {
        "offset": offset,
        "limit": limit,
        "count": total_matches,
        "total": len(products_df),
        "products": jsonable_encoder(
            catalog_page[page_columns].to_dict(orient="records")
        ),
    }


@app.get("/products/{product_id}")
def get_product(product_id: str):
    product_id = product_id.strip()

    matching_products = products_df[
        products_df["id"].astype(str) == product_id
    ]

    if matching_products.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Product not found: {product_id}",
        )

    product = matching_products.iloc[0][
        PUBLIC_PRODUCT_COLUMNS
    ].to_dict()

    return jsonable_encoder(product)


@app.post("/search")
def search(request: SearchRequest):
    results, total_matches = search_products(
        query=request.query,
        limit=request.limit,
        offset=request.offset,
        gender=request.gender,
        include_unisex=request.include_unisex,
        category=request.category,
        size=request.size,
        color=request.color,
        price_min=request.price_min,
        price_max=request.price_max,
        stock=request.stock,
        sort=request.sort,
    )

    return {
        "query": request.query,
        "count": total_matches,
        "offset": request.offset,
        "limit": request.limit,
        "results": jsonable_encoder(
            results.to_dict(orient="records")
        ),
    }


@app.post("/recommend")
def recommend(request: RecommendRequest):
    try:
        results = recommend_products(
            product_id=request.product_id,
            n=request.n,
            gender=request.gender,
            category=request.category,
            include_unisex=request.include_unisex,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        "product_id": request.product_id,
        "count": len(results),
        "results": jsonable_encoder(
            results.to_dict(orient="records")
        ),
    }
