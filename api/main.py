from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_service import (
    recommend_products,
    search_products,
)
from .schemas import (
    RecommendRequest,
    SearchRequest,
)


app = FastAPI(
    title="Clothing Recommendation API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Recommendation API is running",
    }


@app.post("/search")
def search(request: SearchRequest):
    results = search_products(
        query=request.query,
        n=request.n,
        gender=request.gender,
        include_unisex=request.include_unisex,
    )

    return {
        "query": request.query,
        "count": len(results),
        "results": results.to_dict(orient="records"),
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
        "results": results.to_dict(orient="records"),
    }