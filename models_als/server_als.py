"""
FastAPI server for the implicit ALS recommender.

Start with:
    MODEL_PATH=models_als/als_recommender \\
        uvicorn models_als.server_als:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .recommender_als import AmazonRecommenderALS


recommender: AmazonRecommenderALS | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender

    model_path = os.getenv("MODEL_PATH", "models_als/als_recommender")
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"ALS recommender directory not found at '{model_path}'. "
            "Train one with `python -m models_als.train_als ...` "
            "or set MODEL_PATH to a saved recommender directory."
        )

    print(f"Loading ALS recommender from {model_path} ...")
    t0 = time.perf_counter()
    recommender = AmazonRecommenderALS.load(model_path)
    print(f"Model ready in {time.perf_counter() - t0:.2f}s "
          f"({len(recommender.title_map):,} items)")

    yield
    recommender = None


app = FastAPI(
    title="Amazon ALS Recommender API",
    description="Implicit-feedback ALS factorization, CPU/GPU interchangeable.",
    version="1.0.0",
    lifespan=lifespan,
)

_default_origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8080", "http://127.0.0.1:8080",
    "http://localhost:8081", "http://127.0.0.1:8081",
    "http://localhost:3000", "http://127.0.0.1:3000",
]
_origins_env = os.getenv("CORS_ORIGINS", "").strip()
allowed_origins = (
    [o.strip() for o in _origins_env.split(",") if o.strip()]
    if _origins_env else _default_origins
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    asin: str = Field(..., description="Amazon ASIN of the seed product")
    top_k: int = Field(10, ge=1, le=100)


class RecommendItem(BaseModel):
    asin: str
    title: str


class RecommendResponse(BaseModel):
    asin: str
    query_title: str
    recommendations: List[RecommendItem]
    inference_ms: float


class SearchItem(BaseModel):
    asin: str
    item_idx: int
    title: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchItem]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": recommender is not None,
        "catalogue_size": len(recommender.title_map) if recommender else 0,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    item_idx = recommender.asin_to_idx.get(req.asin)
    if item_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"ASIN '{req.asin}' not found in the trained model.",
        )

    t0 = time.perf_counter()
    try:
        raw = recommender.similar_items(item_idx, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    inference_ms = (time.perf_counter() - t0) * 1000

    return RecommendResponse(
        asin=req.asin,
        query_title=recommender.title_map.get(item_idx, "Unknown"),
        recommendations=[RecommendItem(asin=r["asin"], title=r["title"]) for r in raw],
        inference_ms=round(inference_ms, 2),
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Title substring to search for"),
    limit: int = Query(10, ge=1, le=50),
):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    q_lower = q.lower()
    matches: List[SearchItem] = []
    for idx, title in recommender.title_map.items():
        if q_lower in title.lower():
            matches.append(SearchItem(
                asin=recommender.idx_to_asin.get(int(idx), ""),
                item_idx=int(idx),
                title=title,
            ))
        if len(matches) >= limit:
            break

    return SearchResponse(query=q, results=matches)
