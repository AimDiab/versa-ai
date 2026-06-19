"""
Versa AI — FastAPI application entry point.

Start the server:
    uvicorn api.main:app --reload --port 8000

The app exposes a single route family:
    POST /api/chat   — stream a response for a user query
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db.client import close_pool, setup_db
from api.routes.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up the DB connection pool on startup and close it on shutdown."""
    await setup_db()
    yield
    await close_pool()


app = FastAPI(
    title="Versa AI",
    description="Seed-on-First-Query conversational RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the React dev server to call the API during local development.
# Tighten this to a specific domain before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(chat_router, prefix="/api")
