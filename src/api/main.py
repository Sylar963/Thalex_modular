import sys
import os

# Add project root and thalex_py to sys.path BEFORE local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "thalex_py"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from .dependencies import init_dependencies, close_dependencies

# Load environment variables
load_dotenv()

from .v1.endpoints import market, portfolio, simulation, config, aggregated, signals


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_dependencies()
    yield
    # Shutdown
    await close_dependencies()


app = FastAPI(
    title="Thalex Modular API",
    description="API for PNL Simulation and Market Metrics",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
origins = [
    "http://localhost:5173",  # SvelteKit default
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(market.router, prefix="/api/v1/market", tags=["Market"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])
app.include_router(config.router, prefix="/api/v1/config", tags=["Configuration"])
app.include_router(aggregated.router, prefix="/api/v1/aggregated", tags=["Aggregated"])
app.include_router(signals.router, prefix="/api/v1/signals", tags=["Signals"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "thalex-modular-api"}
