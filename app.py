"""
Entry point for the LLMP (Local LLM Platform) FastAPI application.
This file imports the FastAPI app from the main module for compatibility with uvicorn.
"""

from llmp.main import app

__all__ = ["app"]
