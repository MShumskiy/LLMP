import logging
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from llmp import v1
from llmp.auth import authenticate
from llmp.config import settings
from llmp.model_operator import ModelOperatorOllama

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llmp = v1.start_runtime(settings)
    yield
    await v1.stop_runtime(app.state.llmp)


app = FastAPI(title="LLMP", lifespan=lifespan)
app.include_router(v1.router)


# /v1 errors use the OpenAI body shape ({"error": {...}}); the legacy routes keep FastAPI's.
@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/v1/"):
        kind = {401: "authentication_error", 403: "permission_error", 404: "not_found_error"}.get(
            exc.status_code, "invalid_request_error")
        error = {"message": str(exc.detail), "type": kind, "param": None, "code": None}
        return JSONResponse({"error": error}, status_code=exc.status_code, headers=exc.headers)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/v1/"):
        message = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        error = {"message": message, "type": "invalid_request_error", "param": None, "code": None}
        return JSONResponse({"error": error}, status_code=400)
    return await request_validation_exception_handler(request, exc)


# ── Legacy API (/generate, /models) ────────────────────────────────────────────

_model_operator = None
_model_operator_lock = threading.Lock()


def get_model_operator() -> ModelOperatorOllama:
    """Created on first use, so /v1 still starts when the legacy path can't connect."""
    global _model_operator
    with _model_operator_lock:
        if _model_operator is None:
            _model_operator = ModelOperatorOllama()
        return _model_operator


class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: dict = None
    image: str = None
    tools: list[dict] = None
    src: str = None
    temperature: float = 0.5
    max_gen_lenght: int = -1
    provider: str = "ollama"   # "ollama" | "openrouter"


@app.get("/models", dependencies=[Depends(authenticate)])
def list_models():
    """
    Endpoint to get list of available models.
    """
    return get_model_operator().list_models()

@app.post("/generate", dependencies=[Depends(authenticate)])
def generate_response(request: Request, body: GenerateRequest):
    """
    Endpoint to generate a response.
    provider: "ollama" (default) or "openrouter"
    """
    ip_address = request.client.host
    try:
        model_operator = get_model_operator()
        if body.provider == "openrouter":
            response = model_operator.generate_openrouter(
                model=body.model,
                system_prompt=body.system_prompt,
                prompt=body.prompt,
                format=body.format,
                ip_address=ip_address,
                src=body.src,
                temperature=body.temperature,
            )
        else:
            response = model_operator.generate_response(
                model=body.model,
                system_prompt=body.system_prompt,
                prompt=body.prompt,
                format=body.format,
                image=body.image,
                tools=body.tools,
                ip_address=ip_address,
                src=body.src,
                temperature=body.temperature,
                max_gen_lenght=body.max_gen_lenght,
            )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")

def run():
    uvicorn.run("llmp.main:app", host="0.0.0.0", port=8000, reload=False)
