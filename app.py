from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from src.model_operator import ModelOperatorOllama

load_dotenv()
auth_key = os.getenv('authentication_key')
allowed_ips_env = os.getenv("ALLOWED_IPS")
allowed_ips = allowed_ips_env.split(",")
allowed_ips = {ip for ip in allowed_ips}

app = FastAPI()

# initialize model operator
model_operator = ModelOperatorOllama()

# Authentication dependency
def authenticate(request: Request,
                 credentials: str = Header(None,alias='Authorization')):
    
    client_ip = request.client.host
    
    if credentials is None:
        raise HTTPException(status_code = 401, detail = 'Missing credentials')
    if credentials != auth_key:
        raise HTTPException(status_code = 403, detail = 'Invalid credentials')
    if client_ip not in allowed_ips:
        raise HTTPException(status_code=403, detail=f"Access denied for IP: {client_ip}")
    
    return credentials

class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: dict = None
    image: str = None
    tools:list[dict] = None
    src: str = None
    temperature: float = 0.5
    

@app.get("/models", dependencies=[Depends(authenticate)])
def list_models():
    """
    Endpoint to get list of available models.
    """
    return model_operator.list_models()

@app.post("/generate", dependencies=[Depends(authenticate)])
def generate_response(request: Request, body: GenerateRequest):
    """
    Endpoint to generate a response.
    """
    
    ip_address = request.client.host
    try:
        response = model_operator.generate_response(
            model=body.model,
            system_prompt=body.system_prompt,
            prompt=body.prompt,
            format=body.format,
            image=body.image,
            tools=body.tools,
            ip_address=ip_address,
            src=body.src,
            temperature = body.temperature
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")
    