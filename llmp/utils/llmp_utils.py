# ASGENT 0 SIDE NOT RAG
from pydantic import BaseModel
import requests
import os
from typing import Optional, List, Dict
from requests.exceptions import RequestException

llmp_url = os.getenv("LLMP_URL")
llmp_password = os.getenv("LLMP_PASSWORD")
print (llmp_url)

class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: Optional[dict] = None
    image: Optional[str] = None
    tools: Optional[List[Dict]] = None
    src: str = None
    temperature: float = 0.5

def llmp_call(prompt, system_prompt, model,temperature=0.5,src=None,format=None):
        """ 
        Call the LLMP API to generate a response
        All related to the call is processed here
        """
        
        headers = {
        "Content-Type": "application/json",
        "Authorization": llmp_password
    }
        
        # Construct request payload
        request_data = GenerateRequest(
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        tools=None,
        src=src,
        temperature=temperature,
        format = format)

        payload = request_data.model_dump(exclude_none=True)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] Sending request to: {llmp_url}")
                print(f"[DEBUG] Payload: {payload}")
                response = requests.post(llmp_url, headers=headers, json=payload)
                response.raise_for_status()  # raises HTTPError for bad HTTP responses (e.g., 500)
                return response.json()
            except RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if hasattr(response, 'text'):
                    print(f"[DEBUG] Response content: {response.text}")
                if attempt == max_retries - 1:
                    raise Exception("Max retries exceeded. Unable to get a valid response.")
        
def llmp_list_call():
        """ 
        Call the LLMP API to get list of models
        Returns a dictionary with model details
        """
        
        # Use the same base URL as llmp_call
        if llmp_url:
            base_url = llmp_url.rsplit('/', 1)[0]  # Remove /generate from the URL
        else:
            base_url = "http://localhost:8000"
            
        endpoint = "/models"
        url = base_url + endpoint

        # If your API requires a token for authentication, set it here.
        # Adjust the header key and token as needed.
        headers = {
            "Authorization": llmp_password
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            models = response.json()
            print("Available models:", models)
            return models  # Return the full dictionary
        except Exception as e:
            print("Error fetching models:", e)
            return {}