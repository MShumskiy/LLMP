# LLMP (LLM Platform)  
## Purpose 
This tool acts as a LLM gateway. Leveraging ollama's funcitonalities with added custom functionalities.  
## Usage  
1. Configure Postgres DB default on port 5432  
2. configure the following in .env file:  
- url = LLMP's url (default 'http://127.0.0.1:11435/api/') (str)  
- db_user = "llmp" (str)  
- db_password = your LLMP password (str)  
- db_host = your host (default localhost - "127.0.0.1")  
- db_port = "5432" (str)  
- db_database = name of postgres database (example: "llmp_db")   
- authentication_key = LLMP password as defined above  
- ALLOWED_IPS= selection of allowed IPs to you LLMP (example 192.168.1.219,192.168.0.104) (not strings)
3. run H:\projects\ai_based\LLMP> poetry run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
  
