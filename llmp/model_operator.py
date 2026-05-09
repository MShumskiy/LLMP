import requests
import json
import datetime
import psycopg2 as psql
from dotenv import load_dotenv
import os
from openai import OpenAI

# with db connection

class ModelOperatorOllama():
    def __init__(self):
        
        load_dotenv()
        # Ollama configs
        self.url = os.getenv('url')
        self.model_list = self.list_models()
        
        # db configs
        self.db_user = os.getenv('db_user')
        self.db_password = os.getenv('db_password')
        self.db_host = os.getenv('db_host')
        self.db_port = os.getenv('db_port')
        self.db_database = os.getenv('db_database')
        
        # db connection
        self.connection = self.connect_to_db()
        self.cursor = self.connection.cursor()
        # create table if not exists
        self.create_gen_hist_table()
        
        
    def connect_to_db(self):
        """
        Verify if database exists, if not, creates it.
        Connects to the database and returns the connection object.
        """
        try:
            
            # verify if db exists
            tmp_connection = psql.connect(user = self.db_user,
                            password = self.db_password,
                            host = self.db_host,
                            port = self.db_port,
                            database = self.db_database)
            
            tmp_connection.autocommit = True
            tmp_cursor = tmp_connection.cursor()
            tmp_cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.db_database}'")
            exists = tmp_cursor.fetchone()
            
            # if doesn't exist, create it
            if not exists:
                print(f"Database '{self.db_database}' not found. Creating it...")
                tmp_cursor.execute(f"CREATE DATABASE {self.db_database}")
                
            tmp_cursor.close()
            tmp_connection.close()
            
            # conenct to db
            connection = psql.connect(
                dbname=self.db_database,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            print(f"Connected to database: {self.db_database}")
            return connection
        
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise   
        
    def create_gen_hist_table(self):
        """
        Checks if gen_hist table exists, if not creates it.
        """
        table_creation_query = """
        CREATE TABLE IF NOT EXISTS generation_history_v4 (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            src TEXT,
            gen_id TEXT,
            gen_timestamp TEXT,
            caller_address TEXT,
            model TEXT,
            system_prompt TEXT,
            prompt TEXT,
            gen_text TEXT,
            prompt_eval_count INT,
            eval_count INT,
            load_duration FLOAT,
            prompt_eval_duration FLOAT,
            eval_duration FLOAT,
            temperature FLOAT
        );
        """
        self.cursor.execute(table_creation_query)
        
        
    def save_to_db(self, generation_data, ip_address=None):
        """
        Saves the generated response to PostgreSQL.
        
        Parameters:
        - generation_data (dict): JSON object containing generation details.
        """
        try:
            meta = generation_data.get("response_metadata", {})
            usage = generation_data.get("usage", {})
            content = (generation_data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            # Map JSON keys to table columns
            db_columns = {
                "src": meta.get("src"),
                "gen_id": generation_data.get("id"),
                "caller_address": ip_address,
                "gen_timestamp": meta.get("timestamp"),
                "model": generation_data.get("model"),
                "system_prompt": meta.get("system_prompt"),
                "prompt": (meta.get("prompt") or "").replace("\n", " "),
                "gen_text": content.replace("\n", " "),
                "prompt_eval_count": usage.get("prompt_tokens"),
                "eval_count": usage.get("completion_tokens"),
                "load_duration": meta.get("load_duration"),
                "prompt_eval_duration": meta.get("prompt_eval_duration"),
                "eval_duration": meta.get("eval_duration") or meta.get("duration"),
                "temperature": meta.get("temperature")
            }

            # Generate dynamic SQL query
            columns = ", ".join(db_columns.keys())
            placeholders = ", ".join(["%s"] * len(db_columns))
            values = tuple(db_columns.values())

            insert_query = f"""
            INSERT INTO generation_history_v4 ({columns})
            VALUES ({placeholders});
            """

            # Execute and commit the query
            self.cursor.execute(insert_query, values)
            self.connection.commit()
            print("Generation saved to database.")

        except Exception as e:
            self.connection.rollback()
            print(f"Failed to save generation to database: {e}")
        

    def generate_response(self,
                          model,
                          system_prompt,
                          prompt,
                          format=None,
                          image=None,
                          tools=None,
                          ip_address=None,
                          src = None,
                          temperature = 0.5,
                          max_gen_lenght = -1,
                          db_save = False
                          ):
        """
        Sends a request to the LLM API and returns the response.
        """
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        created = int(now.timestamp())
        if model in self.model_list:
            self.model = model
        else:
            raise ValueError(f"Model '{model}' not provided or not found in available models: {self.model_list}")
        
        
        payload = {
            "model": self.model,
            "keep_alive": 0,
            "messages": [
                {'role':'system',
                'content':system_prompt
                },
                {'role':'user',
                'content':prompt}
                ],
            "stream": False,
            "tools": tools,
            "format": format,
            "options": {
                "temperature": temperature,
                'num_predict' : max_gen_lenght
        }}
        
        if image:
            import base64
            filepath = rf"{image}"
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            # attach image to the *user* message
            payload["messages"][-1]["images"] = [encoded_string]
            #payload["model"] = 'llava:latest'
                
            #payload.update({'images':encoded_image})
        
        if format:
            payload.update(format)
            

        headers = {"Content-Type": "application/json"}
        
        print(f"[INFO] Sending request to Ollama: {self.url}chat")
        print(f"[INFO] Model: {self.model} | Temperature: {temperature}")
        response = requests.post(f"{self.url}chat", data=json.dumps(payload), headers=headers)
        print(f"[INFO] Response status: {response.status_code}")
        raw = response.json()

        content = raw.get("message", {}).get("content", "")
        prompt_tokens = raw.get("prompt_eval_count")
        completion_tokens = raw.get("eval_count")
        load_duration = round((raw.get("load_duration") or 0) / 1e9, 2)
        prompt_eval_duration = round((raw.get("prompt_eval_duration") or 0) / 1e9, 2)
        eval_duration = round((raw.get("eval_duration") or 0) / 1e9, 2)

        print(f"[INFO] Load: {load_duration}s | Prompt eval: {prompt_eval_duration}s | Generation: {eval_duration}s")

        result = {
            "id": f"{model}_{timestamp}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": raw.get("done_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            },
            "response_metadata": {
                "provider": "ollama",
                "src": src,
                "timestamp": timestamp,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "temperature": temperature,
                "load_duration": load_duration,
                "prompt_eval_duration": prompt_eval_duration,
                "eval_duration": eval_duration,
            },
        }

        if db_save:
            print("[INFO] Saving generation data to database...")
            self.save_to_db(result, ip_address)
        print("[INFO] Response generation complete")
        return result
        
    def generate_openrouter(self,
                            model,
                            system_prompt,
                            prompt,
                            format=None,
                            ip_address=None,
                            src=None,
                            temperature=0.5):
        """
        Sends a request to OpenRouter and returns a response dict
        with the same shape as generate_response().
        
        Parameters:
        - model: OpenRouter model string, e.g. 'qwen/qwen3-235b-a22b-2507'
        - system_prompt: system message content
        - prompt: user message content
        - format: optional dict; when provided, requests JSON output
                  (pass {"type": "json_object"} or a full json_schema dict)
        - ip_address: caller IP for logging
        - src: arbitrary source tag
        - temperature: sampling temperature
        """
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        created = int(now.timestamp())

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("ORT_API_KEY"),
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        if format:
            kwargs["response_format"] = format

        print(f"[INFO] Sending request to OpenRouter | Model: {model} | Temperature: {temperature}")
        _t0 = datetime.datetime.now()
        completion = client.chat.completions.create(**kwargs)
        duration_s = round((datetime.datetime.now() - _t0).total_seconds(), 3)
        print(f"[INFO] OpenRouter response received in {duration_s}s")

        content = completion.choices[0].message.content
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None
        cost = (usage.model_extra or {}).get("cost") if usage else None

        result = {
            "id": f"{model}_{timestamp}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": completion.choices[0].finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "response_metadata": {
                "provider": "openrouter",
                "src": src,
                "timestamp": timestamp,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "temperature": temperature,
                "duration": duration_s,
                "cost": cost,
            },
        }

        print("[INFO] OpenRouter generation complete")
        return result

    def list_models(self):
        
        response = requests.get(f'{self.url}tags')
        model_dict = {
            model['name']: {'model_name': model['name'],
                            'param_size': model['details']['parameter_size'],
                            'quant_level': model['details']['quantization_level']} for model in response.json()['models']
    }
        return model_dict