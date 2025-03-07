import requests
import json
import datetime
import psycopg2 as psql
from dotenv import load_dotenv
import os

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
            # Map JSON keys to table columns
            db_columns = {
                "src":generation_data.get("src"),
                "gen_id": generation_data.get("gen_id"),
                "caller_address": ip_address,
                "gen_timestamp": generation_data.get("timestamp"),
                "model": generation_data.get("model"),
                "system_prompt": generation_data.get("system_prompt"),
                "prompt": generation_data.get("prompt").replace("\n", " "),
                "gen_text": generation_data.get("message", {}).get("content").replace("\n", " "),
                "prompt_eval_count": generation_data.get("prompt_eval_count"),
                "eval_count": generation_data.get("eval_count"),
                "load_duration": generation_data.get("load_duration"),
                "prompt_eval_duration": generation_data.get("prompt_eval_duration"),
                "eval_duration": generation_data.get("eval_duration"),
                "temperature": generation_data.get("temperature")
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
        

    def generate_response(self,model,system_prompt,prompt,format=None,image=None,tools=None, ip_address=None, src = None, temperature = 0.5):
        """
        Sends a request to the LLM API and returns the response.
        """
        timestamp = datetime.datetime.now().isoformat()
        timestamp = str(timestamp)
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
            "options": {
                "temperature": temperature
        }}
        
        if image:
            import base64
            
            encoded_image = []
            filepath = rf"{image}"
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                encoded_image.append(encoded_string)
            
            payload["messages"][1]["images"] = encoded_image
            payload["model"] = 'llava:latest'
                
            #payload.update({'images':encoded_image})
        
        if format:
            payload.update(format)
            

        headers = {"Content-Type": "application/json"}
        
        response = requests.post(f"{self.url}chat", data=json.dumps(payload), headers=headers)
        
        # get generation timestamp
        
        response_json = response.json()
        response_json.update({'system_prompt':system_prompt})
        response_json.update({'prompt':prompt})
        response_json.update({'timestamp':timestamp})
        response_json['load_duration'] = round(response_json['load_duration']/(10**9),2)
        response_json['prompt_eval_duration'] = round(response_json['prompt_eval_duration']/(10**9),2)
        response_json['eval_duration'] = round(response_json['eval_duration']/(10**9),2)
        response_json['gen_id'] = f'{response_json['model']}_{response_json['timestamp']}'
        response_json['src'] = src
        response_json['temperature'] = temperature
        
        print('saving data')
        self.save_to_db(response_json,ip_address)

        return response_json
        
    def list_models(self):
        
        response = requests.get(f'{self.url}tags')
        model_dict = {
            model['name']: {'model_name': model['name'],
                            'param_size': model['details']['parameter_size'],
                            'quant_level': model['details']['quantization_level']} for model in response.json()['models']
    }
        return model_dict