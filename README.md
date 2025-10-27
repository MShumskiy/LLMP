# LLMP - Local LLM Platform

A FastAPI-based platform for serving local Large Language Models (LLMs) with authentication, database logging, and comprehensive API endpoints.

## 🌟 Features

- **FastAPI-based API**: High-performance REST API for LLM interactions
- **Ollama Integration**: Seamless integration with Ollama for running local LLMs
- **Authentication & Authorization**: API key and IP-based access control
- **Database Logging**: PostgreSQL database for storing generation history and analytics
- **Multi-modal Support**: Text and image processing capabilities
- **Tool Support**: Function calling and structured output generation
- **Docker Support**: Containerized deployment ready
- **Analytics**: Built-in analysis tools for usage patterns and performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- PostgreSQL database
- Ollama running locally with models installed

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LLMP
   ```

2. **Install dependencies using Poetry**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # Ollama Configuration
   url=http://localhost:11434/api/

   # Authentication
   authentication_key=your_secret_key_here
   ALLOWED_IPS=127.0.0.1,192.168.1.0/24

   # Database Configuration
   db_user=your_db_user
   db_password=your_db_password
   db_host=localhost
   db_port=5432
   db_database=llmp_db

   # Client Configuration (for utils)
   LLMP_URL=http://localhost:8000/generate
   LLMP_PASSWORD=your_secret_key_here
   ```

4. **Start the server**
   ```bash
   poetry run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

   Or using the poetry script:
   ```bash
   poetry run llmp-api
   ```

## 🔧 API Endpoints

### Authentication
All endpoints require authentication via the `Authorization` header with your API key.

### Available Endpoints

#### `GET /models`
List all available Ollama models with their details.

**Response:**
```json
{
  "llama3.2:latest": {
    "model_name": "llama3.2:latest",
    "param_size": "3.2B",
    "quant_level": "Q4_0"
  }
}
```

#### `POST /generate`
Generate responses from LLM models.

**Request Body:**
```json
{
  "model": "llama3.2:latest",
  "system_prompt": "You are a helpful AI assistant.",
  "prompt": "Tell me a joke",
  "temperature": 0.7,
  "src": "example_app",
  "max_gen_lenght": -1,
  "format": null,
  "image": null,
  "tools": null
}
```

**Response:**
```json
{
  "model": "llama3.2:latest",
  "message": {
    "content": "Why don't scientists trust atoms? Because they make up everything!"
  },
  "prompt_eval_count": 15,
  "eval_count": 20,
  "load_duration": 0.5,
  "prompt_eval_duration": 0.3,
  "eval_duration": 1.2,
  "gen_id": "llama3.2:latest_2025-10-27T12:00:00",
  "src": "example_app",
  "temperature": 0.7
}
```

## 🛠️ Usage Examples

### Using the Python Client

```python
from llmp.utils.llmp_utils import llmp_call

# Simple text generation
response = llmp_call(
    prompt="Explain quantum computing in simple terms",
    system_prompt="You are a science teacher explaining concepts to high school students.",
    model="llama3.2:latest",
    temperature=0.5,
    src="education_app"
)

print(response['message']['content'])
```

### Using cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: your_secret_key_here" \
  -d '{
    "model": "llama3.2:latest",
    "system_prompt": "You are a helpful AI assistant.",
    "prompt": "What is the capital of France?",
    "temperature": 0.5,
    "src": "test_app"
  }'
```

### Tool Usage Example

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: your_secret_key_here" \
  -d '{
    "model": "llama3.2:latest",
    "system_prompt": "You are a calculator assistant.",
    "prompt": "Add 15 and 27",
    "tools": [{
      "type": "function",
      "function": {
        "name": "add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
          "type": "object",
          "properties": {
            "num1": {"type": "number", "description": "First number"},
            "num2": {"type": "number", "description": "Second number"}
          },
          "required": ["num1", "num2"]
        }
      }
    }]
  }'
```

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t llmp .

# Run the container
docker run -p 8000:8000 --env-file .env llmp
```

### Docker Compose (Recommended)

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  llmp:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - postgres
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: llmp_db
      POSTGRES_USER: llmp_user
      POSTGRES_PASSWORD: your_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

## 📊 Analytics and Monitoring

The platform includes built-in analytics capabilities:

- **Generation History**: All API calls are logged to PostgreSQL
- **Performance Metrics**: Response times, token counts, and model usage
- **Usage Analytics**: Track usage patterns by source applications
- **Analysis Notebook**: Use `analysis.ipynb` for detailed analytics

### Database Schema

The platform automatically creates a `generation_history_v4` table with the following structure:

- `id`: Unique UUID for each generation
- `src`: Source application identifier
- `gen_id`: Generation ID (model + timestamp)
- `gen_timestamp`: When the generation occurred
- `caller_address`: IP address of the caller
- `model`: Model used for generation
- `system_prompt`: System prompt used
- `prompt`: User prompt
- `gen_text`: Generated response
- `prompt_eval_count`: Number of tokens in prompt
- `eval_count`: Number of tokens generated
- `load_duration`: Model loading time (seconds)
- `prompt_eval_duration`: Prompt processing time (seconds)
- `eval_duration`: Generation time (seconds)
- `temperature`: Temperature parameter used

## 🔒 Security Features

- **API Key Authentication**: Secure access control
- **IP Whitelisting**: Restrict access to specific IP addresses/ranges
- **Request Validation**: Pydantic-based input validation
- **Error Handling**: Comprehensive error responses without sensitive data exposure

## 🏗️ Architecture

```
LLMP/
├── llmp/                    # Main package
│   ├── main.py             # FastAPI application and routes
│   ├── model_operator.py   # Ollama integration and database operations
│   └── utils/
│       └── llmp_utils.py   # Client utilities for API calls
├── app.py                  # Application entry point
├── analysis.ipynb         # Analytics and monitoring notebook
├── example_usage.ipynb    # Usage examples and documentation
├── Dockerfile             # Container configuration
├── pyproject.toml         # Project dependencies and configuration
└── README.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Connection refused errors**
   - Ensure Ollama is running on the specified URL
   - Check firewall settings for the specified ports

2. **Database connection issues**
   - Verify PostgreSQL is running and accessible
   - Check database credentials in `.env` file
   - Ensure the database exists or has proper permissions for creation

3. **Authentication failures**
   - Verify the API key in requests matches the `.env` configuration
   - Check that the client IP is in the allowed IPs list

4. **Model not found errors**
   - Ensure the requested model is installed in Ollama
   - Use `/models` endpoint to list available models

### Performance Optimization

- Use appropriate `temperature` values (0.1-0.9)
- Set `max_gen_lenght` to limit response length
- Consider model size vs. performance trade-offs
- Monitor database performance for high-volume usage

## 📞 Support

For issues, questions, or contributions, please open an issue in the GitHub repository.

---

**Author**: Mykola Shumskiy  
**Email**: mykola_shumskiy@outlook.pt  
**Version**: 0.1.0