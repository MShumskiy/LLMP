# LLMP - Local LLM Platform

A FastAPI-based platform for serving local Large Language Models (LLMs) with authentication, database logging, and comprehensive API endpoints.

## 🌟 Features

- **FastAPI-based API**: High-performance REST API for LLM interactions
- **OpenAI-compatible `/v1` API**: multi-turn messages, tool calls, streaming and images through the stock `openai` SDK, on both Ollama and OpenRouter models
- **Ollama Integration**: Seamless integration with Ollama for running local LLMs
- **Authentication & Authorization**: named API keys (bare or `Bearer`) and an IP/CIDR allowlist
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
   # Optional named keys, one per caller; the name is logged with every /v1 call
   LLMP_KEY_VEGA=another_secret_key
   # Addresses and CIDR ranges; "*" allows any address, unset allows loopback only
   ALLOWED_IPS=127.0.0.1,192.168.1.0/24

   # OpenRouter
   ORT_API_KEY=your_openrouter_key

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

   **Option 1: Use the startup scripts (Recommended)**
   
   For Windows users, use the convenient startup scripts that handle both Ollama and LLMP:
   
   - **Batch Script**: Double-click `start_llmp.bat` or run from command prompt
     ```batch
     start_llmp.bat
     ```
   
   - **PowerShell Script**: Run in PowerShell (more robust with error handling)
     ```powershell
     .\start_llmp.ps1
     ```

   **Option 2: Manual startup**
   ```bash
   # Terminal 1: Start Ollama
   ollama serve
   
   # Terminal 2: Start LLMP
   poetry run uvicorn llmp.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   **Option 3: Using poetry script**
   ```bash
   poetry run llmp-api
   ```

## 🎯 Startup Scripts

LLMP includes convenient startup scripts that automatically handle both Ollama and LLMP initialization:

### `start_llmp.bat` (Windows Batch)
- ✅ **Simple & Fast**: Easy double-click execution
- ✅ **Dependency Checking**: Verifies Ollama and Poetry are installed
- ✅ **Auto .env Creation**: Creates template if missing
- ✅ **Background Ollama**: Runs Ollama server in minimized window
- ✅ **Clean Startup**: Handles service initialization automatically

### `start_llmp.ps1` (PowerShell)
- ✅ **Advanced Features**: Comprehensive error handling and cleanup
- ✅ **Health Checks**: Waits for Ollama to be ready before starting LLMP
- ✅ **Colored Output**: Clear status messages and progress indicators
- ✅ **Graceful Shutdown**: Properly stops both services on Ctrl+C
- ✅ **Smart Retry Logic**: Automatically retries Ollama connection

### What the Scripts Do:
1. **Check Prerequisites**: Verify Ollama and Poetry installation
2. **Environment Setup**: Create `.env` template if missing
3. **Start Ollama**: Launch `ollama serve` in background
4. **Health Verification**: Ensure Ollama is responding (PowerShell only)
5. **Start LLMP**: Run the FastAPI server with hot reload
6. **Display Info**: Show server URLs and API documentation links

### Usage:
```bash
# Windows Batch (double-click or command prompt)
start_llmp.bat

# PowerShell (recommended)
.\start_llmp.ps1
```

**Access Points:**
- 🌐 **API Server**: http://localhost:8000
- 📖 **API Documentation**: http://localhost:8000/docs
- 🔍 **API Schema**: http://localhost:8000/redoc

## 🔧 API Endpoints

### Authentication
All endpoints require the `Authorization` header with an API key, either bare (`Authorization: <key>`) or as `Authorization: Bearer <key>`, which is what the OpenAI SDK sends. Valid keys are `authentication_key` (logged as `default`) and every `LLMP_KEY_<NAME>` in `.env` (logged as `<name>`). A missing key gets 401, a wrong key 403. The caller's IP must also match `ALLOWED_IPS`.

### OpenAI-compatible API (`/v1`)

Point any OpenAI client at `http://<llmp-host>:8000/v1`. Model ids are `<provider>/<model>`: `ollama/gpt-oss:20b`, `openrouter/qwen/qwen3-235b-a22b-2507`.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="<key>")
r = await client.chat.completions.create(
    model="ollama/gpt-oss:20b",
    messages=[{"role": "user", "content": "Hi"}],
    tools=[...],                       # standard tool definitions
    stream=True,                       # server-sent events; the last chunk carries usage
    extra_body={"llmp": {"src": "vega", "session_id": "abc", "num_ctx": 32768, "keep_alive": "30m"}},
)
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat Completions: `messages` (system, user, assistant with `tool_calls`, tool), `tools`, `tool_choice`, `stream`, `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `response_format`, `reasoning`. Images go in as `image_url` parts with a `data:` URL. |
| `GET /v1/models` | Models from both providers with `context_window`, `max_output`, `capabilities` (`tools`, `vision`, `reasoning`, `json_schema`) and `pricing` (USD per million tokens). Cached for 10 minutes; `?refresh=1` forces a refresh. OpenRouter models are filtered by the allowlist in `llmp.toml`. |
| `GET /v1/models/{id}` | One model from the catalog. |
| `GET /v1/health` | `{status, providers: {ollama, openrouter}, db}`, each `ok` or `down`. |

The optional `llmp` body field takes `src`, `session_id`, `run_id`, `agent`, `tags` (stored in `chat_log`), `keep_alive`, `num_ctx` and `options` (Ollama only), and `log` (`full`, `last_message` or `meta`, see below).

Responses are standard Chat Completions plus an `llmp` object: `{provider, gen_id, log_id, latency_ms, ttft_ms}`, and for Ollama also `load_ms`, `num_ctx` and `keep_alive`. OpenRouter responses include `usage.cost`.

How the providers are handled:

- **OpenRouter** requests are passed through with the prefix stripped and `usage.include` set, so the cost comes back. `reasoning` and `cache_control` parts pass through unchanged.
- **Ollama** requests go to the native `/api/chat`. Data-URL images become `images`, tool-call arguments come back as JSON strings with ids `call_<n>`, `reasoning` becomes `think`, and the model's thinking comes back as `reasoning`. Calls use `keep_alive` 30m and `num_ctx` 32768 by default (both set in `llmp.toml`), so the model stays loaded between agent steps.
- At most 2 Ollama and 16 OpenRouter calls run at once (also set in `llmp.toml`); further calls queue.

`scripts/smoke_v1.py` checks a deployment end to end with the OpenAI SDK: a two-step tool round trip (streaming and not) per model, or an image description with `--image`.

### Legacy endpoints

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
  "id": "llama3.2:latest_2025-10-27T12:00:00",
  "object": "chat.completion",
  "created": 1761566400,
  "model": "llama3.2:latest",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Why don't scientists trust atoms? Because they make up everything!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35 },
  "response_metadata": {
    "provider": "ollama",
    "src": "example_app",
    "timestamp": "2025-10-27T12:00:00",
    "system_prompt": "You are a helpful AI assistant.",
    "prompt": "Tell me a joke",
    "temperature": 0.7,
    "load_duration": 0.5,
    "prompt_eval_duration": 0.3,
    "eval_duration": 1.2
  }
}
```

Set `"provider": "openrouter"` to send the prompt to an OpenRouter model instead; `response_metadata` then has `duration` and `cost`.

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

print(response['choices'][0]['message']['content'])
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

Every `/v1` chat call writes a row to `chat_log`, created on first use. Rows are written by a background thread through a connection pool, so logging never slows a call down. Columns: `id`, `ts`, `key_name`, `src`, `session_id`, `run_id`, `agent`, `tags`, `client_ip`, `provider`, `model`, `stream`, `log_mode`, `gen_id`, `request`, `response`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens`, `cost`, `latency_ms`, `ttft_ms`, `status` (`ok`, `error` or `cancelled`), `http_status`, `error`.

How much of the request is stored depends on `llmp.log` (default from `llmp.toml`):

- `full`: the whole request.
- `last_message`: only the messages added since the previous call with the same `session_id`, plus the length and a hash of the earlier messages. Agent loops resend the whole history on every step, so this keeps storage linear. Without a `session_id`, only the last message is stored.
- `meta`: parameters and metrics only.

Image data is replaced by its hash and size in every mode.

The legacy `/generate` path uses a `generation_history_v4` table with the following structure:

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
│   ├── main.py             # FastAPI application, legacy routes
│   ├── v1.py               # OpenAI-compatible /v1 routes
│   ├── auth.py             # API keys and IP allowlist
│   ├── config.py           # Settings from .env and llmp.toml
│   ├── chat_log.py         # chat_log table and background writer
│   ├── providers/          # Ollama and OpenRouter adapters
│   ├── model_operator.py   # Legacy /generate implementation
│   └── utils/
│       └── llmp_utils.py   # Client utilities for API calls
├── llmp.toml               # Non-secret settings: defaults, limits, OpenRouter allowlist
├── scripts/smoke_v1.py     # End-to-end check with the OpenAI SDK
├── tests/                  # Offline tests (poetry run pytest)
├── start_llmp.bat          # Windows batch startup script
├── start_llmp.ps1          # PowerShell startup script (recommended)
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