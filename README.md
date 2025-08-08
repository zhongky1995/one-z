# one-z

Simple web editor demonstrating AI-assisted long text generation with segment memory.

## Features
- Stores generated segments with vector embeddings for retrieval.
- Supports multiple providers (OpenAI, Azure OpenAI, 通义千问, DeepSeek, Kimi) with configurable base URL and API key.
- Web interface for entering prompts and viewing results.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn app:app --reload
   ```
3. Open `http://localhost:8000` in a browser and configure provider settings.

This is a minimal demo; integrate with real APIs by providing valid `base_url` and `api_key`.

## Testing
Run unit tests with:
```bash
pytest
```
