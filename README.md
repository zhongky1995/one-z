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

## Manual Test Plan
1. Start the server: `uvicorn app:app --reload` and open `http://localhost:8000` in a browser.
2. Enter a prompt and valid settings, then click **Generate**. A result should appear and the error area should be empty.
3. Stop the server or disable the network and press **Generate** again. The new **Error** section displays the error message from the failed request.
