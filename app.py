from typing import Dict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from memory import MemoryStore
from clients import generate_text, get_embedding

app = FastAPI()
store = MemoryStore()


class GenerateRequest(BaseModel):
    prompt: str
    provider: str
    base_url: str
    api_key: str
    model: str
    embedding_model: str


@app.post("/generate")
def generate(req: GenerateRequest) -> Dict[str, object]:
    cfg = req.dict()
    query_emb = get_embedding(req.prompt, cfg)
    related = store.search(query_emb)
    context = "\n".join(related + [req.prompt])
    result = generate_text(context, cfg)
    store.add(result, get_embedding(result, cfg))
    return {"result": result, "related": related}


@app.get("/")
def index() -> HTMLResponse:
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
