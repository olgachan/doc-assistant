from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag import build_index, rag_answer

import numpy as np
from fastapi import encoders as _encoders

_orig_jsonable = _encoders.jsonable_encoder

def _patched_jsonable(obj, **kwargs):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _orig_jsonable(obj, **kwargs)

_encoders.jsonable_encoder = _patched_jsonable

app = FastAPI(title="Doc Assistant")
index_state = None

@app.on_event("startup")
def load():
    global index_state
    index_state = build_index()
    print("✅ Индекс загружен!")

@app.get("/health")
def health():
    return {"status": "ok"}

class Query(BaseModel):
    question: str
    k: int = 3

@app.post("/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
    answer, sources = rag_answer(index_state, q.question, q.k)
    return {"answer": answer, "sources": sources}
