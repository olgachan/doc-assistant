import pickle
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def build_index():
    print("🔨 Строим индекс...")
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_dataset.csv"))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['cleaned_text'].tolist())
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    print(f"✅ Индекс построен: {len(df)} документов")
    return {'index': index, 'df': df, 'model': model}

def rag_answer(index_state, question, k=3):
    if index_state is None:
        return "Индекс не загружен", []
    question_embedding = index_state['model'].encode([question])
    distances, indices = index_state['index'].search(
        np.array(question_embedding).astype('float32'), k
    )
    sources = []
    context_parts = []
    for idx in indices[0]:
        if idx < len(index_state['df']):
            row = index_state['df'].iloc[idx]
            sources.append({'text': row['cleaned_text'], 'label': row['label']})
            context_parts.append(row['cleaned_text'])
    context = "\n".join(context_parts)
    answer = f"Найдено {len(sources)} документов:\n\n{context}"
    return answer, sources
