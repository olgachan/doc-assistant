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

def rag_answer(query: str, index_state: dict, k: int = 3):
    try:
        # 1. Векторизуем запрос
        query_vec = index_state['model'].encode([query])
        
        # 2. Ищем в FAISS (приводим к float32 для надежности)
        D, I = index_state['index'].search(query_vec.astype('float32'), k)
        
        # 3. Безопасно извлекаем тексты
        df = index_state['df']
        sources = []
        answer_parts = []
        
        for idx in I[0]:
            if idx < len(df):
                row = df.iloc[idx]
                
                # ЗАЩИТА ОТ ОШИБКИ: проверяем тип данных строки
                if isinstance(row, dict):
                    text = str(row.get("text", row))
                    label = row.get("label", "N/A")
                else:
                    # Если это pandas Series
                    text = str(row.get("text", str(row.iloc[0]) if hasattr(row, 'iloc') else str(row)))
                    label = str(row.get("label", "N/A"))
                
                sources.append({"text": text, "label": label})
                answer_parts.append(text)
        
        answer = "\n\n".join(answer_parts)
        return answer, sources
        
    except Exception as e:
        return f"Ошибка при поиске: {e}", []