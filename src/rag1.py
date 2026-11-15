import os
import base64
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from gigachat import GigaChat
from gigachat.models import Messages

# ========== КОНФИГУРАЦИЯ GIGACHAT ==========
GIGA_CLIENT_ID = "01a11afc-9016-74d6-ade0-80838e1fc300"
GIGA_CLIENT_SECRET = "0f63193e-e4a2-43da-a2e0-466d988e17e6"
GIGA_SCOPE = "GIGACHAT_API_PERS"
MODEL_NAME = "GigaChat-2"

# Кодируем credentials в base64 (требование SDK GigaChat)
credentials_b64 = base64.b64encode(
    f"{GIGA_CLIENT_ID}:{GIGA_CLIENT_SECRET}".encode('utf-8')
).decode('utf-8')

# Инициализация клиента
giga_client = GigaChat(
    credentials=credentials_b64,
    scope=GIGA_SCOPE,
    verify_ssl_certs=False,
    model=MODEL_NAME
)
print("✅ Клиент GigaChat инициализирован")

# ========== BERT ЭМБЕДДЕР ==========
class BERTEncoder:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        self.model = SentenceTransformer(self.model_name).to(self.device)

        if self.device == 'cuda':
            self.model.half()

        print(f"✅ BERT-модель загружена: {self.model_name} на {self.device.upper()}")

    def encode(self, texts, batch_size=32, show_progress=True):
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        total = len(texts)

        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_texts = texts[i:i+batch_size]

                features = self.model.tokenize(batch_texts)
                features = batch_to_device(features, self.device)
                out_features = self.model.forward(features)
                embeddings = out_features['sentence_embedding']
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

                if show_progress and (i // batch_size) % 5 == 0:
                    progress = min(i + batch_size, total)
                    print(f"   BERT эмбеддинг: {progress}/{total} ({100*progress/total:.1f}%)")

        return np.concatenate(all_embeddings, axis=0)

    def encode_query(self, query):
        return self.encode([query], batch_size=1, show_progress=False)[0]

bert_encoder = BERTEncoder()
print("✅ Ячейка 2 выполнена успешно!")
import os
import re
import kagglehub
import pandas as pd

# 1. Скачивание датасета с Kaggle
print("⏳ Скачивание датасета...")
dataset_dir = kagglehub.dataset_download("thedevastator/new-dataset-for-text-classification-ag-news")
full_csv_path = os.path.join(dataset_dir, "train.csv")

# 2. Загрузка в Pandas (с автоопределением кодировки)
try:
    df = pd.read_csv(full_csv_path, encoding='utf-8')
except Exception:
    df = pd.read_csv(full_csv_path, encoding='windows-1251')

print(f"Исходный размер датасета: {df.shape}")

# Определяем названия колонок
text_column = 'text' if 'text' in df.columns else df.columns[0]
label_column = 'label' if 'label' in df.columns else df.columns[-1]

# 3. Функция очистки текста
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Применяем очистку
df['cleaned_text'] = df[text_column].apply(clean_text)
df = df[df['cleaned_text'].str.len() >= 50].reset_index(drop=True)

# 4. ⚡ ВАЖНО: Берём маленькую выборку для быстрого теста!
# 100 статей дадут нам около 150-200 чанков. Эмбеддинги для них создадутся за 1 секунду.
# (Когда система заработает, можно будет увеличить n=5000 или убрать эту строку вовсе)
df = df.sample(n=100, random_state=42).reset_index(drop=True)

print(f"✅ Датасет подготовлен! Размер выборки: {df.shape}")
print(f"Категории в выборке: {sorted(df[label_column].unique())}")
def chunk_text(text, chunk_size=500, overlap=50):
    """Разбиение текста на чанки по словам с перекрытием"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

all_chunks = []
chunk_metadata = []

print("⏳ Разбиение текстов на чанки...")

for doc_idx, row in df.iterrows():
    # Разбиваем очищенный текст на чанки
    chunks = chunk_text(row['cleaned_text'], chunk_size=500, overlap=50)

    for chunk_idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({
            'doc_id': int(doc_idx),
            'chunk_id': chunk_idx,
            'label': int(row[label_column])
        })

print(f"✅ Чанкинг завершён!")
print(f"Всего создано чанков: {len(all_chunks)}")
print(f"Пример первого чанка:\n'{all_chunks[0][:150]}...'\n")
print("⏳ Генерация BERT-эмбеддингов для чанков...")

# Используем класс BERTEncoder, который мы создали в Ячейке 2
chunk_embeddings = bert_encoder.encode(
    texts=all_chunks,
    batch_size=32,
    show_progress=True
)

print(f"✅ Эмбеддинги успешно созданы!")
print(f"Форма матрицы эмбеддингов: {chunk_embeddings.shape}")
print(f"Размерность одного вектора (BERT): {chunk_embeddings.shape[1]}")
import faiss
import pickle

# 1. Определяем размерность векторов (для mpnet-base-v2 это 768)
dimension = chunk_embeddings.shape[1]

# 2. Создаём индекс FAISS (Inner Product для нормализованных векторов = косинусное сходство)
faiss_index = faiss.IndexFlatIP(dimension)

# 3. Добавляем наши эмбеддинги в индекс
faiss_index.add(chunk_embeddings.astype(np.float32))

print(f"✅ FAISS индекс создан! Векторов в базе: {faiss_index.ntotal}")

# 4. Сохраняем индекс и метаданные на диск (чтобы не пересчитывать при перезапуске)
faiss.write_index(faiss_index, 'rag_faiss_index.bin')
with open('rag_chunks.pkl', 'wb') as f:
    pickle.dump({'chunks': all_chunks, 'metadata': chunk_metadata}, f)

print("✅ Индекс и чанки сохранены в файлы: 'rag_faiss_index.bin' и 'rag_chunks.pkl'")
def search_chunks(query, top_k=5):
    """Поиск top-k наиболее релевантных чанков по вопросу"""
    # 1. Превращаем вопрос в BERT-вектор
    query_emb = bert_encoder.encode_query(query)

    # 2. Ищем похожие векторы в FAISS индексе
    scores, indices = faiss_index.search(query_emb.reshape(1, -1).astype(np.float32), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                'chunk_text': all_chunks[idx],
                'score': float(score),
                'metadata': chunk_metadata[idx]
            })
    return results

# Быстрый тест
test_results = search_chunks("What is the latest technology news?", top_k=2)
print(f"✅ Функция поиска работает! Найдено совпадений: {len(test_results)}")
from gigachat.models import Chat, Messages

def rag_query_giga(question, top_k=3):
    """Полный цикл RAG: Поиск -> Строгий Промпт -> GigaChat -> Ответ + Токены"""

    # 1. Поиск контекста через BERT
    chunks = search_chunks(question, top_k=top_k)

    # 2. Формирование контекста и списка источников
    context_parts = []
    sources_info = []

    for i, c in enumerate(chunks):
        context_parts.append(f"[{i+1}] (Score: {c['score']:.3f}): {c['chunk_text']}")
        sources_info.append({
            'source_id': i+1,
            'doc_id': c['metadata']['doc_id'],
            'label': c['metadata']['label'],
            'score': c['score'],
            'text_preview': c['chunk_text'][:100] + "..."
        })

    context = "\n\n".join(context_parts)

    # 3. ЖЁСТКИЙ ПРОМПТ НА АНГЛИЙСКОМ
    prompt = f"""You are a strict factual assistant. Answer the user's question STRICTLY and ONLY using the provided context.

RULES:
1. Your answer MUST be in English.
2. If the context does not contain a direct, explicit answer to the question, you MUST reply with EXACTLY this phrase: "No direct answer found in the provided context."
3. DO NOT guess, synthesize unrelated facts, or use outside knowledge.
4. If you find the answer, cite the source numbers in brackets at the end, e.g., [1], [2].

Context:
{context}

User Question: {question}

Your Answer:"""

    # 4. Запрос к GigaChat API
    try:
        chat_request = Chat(
            messages=[Messages(role="user", content=prompt)],
            temperature=0.1,          # Снижаем до 0.1 для максимальной строгости и фактологичности
            max_tokens=150,           # Ограничиваем длину ответа, чтобы модель не начала "лить воду"
            profanity_check=False
        )

        response = giga_client.chat(chat_request)
        answer = response.choices[0].message.content

        # 5. ПОДСЧЁТ ТОКЕНОВ
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        return {
            'question': question,
            'answer': answer,
            'sources': sources_info,
            'tokens': {
                'prompt': prompt_tokens,
                'completion': completion_tokens,
                'total': total_tokens
            }
        }

    except Exception as e:
        return {'error': f"Ошибка вызова GigaChat: {e}"}


# ========== ИНТЕРАКТИВНЫЙ РЕЖИМ ==========
print("=" * 70)
print("🚀 RAG СИСТЕМА (BERT + GigaChat-2) ОБНОВЛЕНА!")
print("Теперь отвечает строго на английском и только по фактам из датасета.")
print("Введите вопрос (или 'exit' для выхода)")
print("=" * 70)

while True:
    try:
        question = input("\n❓ Ваш вопрос (на английском): ").strip()
        if question.lower() in ['exit', 'выход', 'q', 'quit']:
            print("👋 Завершение работы.")
            break
        if not question:
            continue

        print("⏳ Поиск через BERT и генерация ответа...")
        result = rag_query_giga(question, top_k=3)

        if 'error' in result:
            print(result['error'])
            continue

        print("\n" + "="*70)
        print(f"💬 ОТВЕТ GIGACHAT:\n{result['answer']}")
        print("="*70)

        print("\n📚 ИСТОЧНИКИ (найдено BERT):")
        for s in result['sources']:
            # Словарь категорий для наглядности
            labels_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            label_name = labels_map.get(s['label'], str(s['label']))
            print(f"  [{s['source_id']}] Doc:{s['doc_id']}, Category:{label_name}, Score:{s['score']:.3f}")
            print(f"      {s['text_preview']}\n")

        print("📊 РАСХОД ТОКЕНОВ GIGACHAT:")
        print(f"  • Прочитано (промпт + контекст): {result['tokens']['prompt']} ток.")
        print(f"  • Сгенерировано (ответ):         {result['tokens']['completion']} ток.")
        print(f"  • ВСЕГО ПОТРАЧЕНО:               {result['tokens']['total']} ток.")
        print("-" * 70)

    except KeyboardInterrupt:
        print("\n👋 Завершение работы.")
        break
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

