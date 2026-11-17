import json
from src.rag import build_index, rag_answer

# Глобальное состояние индекса (ленивая инициализация)
_index_state = None

def get_index_state():
    global _index_state
    if _index_state is None:
        print("[INFO] Построение индекса для инструмента search...")
        _index_state = build_index()
    return _index_state

# --- 1. Инструмент поиска (RAG) ---
def tool_search(query: str) -> str:
    """Поиск релевантных фрагментов в базе документов.
    Используй, когда нужен факт, определение или цитата из корпуса документов.
    Аргумент query: строка с вопросом или ключевыми словами."""
    try:
        state = get_index_state()
        answer, sources = rag_answer(query, state, k=3)
        return f"Найдено: {answer}\nИсточников: {len(sources)}"
    except Exception as e:
        return f"Ошибка поиска: {e}"

# --- 2. Инструмент вычислений ---
def tool_calc(expression: str) -> str:
    """Вычисляет арифметическое выражение (например, '2 * (3 + 4)').
    Используй ТОЛЬКО для чисел и операторов +, -, *, /, (), **."""
    try:
        # Безопасное вычисление для учебных целей
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления '{expression}': {e}"

# --- 3. Инструмент статистики ---
def tool_stats() -> str:
    """Возвращает общую статистику по базе знаний.
    Используй, когда спрашивают о количестве документов или классов. Не требует аргументов."""
    return "Всего документов в индексе: 364. Количество классов: 3."

# --- Реестр инструментов ---
TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "stats": tool_stats,
}

# --- Схемы для Native-подхода (GigaChat API) ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Поиск релевантных фрагментов в базе документов по запросу.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Текст запроса для поиска"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Вычисление математического выражения.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Математическое выражение, например '364 * 2'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stats",
            "description": "Получение общей статистики по базе знаний. Не требует аргументов.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]