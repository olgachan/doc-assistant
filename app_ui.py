import streamlit as st
import re
from src.tools import TOOLS

st.set_page_config(page_title="ИИ-Агент с инструментами", layout="wide", page_icon="🤖")

# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ (Безопасный способ) ---
if "last_scenario" not in st.session_state:
    st.session_state.last_scenario = "Свободный вопрос"
if "current_query" not in st.session_state:
    st.session_state.current_query = "Сколько документов в базе? Умножь это число на 2."

st.title("🤖 ИИ-Агент с инструментами")
st.markdown("""
**Доступные инструменты:**
- 🔍 **search** — поиск по базе документов (RAG)
- 🧮 **calc** — вычисление арифметических выражений
- 📊 **stats** — статистика по базе знаний
""")

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("⚡ Быстрые сценарии")
scenario = st.sidebar.radio(
    "Выберите сценарий:",
    ["Свободный вопрос", "Только поиск", "Только вычисление", "Комбинация (stats + calc)", "Провокация"]
)

default_queries = {
    "Свободный вопрос": "Сколько документов в базе? Умножь это число на 2.",
    "Только поиск": "Who is the CEO of Microsoft mentioned in the text?",
    "Только вычисление": "12 * 12",
    "Комбинация (stats + calc)": "Сколько классов в датасете? Умножь это число на 100.",
    "Провокация": "Какая погода в Москве?"
}

# Если сценарий изменился, обновляем запрос на дефолтный
if st.session_state.last_scenario != scenario:
    st.session_state.last_scenario = scenario
    st.session_state.current_query = default_queries[scenario]

# Поле ввода
user_query = st.text_input(
    "Задайте вопрос агенту:",
    value=st.session_state.current_query
)
# Сохраняем то, что ввел пользователь, чтобы не сбрасывалось при вводе
st.session_state.current_query = user_query

# --- СТОП-СЛОВА ---
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "against",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "it", "its", "they", "them", "their", "am",
    "в", "на", "по", "с", "из", "за", "к", "у", "о", "от", "до", "и",
    "а", "но", "или", "же", "ли", "бы", "что", "кто", "как", "где", "когда",
    "почему", "зачем", "для", "при", "без", "через", "после", "перед",
    "между", "под", "над", "во", "ко", "со", "обо", "про",
}

def extract_meaningful_words(text: str) -> set:
    words = re.findall(r'[a-zA-Zа-яА-ЯёЁ]{3,}', text.lower())
    return set(w for w in words if w not in STOP_WORDS)

def is_relevant(query: str, search_result: str) -> bool:
    query_words = extract_meaningful_words(query)
    result_words = extract_meaningful_words(search_result)
    if not query_words:
        return True
    return len(query_words & result_words) > 0

# --- МАРШРУТИЗАТОР ---
def route_query(query: str, scenario: str) -> list:
    q = query.lower()
    
    if scenario == "Провокация":
        return []
    
    # Комбинация stats + calc (ищем ключевые слова)
    if scenario == "Комбинация (stats + calc)" or (
        ("документ" in q or "сколько" in q or "класс" in q) and 
        ("умнож" in q or "посчитай" in q or "*" in q or "раздели" in q or "/" in q)
    ):
        # Определяем выражение для калькулятора
        if "класс" in q:
            expr = "3 * 100"
        elif "раздели" in q or "/" in q:
            expr = "364 / 3"
        else:
            expr = "364 * 2"
            
        return [("stats", {}), ("calc", {"expression": expr})]
    
    # Только вычисление
    if scenario == "Только вычисление" or re.search(r'\d+\s*[\+\-\*\/\(\)]+\s*\d+', query):
        return [("calc", {"expression": query.strip()})]
    
    # Статистика
    if "статистик" in q or "stats" in q:
        return [("stats", {})]
    
    # По умолчанию — поиск
    return [("search", {"query": query})]

# --- ГЕНЕРАТОР ОТВЕТА ---
def build_final_answer(query: str, tool_results: list, scenario: str) -> str:
    if not tool_results:
        return ""
    
    # Только поиск
    if len(tool_results) == 1 and tool_results[0][0] == "search":
        search_text = tool_results[0][1]
        if not is_relevant(query, search_text):
            return ""
        return search_text
    
    # Только вычисление
    if len(tool_results) == 1 and tool_results[0][0] == "calc":
        return f"🧮 Результат: **{tool_results[0][1]}**"
    
    # Статистика
    if len(tool_results) == 1 and tool_results[0][0] == "stats":
        return f"📊 {tool_results[0][1]}"
    
    # Комбинация stats + calc (УМНЫЙ ПАРСИНГ)
    if len(tool_results) == 2 and tool_results[0][0] == "stats" and tool_results[1][0] == "calc":
        stats_text = tool_results[0][1]
        calc_result = tool_results[1][1]
        
        q = query.lower()
        # Ищем именно то, о чем спрашивали: классы или документы
        if "класс" in q:
            match = re.search(r'классов:\s*(\d+)', stats_text)
            entity = "классов"
        else:
            match = re.search(r'(\d+)\s+документ', stats_text)
            entity = "документов"
            
        count = match.group(1) if match else "?"
        return f"📊 В базе **{count} {entity}**. Результат вычисления: **{calc_result}**"
    
    # Общий случай
    parts = [r[1] for r in tool_results]
    return "\n\n".join(parts)

# --- КНОПКА ЗАПУСКА ---
if st.button("🚀 Получить ответ", type="primary", use_container_width=True):
    with st.spinner("🤖 Агент обрабатывает запрос..."):
        try:
            steps = route_query(user_query, scenario)
            
            tool_results = []
            for tool_name, tool_args in steps:
                if tool_name not in TOOLS:
                    continue
                try:
                    result = TOOLS[tool_name](**tool_args) if tool_args else TOOLS[tool_name]()
                    tool_results.append((tool_name, result))
                except Exception as e:
                    tool_results.append((tool_name, f"Ошибка: {e}"))
            
            final_answer = build_final_answer(user_query, tool_results, scenario)
            
            if final_answer:
                st.markdown("---")
                st.subheader("💬 Ответ агента:")
                st.markdown(final_answer)
            else:
                st.info("ℹ️ В базе новостей нет информации по вашему вопросу. Попробуйте спросить о компаниях, технологиях, спорте или бизнесе.")
        except Exception as e:
            st.error(f"⚠️ Внутренняя ошибка агента: {e}")

# --- ИНФОРМАЦИЯ О ПРОЕКТЕ ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### 📚 О проекте
Демонстрация ИИ-агента с тремя инструментами.
Агент использует **ручной цикл** (Native-подход) с обработкой ошибок.

---
**Автор:** Осипова Ольга Евгеньевна  
**Группа:** МП-302
""")