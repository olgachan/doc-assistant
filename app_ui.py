import streamlit as st
import json
from src.tools import TOOLS

# --- ЗАГОЛОВОК ---
st.set_page_config(page_title="ИИ-Агент с инструментами", layout="wide")
st.title("🤖 ИИ-Агент с инструментами")
st.markdown("""
**Доступные инструменты:**
- 🔍 **search** — поиск по базе документов (RAG)
- 🧮 **calc** — вычисление арифметических выражений
- 📊 **stats** — статистика по базе знаний
""")

# --- БОКОВАЯ ПАНЕЛЬ С БЫСТРЫМИ СЦЕНАРИЯМИ ---
st.sidebar.header("⚡ Быстрые сценарии")
scenario = st.sidebar.radio(
    "Выберите сценарий:",
    ["Свободный вопрос", "Только поиск", "Только вычисление", "Комбинация (stats + calc)", "Провокация"]
)

# --- ГЛАВНАЯ ОБЛАСТЬ ---
if scenario == "Свободный вопрос":
    user_query = st.text_input("Задайте вопрос агенту:", "Сколько документов в базе? Умножь это число на 2.")
elif scenario == "Только поиск":
    user_query = st.text_input("Запрос для поиска:", "Soka University")
elif scenario == "Только вычисление":
    user_query = st.text_input("Арифметическое выражение:", "12 * 12")
elif scenario == "Комбинация (stats + calc)":
    user_query = "Узнай общее количество документов в базе, а затем умножь это число на 2."
elif scenario == "Провокация":
    user_query = "Какая погода в Москве?"

# --- КНОПКА ЗАПУСКА ---
if st.button("🚀 Запустить агента", type="primary"):
    st.subheader("📋 Лог выполнения")
    
    # Имитация работы агента (мок)
    messages = [{"role": "user", "content": user_query}]
    step_counter = [0]
    
    # Функция-мок для демонстрации
    def mock_llm(messages, step_counter):
        step = step_counter[0]
        step_counter[0] += 1
        
        if "погод" in user_query.lower():
            # Провокация — несуществующий инструмент
            return {"tool_calls": [{"id": "c1", "function": {"name": "weather", "arguments": '{"city": "Moscow"}'}}]}
        elif "документ" in user_query.lower() and "умнож" in user_query.lower():
            # Комбинация
            if step == 0:
                return {"tool_calls": [{"id": "c1", "function": {"name": "stats", "arguments": "{}"}}]}
            elif step == 1:
                return {"tool_calls": [{"id": "c2", "function": {"name": "calc", "arguments": '{"expression": "364 * 2"}'}}]}
            else:
                return {"content": "В базе 364 документа. 364 × 2 = 728."}
        elif "search" in user_query.lower() or scenario == "Только поиск":
            return {"tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": f'{{"query": "{user_query}"}}'}}]}
        elif scenario == "Только вычисление" or any(c in user_query for c in "+-*/"):
            return {"tool_calls": [{"id": "c1", "function": {"name": "calc", "arguments": f'{{"expression": "{user_query}"}}'}}]}
        else:
            return {"content": f"Ответ на вопрос: {user_query}"}
    
    # Цикл агента
    final_answer = None
    for step_num in range(5):
        response = mock_llm(messages, step_counter)
        
        if "tool_calls" not in response or not response["tool_calls"]:
            final_answer = response.get("content", "Нет ответа")
            st.success(f"✅ **Финальный ответ:** {final_answer}")
            break
        
        for call in response["tool_calls"]:
            func_name = call["function"]["name"]
            func_args = json.loads(call["function"]["arguments"]) if call["function"]["arguments"] else {}
            
            # Показываем шаг в UI
            with st.expander(f"🔹 Шаг {step_num + 1}: Вызов `{func_name}`", expanded=True):
                st.code(f"Инструмент: {func_name}\nАргументы: {func_args}", language="json")
                
                # Вызываем реальный инструмент
                try:
                    if func_args:
                        result = TOOLS[func_name](**func_args)
                    else:
                        result = TOOLS[func_name]()
                    st.info(f"📥 Результат: {result}")
                except Exception as e:
                    error_msg = f"❌ Ошибка: неизвестный инструмент '{func_name}'"
                    st.error(error_msg)
                    result = error_msg
                
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": call["id"]
                })
    
    if final_answer is None:
        st.warning("⚠️ Не удалось получить ответ за отведенное число шагов.")

# --- ИНФОРМАЦИЯ О ПРОЕКТЕ ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 О проекте
Это демонстрация ИИ-агента с тремя инструментами:
- **RAG-поиск** по базе из 364 документов
- **Калькулятор** для арифметики
- **Статистика** по датасету

Агент использует **ручной цикл** (Native-подход) с обработкой ошибок и тестами.
""")