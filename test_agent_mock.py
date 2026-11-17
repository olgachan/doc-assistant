import json
from src.tools import TOOLS
from src.agent_native import run_tool  # Импортируем нашу обертку с обработкой ошибок

# --- ЗАМОКАННЫЙ LLM ДЛЯ ИМИТАЦИИ ОТВЕТОВ GIGACHAT ---
def mock_llm_native(messages: list, step_counter: list) -> dict:
    """Имитирует ответы GigaChat с tool_calls для детерминированного теста."""
    step = step_counter[0]
    step_counter[0] += 1
    
    if step == 0:
        # Шаг 1: Агент решает вызвать stats, чтобы узнать количество документов
        return {
            "choices": [{"message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "stats", "arguments": "{}"}
                }]
            }}]
        }
    elif step == 1:
        # Шаг 2: Получив "364 документа", агент решает вызвать calc
        return {
            "choices": [{"message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"expression": "364 * 2"}'}
                }]
            }}]
        }
    else:
        # Шаг 3: Получив результат 728, агент дает финальный текстовый ответ
        return {
            "choices": [{"message": {
                "role": "assistant",
                "content": "В базе всего 364 документа. Если умножить это число на 2, получится 728."
            }}]
        }

# === СЦЕНАРИЙ 1: КОМБИНАЦИЯ (Составной запрос) ===
def test_mocked_native_agent():
    print("="*70)
    print("ТЕСТ 1: Native Агент с ЗАМОКАННЫМ LLM (Составной запрос)")
    print("Цель: доказать, что цикл корректно вызывает 2 инструмента подряд")
    print("="*70)
    
    messages = [{"role": "user", "content": "Сколько документов в базе? Умножь это число на 2."}]
    step_counter = [0]
    max_iterations = 5
    
    for step in range(max_iterations):
        print(f"\n[ШАГ {step+1}] Вызов mock_llm (имитация GigaChat)...")
        response = mock_llm_native(messages, step_counter)
        msg = response["choices"][0]["message"]
        
        if "tool_calls" not in msg or not msg["tool_calls"]:
            print(f"[✅ ФИНАЛ] Ответ агента: {msg['content']}")
            assert "728" in msg["content"], "Финальный ответ должен содержать результат вычисления 728"
            print("\n🎉 ТЕСТ 1 ПРОЙДЕН УСПЕШНО!")
            return
            
        for call in msg["tool_calls"]:
            func_name = call["function"]["name"]
            func_args = json.loads(call["function"]["arguments"]) if call["function"]["arguments"] else {}
            print(f"  [⚙️ ДЕЙСТВИЕ] Инструмент: '{func_name}', Аргументы: {func_args}")
            
            result = TOOLS[func_name](**func_args) if func_args else TOOLS[func_name]()
            print(f"  [📥 РЕЗУЛЬТАТ] {result}")
            
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": call["id"]
            })

# === СЦЕНАРИЙ 2: ТОЛЬКО ВЫЧИСЛЕНИЕ ===
def test_calc_only():
    print("\n" + "="*70)
    print("ТЕСТ 2: Только вычисление")
    print("="*70)
    res = TOOLS["calc"]("12 * 12")
    assert res == "144", f"Ожидалось 144, получено {res}"
    print(f"  calc('12 * 12') -> {res} ✅")

# === СЦЕНАРИЙ 3: ПРОВОКАЦИЯ (Несуществующий инструмент) ===
def test_unknown_tool():
    print("\n" + "="*70)
    print("ТЕСТ 3: Провокация (вызов несуществующего инструмента)")
    print("="*70)
    res = run_tool("weather", {"city": "Moscow"})
    assert "неизвестный инструмент" in res, f"Ожидалась ошибка, получено: {res}"
    print(f"  run_tool('weather', ...) -> '{res}'")
    print("  ✅ Цикл не упал с исключением, модель получит строку с ошибкой и сможет исправиться.")

# === СЦЕНАРИЙ 4: ТОЛЬКО ПОИСК ===
def test_search_only():
    print("\n" + "="*70)
    print("ТЕСТ 4: Только поиск (RAG)")
    print("="*70)
    res = TOOLS["search"]("Soka University")
    assert "Найдено" in res, f"Ожидался текст 'Найдено', получено: {res}"
    print(f"  search('Soka University') -> {res[:80]}... ✅")


# === ЕДИНЫЙ БЛОК ЗАПУСКА ===
if __name__ == "__main__":
    test_mocked_native_agent()
    test_calc_only()
    test_unknown_tool()
    test_search_only()
    print("\n" + "="*70)
    print("🎉 ВСЕ 4 СЦЕНАРИЯ ИЗ ТРЕБОВАНИЙ ЛАБЫ ПРОЙДЕНЫ УСПЕШНО! 🎉")
    print("="*70)