import os
import re
import json
from gigachat import GigaChat
from gigachat.models import Messages, Chat
from src.tools import TOOLS

CLIENT_SECRET = os.getenv("GIGA_CLIENT_SECRET", "0f63193e-e4a2-43da-a2e0-466d988e17e6")
SCOPE = os.getenv("GIGA_SCOPE", "GIGACHAT_API_PERS")
MODEL = os.getenv("MODEL_NAME", "GigaChat-2")

SYSTEM_PROMPT = (
    "Ты полезный агент с доступом к инструментам: search(query), calc(expression), stats().\n"
    "На каждом шаге ты должен вернуть ТОЛЬКО валидный JSON без пояснений в формате:\n"
    '{"action": "search" | "calc" | "stats" | "final", "input": "аргументы или пусто", "answer": "финальный ответ если action=final"}\n'
    "Для action=final заполни поле answer. Для остальных действий заполни input и оставь answer пустым.\n"
    "Используй инструменты последовательно, пока не соберешь достаточно информации для финального ответа."
)

def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("В ответе модели нет JSON")
    return json.loads(match.group(0))

def run_tool_json(name: str, input_value: str) -> str:
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'. Доступны: {list(TOOLS.keys())}."
    try:
        # Для calc и search аргумент - строка, для stats - пусто
        if name == "stats":
            return TOOLS[name]()
        return TOOLS[name](input_value)
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"

def run_agent_json(user_query: str, max_steps: int = 5) -> str:
    llm = GigaChat(
        credentials=CLIENT_SECRET,
        client_id=CLIENT_ID,          # <--- ДОБАВИТЬ ЭТУ СТРОКУ
        scope=SCOPE,
        verify_ssl_certs=False,
        model=MODEL
    )
    history = f"Вопрос: {user_query}\n"
    
    for step_num in range(max_steps):
        print(f"\n[ШАГ {step_num+1}] Отправляем запрос в GigaChat (JSON-режим)...")
        
        prompt = SYSTEM_PROMPT + "\n\nИстория:\n" + history
        response = llm.chat(Chat(messages=[Messages(role="user", content=prompt)]))
        raw_text = response.choices[0].message.content
        print(f"[СЫРОЙ ОТВЕТ] {raw_text[:200]}...")
        
        try:
            step_data = extract_json(raw_text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[ОШИБКА ПАРСИНГА] {e}. Просим модель исправить.")
            history += f"Ошибка: ответ не парсится как JSON ({e}). Верни чистый JSON.\n"
            continue
            
        action = str(step_data.get("action", "")).strip().lower()
        
        if action == "final":
            final_answer = step_data.get("answer", "Модель не дала ответ")
            print(f"[ФИНАЛ] Ответ модели: {final_answer}")
            return final_answer
            
        input_value = str(step_data.get("input", ""))
        print(f"[ДЕЙСТВИЕ] Вызов инструмента: {action}, input: {input_value}")
        
        result = run_tool_json(action, input_value)
        print(f"[РЕЗУЛЬТАТ] {result[:150]}...")
        
        history += f"Действие: {action}({input_value}) -> {result}\n"
        
    return "Не удалось получить ответ за отведенное число шагов."

if __name__ == "__main__":
    query = "Узнай общее количество документов в базе, а затем умножь это число на 2."
    print(f"ЗАПРОС: {query}")
    final_answer = run_agent_json(query)
    print(f"\nИТОГОВЫЙ ОТВЕТ: {final_answer}")