import os
import json
from gigachat import GigaChat
from gigachat.models import Messages, Chat
from src.tools import TOOLS, TOOL_SCHEMAS

# ЖЕСТКО ПРОПИСЫВАЕМ КЛЮЧИ ДЛЯ ПРОВЕРКИ
CLIENT_ID = "01a11afc-9016-74d6-ade0-80838e1fc300"
CLIENT_SECRET = "0f63193e-e4a2-43da-a2e0-466d988e17e6"
SCOPE = "GIGACHAT_API_PERS"
MODEL = "GigaChat-2"


def run_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'"
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"


def run_agent_native(user_query: str, max_iterations: int = 5) -> str:
    llm = GigaChat(
        credentials=CLIENT_SECRET,
        client_id=CLIENT_ID,          # <--- ДОБАВИТЬ ЭТУ СТРОКУ
        scope=SCOPE,
        verify_ssl_certs=False,
        model=MODEL
    )

    messages = [Messages(role="user", content=user_query)]

    for step in range(max_iterations):
        print(f"\n[ШАГ {step+1}] Отправляем запрос в GigaChat...")

        # tools, а не functions — этот параметр принимает OpenAI-формат схем
        response = llm.chat(Chat(messages=messages, tools=TOOL_SCHEMAS))
        msg = response.choices[0].message

        # Нет tool_calls — финальный ответ
        if not msg.tool_calls:
            print(f"[ФИНАЛ] Ответ модели: {msg.content}")
            return msg.content

        # ВАЖНО: добавляем в историю сообщение ассистента с tool_calls,
        # иначе API не примет tool-результаты на следующем шаге
        messages.append(msg)

        for call in msg.tool_calls:
            func_name = call.function.name
            try:
                func_args = json.loads(call.function.arguments) \
                    if call.function.arguments else {}
            except json.JSONDecodeError:
                func_args = {}

            print(f"[ДЕЙСТВИЕ] Вызов инструмента: {func_name}, аргументы: {func_args}")
            result = run_tool(func_name, func_args)
            print(f"[РЕЗУЛЬТАТ] {result[:150]}...")

            messages.append(Messages(
                role="tool",
                content=result,
                tool_call_id=call.id,
            ))

    return "Не удалось получить ответ за отведенное число шагов."


if __name__ == "__main__":
    query = "Узнай общее количество документов в базе, а затем умножь это число на 2."
    print(f"ЗАПРОС: {query}")
    final_answer = run_agent_native(query)
    print(f"\nИТОГОВЫЙ ОТВЕТ: {final_answer}")