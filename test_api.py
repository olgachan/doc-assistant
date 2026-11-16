import requests

BASE = "http://127.0.0.1:8000"

r = requests.get(f"{BASE}/health")
print("health:", r.status_code, r.json())

r = requests.post(f"{BASE}/ask", json={"question": "О чём документы?", "k": 3})
print("ask:", r.status_code, str(r.json())[:150])

r = requests.post(f"{BASE}/ask", json={"question": "   ", "k": 3})
print("пустой вопрос:", r.status_code, r.json())