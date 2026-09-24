import os
import re
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Импортируем PyTorch библиотеки
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Указываем путь к предустановленным базам данных nltk в Kaggle
nltk.data.path.append("/usr/share/nltk_data")

# ==========================================
# 1. Загружаем датасет с Kaggle
# ==========================================
file_path = "train.csv" 

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS, 
    "thedevastator/new-dataset-for-text-classification-ag-news", 
    file_path
)

text_column = 'text' if 'text' in df.columns else df.columns[0]

print(f"Используется колонка датасета: {text_column}")
print("Исходный размер датасета:", df.shape)

# ==========================================
# 2. Функция для очистки текста
# ==========================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # ИЗМЕНЕНИЕ ЗДЕСЬ: убрали '0-9' из шаблона. 
    # Теперь остаются только буквы (латиница/кириллица) и пробелы. Цифры удаляются.
    text = re.sub(r'[^a-zа-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 3. Фильтрация и отбор первых 500 подходящих строк
# ==========================================
cleaned_rows = []
quantity_rows = 500 # сколько строк хотим взять

for raw_text in df[text_column]:
    if len(cleaned_rows) >= quantity_rows:
        break
        
    cleaned = clean_text(raw_text)
    if len(cleaned) >= 50:
        cleaned_rows.append(cleaned)

cln_df = pd.DataFrame(cleaned_rows, columns=['cleaned_text'])
print(f"Собрано строк из исходного датасета: {len(cln_df)}")

# ==========================================
# 4. Поиск .txt файлов и добавление их содержимого
# ==========================================
txt_rows = []
search_dir = "."  # ДИРЕКТОРИЯ ОТКУДА БРАТЬ .TXT
txt_files_found = 0

print(f"\n--- Сканирование {search_dir} на наличие .txt файлов ---")

if os.path.exists(search_dir):
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files_found += 1
                file_full_path = os.path.join(root, file)
                file_saved_lines = 0  
                
                try:
                    with open(file_full_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            cleaned_line = clean_text(line)
                            if len(cleaned_line) >= 50:
                                txt_rows.append(cleaned_line)
                                file_saved_lines += 1
                    print(f" Найдено: '{file}' -> Сохранено строк: {file_saved_lines}")
                except Exception as e:
                    print(f" Ошибка при чтении файла '{file}': {e}")
else:
    print(f" Директория {search_dir} не найдена. Убедитесь, что код запущен в Kaggle.")

if txt_files_found == 0:
    print(f" В {search_dir} файлов .txt не обнаружено.")
else:
    print(f"Всего обработано .txt файлов из {search_dir}: {txt_files_found}")

if txt_rows:
    txt_df = pd.DataFrame(txt_rows, columns=['cleaned_text'])
    cln_df = pd.concat([cln_df, txt_df], ignore_index=True)
    print(f"Суммарно добавлено строк из всех .txt файлов: {len(txt_rows)}")

# Разметка данных
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    import zipfile
    vader_zip_path = '/usr/share/nltk_data/sentiment/vader_lexicon.zip'
    if os.path.exists(vader_zip_path):
        with zipfile.ZipFile(vader_zip_path, 'r') as zip_ref:
            zip_ref.extractall('/tmp/vader_lexicon')
        nltk.data.path.append('/tmp/vader_lexicon')
        sia = SentimentIntensityAnalyzer()
    else:
        raise FileNotFoundError("Не удалось найти встроенный vader_lexicon в Kaggle.")

# ==========================================
# 5. Функция разметки
# ==========================================
def auto_label(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0  # нейтральный
    
    score = sia.polarity_scores(text)['compound']
    
    if score >= 0.05:
        return 2  # positive
    elif score <= -0.05:
        return 1  # negative
    else:
        return 0  # neutral

# Применяем разметку
print("\nЗапуск автоматической разметки...")
cln_df['label'] = cln_df['cleaned_text'].apply(auto_label)

print("\nРаспределение полученных классов:")
print(cln_df['label'].value_counts())

# Сохранение промежуточного текстового результата
cln_df.to_csv('cleaned.csv', index=False, encoding='utf-8')
print("Файл 'cleaned.csv' успешно сохранен!")


# ==========================================
# 6. БИНАРНАЯ ВЕКТОРИЗАЦИЯ (Топ-1000 слов, БЕЗ ЦИФР)
# ==========================================
print("\nИнициализация бинарного векторизатора (топ-1000 самых частых слов, значения 0 или 1)...")
vectorizer = CountVectorizer(max_features=1000, binary=True)

# Векторизация текстов
binary_matrix = vectorizer.fit_transform(cln_df['cleaned_text'])

# Получение списка 1000 самых частых слов (названия столбцов)
feature_names = vectorizer.get_feature_names_out()
print(f"Векторизация завершена! Размер матрицы: {binary_matrix.shape}")
print(f"Количество признаков (слов): {len(feature_names)}")
print(f"Примеры первых 10 самых частых слов: {list(feature_names[:10])}")

# Преобразуем разреженную матрицу в DataFrame
df_final = pd.DataFrame(binary_matrix.toarray(), columns=feature_names)

# Сохранение векторизованного CSV на диск
output_file_csv = '/kaggle/working/vectorized_cybersecurity_data_binary.csv'
df_final.to_csv(output_file_csv, index=False, encoding='utf-8')
print(f"\n✅ Итоговый файл успешно сохранен в формате CSV: {output_file_csv}")
print(f"✅ В файле {len(df_final)} строк и {len(df_final.columns)} столбцов (только 0 и 1)")

# Проверка содержимого сохраненного CSV
print("\n--- Проверка содержимого сохраненного CSV (первые 3 строки, первые 5 столбцов) ---")
df_check = pd.read_csv(output_file_csv, nrows=3)
print(df_check.iloc[:, :5].to_string())
print("-----------------------------------------------------------------------------------")


# ==========================================
# 7. Создание PyTorch Dataset и DataLoader
# ==========================================
class AgentTextDataset(Dataset):
    def __init__(self, features_df, labels_series):
        self.states = torch.tensor(features_df.values, dtype=torch.float32)
        self.labels = torch.tensor(labels_series.values, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]

# Создаем загрузчик данных с батч-сайзом 32
torch_dataset = AgentTextDataset(df_final, cln_df['label'])
train_loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)


# ==========================================
# 8. Архитектура нейросети на PyTorch
# ==========================================
class AgentClassifierNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(AgentClassifierNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 9. Инициализация и цикл обучения
# ==========================================
input_features_count = df_final.shape[1] 

model = AgentClassifierNetwork(input_dim=input_features_count, num_classes=3)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n--- Запуск процесса обучения нейросети PyTorch ---")
epochs = 5
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    correct_predictions = 0
    
    for batch_states, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_states)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_states.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = (correct_predictions / len(train_loader.dataset)) * 100
    print(f"Эпоха [{epoch+1}/{epochs}] | Ошибка (Loss): {epoch_loss:.4f} | Точность (Accuracy): {epoch_acc:.2f}%")

print("\n✅ Скрипт полностью выполнен! Пайплайн от сырого текста до векторизации и обучения сети PyTorch завершен.")