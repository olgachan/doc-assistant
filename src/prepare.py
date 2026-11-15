import os
import re
import pickle
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ========== 0. НАСТРОЙКА ПУТИ К ПАПКЕ ==========
# Используем r перед строкой для корректной обработки слешей в Windows
DATA_DIR = r"C:\Users\user\doc-assistant\data"

# Автоматически создаем папку, если её еще нет
os.makedirs(DATA_DIR, exist_ok=True)
print(f"📁 Папка для данных готова: {DATA_DIR}")
# ========== 1. ЗАГРУЗКА ДАТАСЕТА ==========
file_name = "train.csv"
dataset_dir = kagglehub.dataset_download("thedevastator/new-dataset-for-text-classification-ag-news")
full_csv_path = os.path.join(dataset_dir, file_name)

try:
    df = pd.read_csv(full_csv_path, encoding='utf-8')
except Exception:
    try:
        df = pd.read_csv(full_csv_path, encoding='windows-1251')
    except Exception:
        df = pd.read_csv(full_csv_path, encoding='cp1251')

print(f"✅ Датасет загружен: {df.shape}")

text_column = 'text' if 'text' in df.columns else df.columns[0]
label_column = 'label' if 'label' in df.columns else df.columns[-1]

# ========== 2. ОЧИСТКА ТЕКСТА ==========
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========== 3. ПОДГОТОВКА ДАННЫХ ==========
new_cleaned_rows = []
new_labels = []
skipped_rows = 0

for raw_text, raw_label in zip(df[text_column], df[label_column]):
    cleaned = clean_text(raw_text)
    if len(cleaned) >= 50:
        skipped_rows += 1
        if skipped_rows > 5000:
            if len(new_cleaned_rows) >= 500:
                break
            new_cleaned_rows.append(cleaned)
            new_labels.append(int(raw_label) - 1)

df_fine_tune = pd.DataFrame({'cleaned_text': new_cleaned_rows, 'label': new_labels})
df_fine_tune = df_fine_tune.dropna(subset=['label'])
df_fine_tune = df_fine_tune[(df_fine_tune['label'] >= 0) & (df_fine_tune['label'] <= 3)].reset_index(drop=True)

print(f"✅ Очищенный датасет: {df_fine_tune.shape}")
print(df_fine_tune.head())

# ========== 3.1. СОХРАНЕНИЕ ОЧИЩЕННОГО ДАТАСЕТА В data ==========
cleaned_csv_path = os.path.join(DATA_DIR, "cleaned_dataset.csv")
df_fine_tune.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
print(f"💾 Очищенный датасет сохранён: {cleaned_csv_path}")

# ========== 4. ВЕКТОРИЗАЦИЯ (TF-IDF) ==========
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_new_matrix = vectorizer.fit_transform(df_fine_tune['cleaned_text']).toarray()
X_new_df = pd.DataFrame(X_new_matrix, columns=vectorizer.get_feature_names_out())

# Сохраняем vectorizer в data
vectorizer_path = os.path.join(DATA_DIR, "vectorizer.pkl")
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"💾 Vectorizer сохранён: {vectorizer_path}")

# ========== 5. РАЗДЕЛЕНИЕ НА TRAIN/TEST ==========
X_ft_train, X_ft_test, y_ft_train, y_ft_test = train_test_split(
    X_new_df, df_fine_tune['label'], test_size=0.2, random_state=42
)

print(f"✅ Train: {X_ft_train.shape}, Test: {X_ft_test.shape}")

# ========== 5.1. СОХРАНЕНИЕ TRAIN/TEST В data ==========
train_data = X_ft_train.copy()
train_data['label'] = y_ft_train.values
test_data = X_ft_test.copy()
test_data['label'] = y_ft_test.values

train_data.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
test_data.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)
print("💾 Train/Test сплиты сохранены в data/")