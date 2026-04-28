import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# 1. Загрузка данных
print("Загрузка данных... Пожалуйста, подожди, файл большой.")

# noinspection PyArgumentList
data = pd.read_csv('air_data.csv')

# Целевая переменная: Arithmetic Mean (уровень озона)
y = data['Arithmetic Mean']

# Признаки (Features): выбираем только те колонки, которые содержат числа
features = ['Latitude', 'Longitude', 'AQI', 'Observation Count', 'Observation Percent']
X = data[features]

# Разделяем на обучающую и тестовую выборки (80% учим, 20% проверяем)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# 2. Обработка пропусков (Imputation)
# Если в данных есть пустые ячейки (NaN), заполним их средним значением
imputer = SimpleImputer(strategy='mean')
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
imputed_X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 3. Обучение модели
print(f"Начинаю обучение на {len(X_train)} строках. Это может занять около минуты...")
# n_jobs=-1 заставит процессор работать на максимум, ускоряя процесс
model = RandomForestRegressor(n_estimators=50, random_state=1, n_jobs=-1)
model.fit(imputed_X_train, y_train)

# 4. Получение результата
print("Оценка точности...")
preds = model.predict(imputed_X_test)
mae = mean_absolute_error(y_test, preds)

print("=" * 40)
print(f"Готово! Средняя ошибка (MAE): {mae:.6f}")
print("Это значит, что в среднем модель ошибается лишь на такую величину при предсказании уровня озона.")
print("=" * 40)