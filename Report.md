# Отчет по Практической работе №3. Построение сквозного ML-пайплайна на больших данных с помощью Spark MLlib.

**Студент:** [Быков Владимир Валерьевич]

**Группа:**[БД-251м, Магистратура "Бизнес-информатика"]

**Вариант:** 7

---

## 1. Введение

**Цель работы:** Построить модель машинного обучения для предсказания пола пользователя (`gender`) по данным о тренировках из датасета EndomondoHR.

**Бизнес-кейс:** Вы — Data Scientist в компании HealthTech (аналог Endomondo/Strava). 
Задача — разработать прогностические модели для улучшения удержания клиентов, персонализации предложений и автоматической классификации активности.

**Датасет:** `endomondoHR.json` содержит 253,020 записей о тренировках пользователей с массивами данных пульса (`heart_rate`), скорости (`speed`), 
высоты (`altitude`), а также информацией о виде спорта (`sport`) и поле (`gender`).


## 2. Предобработка данных

### 2.1 Очистка данных
Первым этапом выполнена фильтрация данных для обеспечения качества обучающей выборки:

```python
df_clean = (raw_df
    .filter(F.size(F.col('heart_rate')) >= 30)      # минимум 30 точек пульса
    .filter(F.size(F.col('speed')) >= 10)           # минимум 10 точек скорости
    .dropna(subset=['sport', 'gender'])              # обязательные признаки
    .filter(F.col('gender').isin('male', 'female'))  # только валидные значения
)
```

**Результат очистки:**

* Исходное количество записей: 253,020

* После очистки: 48,401 записей (19.1%)

* Распределение по полу:
*      Male: 45,029
*      Female: 3,372

### 2.2 Обработка пропусков
Анализ пропущенных значений показал, что поле `speed` содержит 80.4% пропусков, однако после фильтрации записей с размером `speed >= 10` проблема решена.

### 2.3 Конструирование признаков (Feature Engineering)
В соответствии с заданием 1, выполнена обработка массива heart_rate путем обрезки до первых N значений (N = 30):

```python
from pyspark.sql.types import ArrayType, LongType

def slice_array_udf(n):
    return F.udf(lambda arr: arr[:n] if arr else [], ArrayType(LongType()))

N_FIRST = 30
df_sliced = df_clean.withColumn(
    'heart_rate_sliced', 
    slice_array_udf(N_FIRST)(F.col('heart_rate'))
)
```
Из обрезанного массива созданы следующие признаки:

* `hr_mean_sliced` - Среднее значение пульса (первые 30 измерений)
* `hr_max_sliced` - Максимальное значение пульса
* `hr_min_sliced` - Минимальное значение пульса
* `hr_range_sliced` - Размах пульса (max - min)

Дополнительные признаки:

* `speed_mean`, `speed_max` — статистики скорости
* `alt_mean`, `alt_range` — статистики высоты
* `workout_length` — длина тренировки


## 3. Моделирование
### 3.1 Разбиение данных
Данные разделены на тренировочную и тестовую выборки в соотношении 80/20 со стратификацией по полу:

```python
f_train, f_test = df_features.filter(F.col('gender')=='female').randomSplit([0.8, 0.2], seed=42)
m_train, m_test = df_features.filter(F.col('gender')=='male').randomSplit([0.8, 0.2], seed=42)

train_df = f_train.union(m_train)
test_df  = f_test.union(m_test)

print(f'Train: {train_df.count():,}  |  Test: {test_df.count():,}')
print('\nTrain — распределение:')
train_df.groupBy('gender').count().show()
```
Результат:
Train: 38,692 записей (female: 2,671, male: 36,021)
Test: 9,709 записей

### 3.2 ML-трансформеры для Pipeline
```python
#Кодирование целевой переменной
gender_indexer = StringIndexer(inputCol='gender', outputCol='label', handleInvalid='skip')

#Кодирование вида спорта
sport_indexer = StringIndexer(inputCol='sport', outputCol='sport_idx', handleInvalid='keep')
sport_encoder = OneHotEncoder(inputCols=['sport_idx'], outputCols=['sport_ohe'])

#Числовые признаки
NUMERIC_FEATURES = [
    'hr_mean_sliced', 'hr_max_sliced', 'hr_min_sliced', 'hr_range_sliced',
    'speed_mean', 'speed_max',
    'alt_mean', 'alt_range',
    'workout_length'
]

#Сборка вектора признаков
assembler = VectorAssembler(
    inputCols=NUMERIC_FEATURES + ['sport_ohe'],
    outputCol='features_raw',
    handleInvalid='skip'
)

#Масштабирование
scaler = StandardScaler(
    inputCol='features_raw',
    outputCol='features_scaled',
    withMean=True, 
    withStd=True
)
```
### 3.3 Random Forest
```python
rf_base = RandomForestClassifier(
    labelCol='label', 
    featuresCol='features_scaled',
    numTrees=50, 
    maxDepth=5, 
    seed=42
)

pipeline_rf = Pipeline(stages=[gender_indexer, sport_indexer, sport_encoder,
                               assembler, scaler, rf_base])
model_rf = pipeline_rf.fit(train_df)
```
**Время обучения:** 1061.9 секунд

### 3.4 Основная модель: Gradient Boosted Trees (GBT)
```python
gbt = GBTClassifier(
    labelCol='label',
    featuresCol='features_scaled',
    maxIter=50,
    maxDepth=5,
    seed=42
)

pipeline_gbt = Pipeline(stages=[gender_indexer, sport_indexer, sport_encoder,
                                assembler, scaler, gbt])
model_gbt = pipeline_gbt.fit(train_df)
```
**Время обучения:** 1588.9 секунды (в 1.5 раза дольше Random Forest)


## 4. Результаты
### 4.1 Сравнение метрик
|Метрика|Random Forest|GBT|Δ|
|-|------|-|-|
|ROC-AUC|0.7516| 0.8540 |+0.1024|
|PR-AUC|0.3244| 0.4580  |+0.1336|
|Accuracy |0.9280| 0.9379 |+0.0099|
|F1-Score |0.8936|0.9185 |+0.0249|

Наблюдения:
* GBT значительно превосходит Random Forest по всем метрикам
* Наибольший прирост — в ROC-AUC (+10.24%) и PR-AUC (+13.36%)
* Accuracy выросла на +0.99%, F1-мера на +2.49%

### 4.2 Матрица ошибок (GBT)
| |Pred Female|Pred Male|
|-|---|-|
|True Female|8984|24|
|True Male|579|122|

Интерпретация:
Модель отлично определяет мужчин (99.4% точность), но хуже определяет женщин (83.6% точность)

### 4.3 Важность признаков
Топ-10 наиболее важных признаков для GBT:

Важность признаков (Top-10):
        feature  importance
     speed_mean    0.220785
       alt_mean    0.220754
      alt_range    0.128575
      speed_max    0.071199
 hr_mean_sliced    0.068343
 workout_length    0.059817
  hr_max_sliced    0.052506
hr_range_sliced    0.042889
  hr_min_sliced    0.025001
    sport_ohe_0    0.019928
Вывод: Наиболее информативными оказались скоростные характеристики и данные о высоте.
