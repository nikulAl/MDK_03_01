# -*- coding: utf-8 -*-
"""
Создано: Вторник, 10 апреля 2018
Автор: Alvaro

Главный модуль веб-приложения на Flask. 
Управляет загрузкой данных, вызовом методов предобработки, обучения моделей и визуализации.
"""

from flask import Flask, render_template, request, redirect, make_response, send_file
import os
import pandas as pd
import numpy as np
import models as algorithms  # Импорт пользовательского модуля со словарями моделей
import plotfunctions as plotfun  # Импорт пользовательского модуля построения графиков
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from flask_bootstrap import Bootstrap

# Инициализация Flask-приложения и интеграция стилей Bootstrap
app = Flask(__name__)
bootstrap = Bootstrap(app)

# =============================================================================
# ВСПОРМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛАМИ ДАННЫХ
# =============================================================================

def datasetList():
    """
    Сканирует папки 'datasets' (исходные) и 'preprocessed' (обработанные),
    возвращая списки имен файлов, их расширений и соответствующих папок.
    """
    datasets = [x.split('.')[0] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    extensions = [x.split('.')[1] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    folders = [f for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    return datasets, extensions, folders

def loadColumns(dataset):
    """
    Считывает только заголовки (колонки) датасета без загрузки всего файла в память (nrows=0).
    Поддерживает форматы .txt и .csv.
    """
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'), nrows=0)
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'), nrows=0)
        return df.columns

def loadDataset(dataset):
    """
    Полностью загружает выбранный датасет в виде объекта pandas DataFrame.
    """
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
        return df


# =============================================================================
# МАРШРУТЫ (ЭНДПОИНТЫ) ВЕБ-ПРИЛОЖЕНИЯ
# =============================================================================

@app.route('/', methods = ['GET', 'POST'])
def index():
    """
    Главная страница приложения. 
    GET: Отображает списки доступных исходных и обработанных датасетов.
    POST: Принимает и сохраняет новый файл датасета от пользователя.
    """
    datasets,_,folders = datasetList()
    originalds = []
    featuresds = []
    for i in range(len(datasets)):
        if folders[i] == 'datasets': originalds += [datasets[i]]
        else: featuresds += [datasets[i]]
    if request.method == 'POST':
            f = request.files['file']
            f.save(os.path.join('datasets', f.filename))
            return redirect('/')
    return render_template('index.html', originalds = originalds, featuresds = featuresds)

@app.route('/datasets/')
def datasets():
    """Перенаправление с пустого адреса датасетов на главную."""
    return redirect('/')

@app.route('/datasets/<dataset>')
def dataset(description = None, head = None, dataset = None):
    """
    Страница просмотра датасета.
    Генерирует HTML-таблицы с первыми 5 строками и базовым статистическим описанием (describe).
    """
    df = loadDataset(dataset)
    try:
        description = df.describe().round(2)
        head = df.head(5)
    except: pass
    return render_template('dataset.html',
                           description = description.to_html(classes='table table-striped table-hover'),
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           dataset = dataset)

@app.route('/datasets/<dataset>/models')
def models(dataset = dataset):
    """
    Страница выбора параметров обучения.
    Передает в HTML форму списки колонок и доступных алгоритмов классификации/регрессии.
    """
    columns = loadColumns(dataset)
    clfmodels = algorithms.classificationModels()
    predmodels = algorithms.regressionModels()
    return render_template('models.html', dataset = dataset,
                           clfmodels = clfmodels,
                           predmodels = predmodels,
                           columns = columns)

@app.route('/datasets/<dataset>/modelprocess/', methods=['POST'])
def model_process(dataset = dataset):
    """
    Основной обработчик обучения моделей.
    Считывает настройки из веб-формы (выбор фичей, масштабирование, K-Fold),
    запускает кросс-валидацию и возвращает метрики вместе с построенным графиком оценки (ROC или Pred vs Real).
    """
    algscore = request.form.get('model')
    res = request.form.get('response') # Целевая переменная (Y)
    kfold = request.form.get('kfold')
    alg, score = algscore.split('.') # Разделение имени алгоритма и типа задачи
    scaling = request.form.get('scaling')
    variables = request.form.getlist('variables') # Выбранные независимые признаки (X)
    
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    df = loadDataset(dataset)
    y = df[str(res)]

    # Формирование матрицы признаков X
    if variables != [] and '' not in variables: df = df[list(set(variables + [res]))]
    X = df.drop(str(res), axis=1)
    try: X = pd.get_dummies(X) # One-Hot кодирование категориальных признаков
    except: pass
    
    predictors = X.columns
    if len(predictors)>10: pred = str(len(predictors))
    else: pred = ', '.join(predictors)    

    # Блок для задачи КЛАССИФИКАЦИИ
    if score == 'Classification':
        scoring = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']
        if scaling == 'Yes':
            clf = algorithms.classificationModels()[alg]
            mod = Pipeline([('scaler', StandardScaler()), ('clf', clf)]) # Пайплайн со стандартизацией данных
        else: 
            mod = algorithms.classificationModels()[alg]
        fig = plotfun.plot_ROC(X.values, y, mod, int(kfold)) # Генерация ROC-кривой в base64

    # Блок для задачи РЕГРЕССИИ
    elif score == 'Regression':
        scoring = ['explained_variance', 'r2', 'mean_squared_error']
        if scaling == 'Yes':
            pr = algorithms.regressionModels()[alg]  
            mod = Pipeline([('scaler', StandardScaler()), ('clf', pr)])
        else: mod = algorithms.regressionModels()[alg]
        fig = plotfun.plot_predVSreal(X, y, mod, int(kfold)) # Генерация графика реальных/предсказанных значений
    
    # Расчет метрик через кросс-валидацию scikit-learn
    scores = cross_validate(mod, X, y, cv=int(kfold), scoring=scoring)
    for s in scores:
        scores[s] = str(round(np.mean(scores[s]),3)) # Усреднение результатов по всем фолдам
        
    return render_template('scores.html', scores = scores, dataset = dataset, alg=alg,
                           res = res, kfold = kfold, score = score,
                           predictors = pred, response = str(fig, 'utf-8'))
    
@app.route('/datasets/<dataset>/preprocessing')
def preprocessing(dataset = dataset):
    """Страница выбора параметров предобработки (отбор признаков, удаление пропусков)."""
    columns = loadColumns(dataset)
    return render_template('preprocessing.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/preprocessed_dataset/', methods=['POST'])
def preprocessed_dataset(dataset):
    """
    Обработчик предобработки данных.
    Выполняет очистку от NaN, удаление константных колонок, автоматический отбор K лучших признаков 
    методом Хи-квадрат (SelectKBest) или ручной выбор признаков, сохраняя новый файл в 'preprocessed/'.
    """
    numFeatures = request.form.get('nfeatures')
    manualFeatures = request.form.getlist('manualfeatures')
    datasetName = request.form.get('newdataset')
    response = request.form.get('response')
    dropsame = request.form.get('dropsame')
    dropna = request.form.get('dropna')
    
    df = loadDataset(dataset)

    # Очистка пустых значений (строк или колонок)
    if dropna == 'all':
        df = df.dropna(axis=1, how='all')
    elif dropna == 'any':
        df.dropna(axis=1, how='any')
        
    filename = dataset + '_'
    try:
        # Автоматический отбор K лучших признаков через статистический критерий Хи-квадрат
        nf = int(numFeatures)
        from sklearn.feature_selection import SelectKBest, chi2
        X = df.drop(str(response), axis=1)
        y = df[str(response)]
        kbest = SelectKBest(chi2, k=nf).fit(X,y)
        mask = kbest.get_support()
        
        best_features = []
        for bool, feature in zip(mask, list(df.columns)):
            if bool: best_features.append(feature)
            
        df = pd.DataFrame(kbest.transform(X), columns=best_features)
        df.insert(0, str(response), y)
        filename += numFeatures + '_' + 'NA' + dropna + '_Same' + dropsame + '.csv'
    
    except:
        # Если автоматический выбор не удался — берется список колонок, указанный вручную
        df = df[manualFeatures]
        filename += str(datasetName) + '_' + str(response) + '.csv'
    
    # Удаление колонок, где все значения одинаковы (нулевая дисперсия)
    if dropsame == 'Yes':
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
