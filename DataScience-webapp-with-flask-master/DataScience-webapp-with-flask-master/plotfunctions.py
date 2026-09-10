# -*- coding: utf-8 -*-
"""
Создано: Четверг, 12 апреля 2018
Автор: Alvaro

Модуль для визуализации результатов машинного обучения и анализа данных.
Все графики конвертируются в формат base64 для удобной интеграции в веб-интерфейсы.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =============================================================================
# 1. ФУНКЦИЯ ДЛЯ ПОСТРОЕНИЯ ROC-КРИВЫХ (ДЛЯ КЛАССИФИКАЦИИ)
# =============================================================================
def plot_ROC(X, y, classifier, cv):
    """
    Строит ROC-кривые для каждого фолда кросс-валидации, а также среднюю ROC-кривую.
    Помогает оценить качество модели бинарной классификации.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp
    
    # Инициализация стратифицированной кросс-валидации для сохранения баланса классов
    cv = StratifiedKFold(n_splits=cv)

    tprs = []       # Список для хранения интерполированных значений True Positive Rate
    aucs = []       # Список для хранения значений площади под кривой (AUC) для каждого фолда
    mean_fpr = np.linspace(0, 1, 100) # Базовая сетка для усреднения False Positive Rate

    i = 0
    # Проход по всем фолдам (итерациям) кросс-валидации
    for train, test in cv.split(X, y):
        # Получение вероятностей принадлежности к целевому классу (столбец 1)
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        
        # Вычисление точек ROC-кривой (FPR и TPR) и порогов
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        
        # Интерполяция кривой для приведения к единой сетке mean_fpr (нужно для последующего усреднения)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0 # Корректировка начальной точки
        
        # Расчет площади под кривой (AUC) для текущего фолда
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Отрисовка ROC-кривой текущего фолда (полупрозрачная линия)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
        
    # Очистка текущего окна графика перед финальной отрисовкой
    plt.gcf().clear()
    
    # Отрисовка базовой линии «случайного угадывания» (диагональ)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # Расчет средней кривой TPR и среднего значения AUC со стандартным отклонением
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 # Корректировка конечной точки
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Отрисовка итоговой средней ROC-кривой
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # Расчет и закраска области стандартного отклонения вокруг средней кривой
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    # Настройка осей, заголовков и легенды
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    
    # Блок конвертации готового графика в формат base64 для отправки
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # Сброс указателя в начало файла
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png
    
# =============================================================================
# 2. ФУНКЦИЯ ДЛЯ СРАВНЕНИЯ ПРЕДСКАЗАНИЙ С РЕАЛЬНОСТЬЮ (ДЛЯ РЕГРЕССИИ)
# =============================================================================
def plot_predVSreal(X, y, classifier, cv):
    """
    Строит диаграмму рассеяния (Scatter Plot) предсказанных значений против реальных.
    Используется для оценки качества моделей регрессии.
    """
    from sklearn.model_selection import cross_val_predict
    
    # Получение предсказаний модели с помощью кросс-валидации
    predicted = cross_val_predict(classifier, X, y, cv=cv)
    plt.gcf().clear()
    
    # Отрисовка точек данных (реальное значение по X, предсказанное — по Y)
    plt.scatter(y, predicted, edgecolors=(0, 0, 0))
    
    # Отрисовка идеальной линии прогноза (где предсказание строго равно реальности)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    
    # Сохранение графика в буфер и кодирование в base64
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

# =============================================================================
# 3. ФУНКЦИЯ СГЛАЖЕННОЙ ГИСТОГРАММЫ (РАСПРЕДЕЛЕНИЕ ПРИЗНАКОВ)
# =============================================================================
def plot_histsmooth(ds, columns):
    """
    Строит графики распределения (гистограммы с ядерной оценкой плотности KDE) 
    для выбранных числовых колонок датасета.
    """
    sns.set() # Применение стилей библиотеки Seaborn
    plt.gcf().clear()
    
    # Поочередное наложение графиков распределения для каждой переданной колонки
    for col in columns:
        sns.distplot(ds[col], label = col)
        
    from io import BytesIO
    plt.xlabel('')
    plt.legend() # Отображение легенды с названиями колонок
    
    # Сохранение в буфер и кодирование в base64
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

# =============================================================================
# 4. ФУНКЦИЯ МАТРИЦЫ КОРРЕЛЯЦИЙ И СВЯЗЕЙ (PAIRPLOT)
# =============================================================================
def plot_correlations(ds, corr, corrcat):
    """
    Строит матрицу парных диаграмм рассеяния (Pair Plot) признаков.
    Помогает визуально найти корреляции между переменными.
    """
    sns.set()
    plt.gcf().clear()
    
    # Если задана категориальная переменная (corrcat), точки окрасятся в разные цвета по группам
    if corrcat != '': 
        sns.pairplot(ds[corr], hue = corrcat)
    else: 
        sns.pairplot(ds[corr])
        
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

# =============================================================================
# 5. ФУНКЦИЯ ДЛЯ ПОСТРОЕНИЯ ЯЩИКА С УСАМИ (BOXPLOT)
# =============================================================================
def plot_boxplot(ds, cat, num):
    """
    Строит "ящик с усами" (Boxplot) для сравнения распределения 
    числового признака (num) по разным категориям (cat).
    """
    sns.set()
    plt.gcf().clear()
    
    # Настройка стиля отображения засечек на осях
    with sns.axes_style(style='ticks'):
        # Построение коробчатой диаграммы (kind="box")
        sns.factorplot(cat, num, data=ds, kind="box")
        
    from io import BytesIO
    plt.xlabel(cat)
    plt.ylabel(num)
    
    # Сохранение в буфер и кодирование в base64
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png
