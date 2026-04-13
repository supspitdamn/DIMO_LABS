import medmnist
from medmnist import INFO

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
import time

data_flag = 'tissuemnist'
download = True

info = INFO[data_flag] 
DataClass = getattr(medmnist, info["python_class"]) # Загружаю датасет

train_dataset = DataClass(split='train', download = download)
test_dataset = DataClass(split='test', download = download)

X = train_dataset.imgs.reshape(len(train_dataset.imgs), -1) # Двумерная матрица
y = train_dataset.labels.flatten()# делаю вектор строку

X_norm = X.astype(float)/255.0 # нормировка (изображение в градациях серого, поэтому делю на 255.0)

pca_80_dispersion = PCA(n_components=0.8) # n_components < 1 - автоподбор числа компонент для удовлетворения дисперсии, в противном случае - число компонент
X_pca_80 = pca_80_dispersion.fit_transform(X_norm)

print(f"Количество компонент для 80 процентов: {pca_80_dispersion.n_components_}") # поле n_comp pca_80 хранит информацию о количестве компонент
print(f"Накопленная дисперсия: {pca_80_dispersion.explained_variance_ratio_.sum()}") # Суммирует вклад каждой оси в объяснение дисперсии (должно быть 80%)

X_sample = X_norm[:1000] # Беру первые 1000 изображений
y_sample = y[:1000] # Беру ответы для каждого из 1000 изображений

"""
ИСПОЛЬЗУЮ n_components = 2 для того чтобы показать на плосоксти
"""

methods = {
            "PCA" : PCA(n_components=2),
            "t_SNE" : TSNE(n_components=2, random_state=42, perplexity=30), # перплексити - параметр для гауссова распределения (знаменатель экспоненты)
            "UMAP" : umap.UMAP(n_components=2, random_state=42, n_neighbors=5) # неигхборс - числол соседей для построения графа связей
          }

plt.figure(figsize=(20, 6))
result_time = {}

for i, (name, method) in enumerate(methods.items()):

    print(f"Старт {name}")
    start = time.time()

    X_embedded = method.fit_transform(X_sample)

    duration = time.time() - start
    result_time[name] = duration

    plt.subplot(1, 3, i+1)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = y_sample, cmap = 'tab10', s = 10, alpha=0.6)
    plt.title(f"{name}\nВремя: {duration} сек")

    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nРезультаты по времени")
for name, t in result_time.items():
    print(f"\n{name}: {t} сек")