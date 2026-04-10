import medmnist
from medmnist import INFO, Evaluator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
import seaborn as sns

import time
import numpy as np

data_flag = 'tissuemnist'
download = True

info = INFO[data_flag]
DataClass = getattr(medmnist, info["python_class"])

train_dataset = DataClass(split='train', download = download)
test_dataset = DataClass(split='test', download = download)

X = train_dataset.imgs.reshape(len(train_dataset.imgs), -1)
y = train_dataset.labels.flatten()

pca_80_dispersion = PCA(n_components=0.8)
X_pca_80 = pca_80_dispersion.fit_transform(X)

print(f"Количество компонент для 80 процентов: {pca_80_dispersion.components_}")
print(f"Накопленная дисперсия: {pca_80_dispersion.explained_variance_}")

X_norm = X.astype(float)/255.0
X_sample = X_norm[:10000]
y_sample = y[:10000]

methods = {
    "PCA" : PCA(n_components=2),
    "t_SNE" : TSNE(n_components=2, random_state=42),
    "UMAP" : umap.UMAP(n_components=2, random_state=42)
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
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = y_sample, cmap = 'tab10', s = 1, alpha=0.6)
    plt.title(f"{name}\nВремя: {duration:.2f} сек")
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\nРезультаты по времени")
for name, t in result_time.items():
    print(f"\n{name}: {t} сек")