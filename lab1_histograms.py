# lab1_histograms.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Создаём директорию для результатов
os.makedirs('results/lab1', exist_ok=True)

# Параметры распределений
distributions = {
    'Normal': {'func': stats.norm, 'params': (0, 1), 'label': 'N(0, 1)'},
    'Cauchy': {'func': stats.cauchy, 'params': (0, 1), 'label': 'C(0, 1)'},
    'Laplace': {'func': stats.laplace, 'params': (0, 1/np.sqrt(2)), 'label': 'L(0, 1/sqrt(2))'},
    'Poisson': {'func': stats.poisson, 'params': (5,), 'label': 'P(5)', 'discrete': True},
    'Uniform': {'func': stats.uniform, 'params': (-np.sqrt(3), 2*np.sqrt(3)), 'label': 'U(-sqrt(3), sqrt(3))'}
}

sample_sizes = [10, 100, 1000]

# Функция для построения гистограммы с теоретической плотностью
def plot_histogram_with_density(dist_name, dist_info, n, ax):
    np.random.seed(42)
    
    # Генерация выборки
    sample = dist_info['func'].rvs(*dist_info['params'], size=n)
    
    if dist_name == 'Cauchy':
        sample = sample[np.abs(sample) < 10] # обрезать выбросы
    if dist_name == 'Normal':
        ax.set_xlim(-4, 4)  # Показываем только основную часть
    if dist_name == 'Poisson':
        ax.set_xlim(0, 10)  # Показываем только основную часть
    
    ax.hist(sample, bins='auto', density=True, alpha=0.6, color='skyblue', 
            edgecolor='black')
    
    # Теоретическая плотность
    x_min, x_max = min(min(sample), -4), max(max(sample), 4)
    if(dist_info.get('discrete', False)):
        x_vals = np.arange(0, 11) # integers only
        pdf = dist_info['func'].pmf(x_vals, *dist_info['params'])
    else:
        x_vals = np.linspace(x_min, x_max, 1000)
        pdf = dist_info['func'].pdf(x_vals, *dist_info['params'])
    ax.plot(x_vals, pdf, 'r-', linewidth=2)
    
    ax.set_xlabel('Значение')
    ax.set_ylabel('Плотность')
    ax.set_title(f'{dist_info["label"]}, n={n}')
    ax.grid(True, alpha=0.3)

# Построение всех графиков
fig, axes = plt.subplots(len(distributions), len(sample_sizes), figsize=(15, 12))

for i, (dist_name, dist_info) in enumerate(distributions.items()):
    for j, n in enumerate(sample_sizes):
        ax = axes[i, j] if len(distributions) > 1 else axes[j]
        plot_histogram_with_density(dist_name, dist_info, n, ax)

plt.tight_layout()
plt.savefig('results/lab1/all_histograms.png', dpi=300, bbox_inches='tight')
plt.close()

# Сохранение отдельных графиков для отчёта
for dist_name, dist_info in distributions.items():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j, n in enumerate(sample_sizes):
        plot_histogram_with_density(dist_name, dist_info, n, axes[j])
    plt.tight_layout()
    plt.savefig(f'results/lab1/{dist_name}_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Lab 1 completed! Graphs saved to results/lab1/")