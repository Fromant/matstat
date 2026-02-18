# lab3_boxplot.py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

os.makedirs('results/lab3', exist_ok=True)

# Параметры распределений
distributions = {
    'Normal': {'func': stats.norm, 'params': (0, 1), 'label': 'N(0, 1)'},
    'Cauchy': {'func': stats.cauchy, 'params': (0, 1), 'label': 'C(0, 1)'},
    'Laplace': {'func': stats.laplace, 'params': (0, 1/np.sqrt(2)), 'label': 'L(0, 1/√2)'},
    'Poisson': {'func': stats.poisson, 'params': (10,), 'label': 'P(10)', 'discrete': True},
    'Uniform': {'func': stats.uniform, 'params': (-np.sqrt(3), 2*np.sqrt(3)), 'label': 'U(-√3, √3)'}
}

sample_sizes = [20, 100]
n_iterations = 1000

# Функция для определения выбросов по методу Тьюки
def find_outliers_tukey(sample):
    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
    return len(outliers), len(outliers) / len(sample)

# Результаты
outlier_results = {}

for dist_name, dist_info in distributions.items():
    outlier_results[dist_name] = {}
    
    for n in sample_sizes:
        outlier_proportions = []
        total_outliers = 0
        
        for _ in range(n_iterations):
            if dist_name == 'Poisson':
                sample = dist_info['func'].rvs(*dist_info['params'], size=n)
            else:
                sample = dist_info['func'].rvs(*dist_info['params'], size=n)
            
            n_outliers, proportion = find_outliers_tukey(sample)
            outlier_proportions.append(proportion)
            total_outliers += n_outliers
        
        outlier_results[dist_name][n] = {
            'mean_proportion': np.mean(outlier_proportions),
            'std_proportion': np.std(outlier_proportions),
            'total_outliers': total_outliers,
            'total_samples': n_iterations * n
        }

# Вывод результатов
print("=" * 80)
print("LAB 3: БОКСПЛОТ ТЬЮКИ И АНАЛИЗ ВЫБРОСОВ")
print("=" * 80)

for dist_name, dist_info in distributions.items():
    print(f"\n{dist_name} ({dist_info['label']}):")
    print("-" * 80)
    
    for n in sample_sizes:
        data = outlier_results[dist_name][n]
        print(f"  n = {n}:")
        print(f"    Средняя доля выбросов: {data['mean_proportion']:.4f} ± {data['std_proportion']:.4f}")
        print(f"    Всего выбросов: {data['total_outliers']} из {data['total_samples']}")

# Сохранение результатов
with open('results/lab3/outlier_results.txt', 'w', encoding='utf-8') as f:
    f.write("LAB 3: БОКСПЛОТ ТЬЮКИ И АНАЛИЗ ВЫБРОСОВ\n")
    f.write("=" * 80 + "\n\n")
    
    for dist_name, dist_info in distributions.items():
        f.write(f"{dist_name} ({dist_info['label']}):\n")
        f.write("-" * 80 + "\n")
        
        for n in sample_sizes:
            data = outlier_results[dist_name][n]
            f.write(f"  n = {n}:\n")
            f.write(f"    Средняя доля выбросов: {data['mean_proportion']:.4f} ± {data['std_proportion']:.4f}\n")
            f.write(f"    Всего выбросов: {data['total_outliers']} из {data['total_samples']}\n")
        f.write("\n")

# Построение боксплотов
fig, axes = plt.subplots(len(distributions), len(sample_sizes), figsize=(15, 10))

for i, (dist_name, dist_info) in enumerate(distributions.items()):
    for j, n in enumerate(sample_sizes):
        ax = axes[i, j] if len(distributions) > 1 else axes[j]
        
        # Генерация выборки для боксплота
        np.random.seed(42)
        if dist_name == 'Poisson':
            sample = dist_info['func'].rvs(*dist_info['params'], size=n)
        else:
            sample = dist_info['func'].rvs(*dist_info['params'], size=n)
        
        # Построение боксплота
        bp = ax.boxplot(sample, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='blue'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='green'),
                       capprops=dict(color='green'),
                       flierprops=dict(marker='o', markerfacecolor='orange', 
                                      markersize=6, linestyle='none'))
        
        ax.set_title(f'{dist_info["label"]}, n={n}')
        ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Подпись доли выбросов
        outlier_data = outlier_results[dist_name][n]
        ax.text(0.5, -0.1, f'Выбросы: {outlier_data["mean_proportion"]:.2%}',
               transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/lab3/boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# График зависимости доли выбросов от размера выборки
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['blue', 'red', 'green', 'orange', 'purple']
for idx, (dist_name, dist_info) in enumerate(distributions.items()):
    proportions = [outlier_results[dist_name][n]['mean_proportion'] for n in sample_sizes]
    ax.plot(sample_sizes, proportions, 'o-', color=colors[idx], 
            linewidth=2, markersize=8, label=dist_info['label'])

ax.set_xlabel('Размер выборки (n)', fontsize=12)
ax.set_ylabel('Средняя доля выбросов', fontsize=12)
ax.set_title('Зависимость доли выбросов от размера выборки', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('results/lab3/outlier_proportion.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nLab 3 completed! Results saved to results/lab3/")