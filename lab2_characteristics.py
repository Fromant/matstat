# lab2_characteristics.py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

os.makedirs('results/lab2', exist_ok=True)

# Параметры распределений
distributions = {
    'Normal': {'func': stats.norm, 'params': (0, 1), 'label': 'N(0, 1)', 'theoretical_mean': 0},
    'Cauchy': {'func': stats.cauchy, 'params': (0, 1), 'label': 'C(0, 1)', 'theoretical_mean': None},
    'Laplace': {'func': stats.laplace, 'params': (0, 1/np.sqrt(2)), 'label': 'L(0, 1/√2)', 'theoretical_mean': 0},
    'Poisson': {'func': stats.poisson, 'params': (5,), 'label': 'P(10)', 'theoretical_mean': 10},
    'Uniform': {'func': stats.uniform, 'params': (-np.sqrt(3), 2*np.sqrt(3)), 'label': 'U(-√3, √3)', 'theoretical_mean': 0}
}

sample_sizes = [10, 100, 1000]
n_iterations = 1000

# Функции для расчёта характеристик
def calculate_mean(sample):
    return np.mean(sample)

def calculate_median(sample):
    return np.median(sample)

def calculate_mid_range(sample):
    return (np.min(sample) + np.max(sample)) / 2

def calculate_mid_quartile(sample):
    q1, q3 = np.percentile(sample, [25, 75])
    return (q1 + q3) / 2

def calculate_trimmed_mean(sample, proportion=0.1):
    return stats.trim_mean(sample, proportion)

characteristics = {
    'mean': calculate_mean,
    'median': calculate_median,
    'mid_range': calculate_mid_range,
    'mid_quartile': calculate_mid_quartile,
    'trimmed_mean': calculate_trimmed_mean
}

# Результаты
results = {}

for dist_name, dist_info in distributions.items():
    results[dist_name] = {}
    
    for n in sample_sizes:
        results[dist_name][n] = {}
        
        for char_name, char_func in characteristics.items():
            values = []
            
            for _ in range(n_iterations):
                if dist_name == 'Poisson':
                    sample = dist_info['func'].rvs(*dist_info['params'], size=n)
                else:
                    sample = dist_info['func'].rvs(*dist_info['params'], size=n)
                
                # Для Коши пропускаем выбросы при расчёте среднего
                if dist_name == 'Cauchy' and char_name == 'mean':
                    sample = sample[np.abs(sample) < 10]  # Ограничиваем выбросы
                
                if len(sample) > 0:
                    values.append(char_func(sample))
            
            if len(values) > 0:
                E_z = np.mean(values)
                D_z = np.var(values)
                std_z = np.std(values)
                
                results[dist_name][n][char_name] = {
                    'E': E_z,
                    'D': D_z,
                    'std': std_z
                }

# Вывод результатов в консоль
print("=" * 80)
print("LAB 2: ХАРАКТЕРИСТИКИ ПОЛОЖЕНИЯ И РАССЕЯНИЯ")
print("=" * 80)

for dist_name, dist_info in distributions.items():
    print(f"\n{dist_name} ({dist_info['label']}):")
    print(f"Теоретическое среднее: {dist_info['theoretical_mean']}")
    print("-" * 80)
    
    for n in sample_sizes:
        print(f"\n  n = {n}:")
        for char_name, char_data in results[dist_name][n].items():
            print(f"    {char_name:15s}: E = {char_data['E']:8.4f}, D = {char_data['D']:8.4f}, std = {char_data['std']:8.4f}")

# Сохранение результатов в файл
with open('results/lab2/results.txt', 'w', encoding='utf-8') as f:
    f.write("LAB 2: ХАРАКТЕРИСТИКИ ПОЛОЖЕНИЯ И РАССЕЯНИЯ\n")
    f.write("=" * 80 + "\n\n")
    
    for dist_name, dist_info in distributions.items():
        f.write(f"{dist_name} ({dist_info['label']}):\n")
        f.write(f"Теоретическое среднее: {dist_info['theoretical_mean']}\n")
        f.write("-" * 80 + "\n")
        
        for n in sample_sizes:
            f.write(f"\nn = {n}:\n")
            for char_name, char_data in results[dist_name][n].items():
                f.write(f"  {char_name:15s}: E = {char_data['E']:8.4f}, D = {char_data['D']:8.4f}, std = {char_data['std']:8.4f}\n")
        f.write("\n")

# Построение графиков сходимости
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# График 1: Сходимость среднего к теоретическому значению
ax = axes[0, 0]
for dist_name, dist_info in distributions.items():
    if dist_info['theoretical_mean'] is not None:
        errors = [abs(results[dist_name][n]['mean']['E'] - dist_info['theoretical_mean']) 
                 for n in sample_sizes]
        ax.plot(sample_sizes, errors, 'o-', label=dist_info['label'])

ax.set_xlabel('Размер выборки (n)')
ax.set_ylabel('Абсолютная ошибка среднего')
ax.set_title('Сходимость выборочного среднего')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# График 2: Дисперсия характеристик
ax = axes[0, 1]
char_colors = {'mean': 'blue', 'median': 'green', 'mid_range': 'red', 
               'mid_quartile': 'orange', 'trimmed_mean': 'purple'}

for char_name, color in char_colors.items():
    variances = [results['Normal'][n][char_name]['D'] for n in sample_sizes]
    ax.plot(sample_sizes, variances, 'o-', color=color, label=char_name)

ax.set_xlabel('Размер выборки (n)')
ax.set_ylabel('Дисперсия оценки')
ax.set_title('Дисперсия характеристик (Normal)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# График 3: Сравнение робастности (Cauchy)
ax = axes[0, 2]
for char_name, color in char_colors.items():
    if char_name in results['Cauchy'][100]:
        stds = [results['Cauchy'][n][char_name]['std'] for n in sample_sizes]
        ax.plot(sample_sizes, stds, 'o-', color=color, label=char_name)

ax.set_xlabel('Размер выборки (n)')
ax.set_ylabel('Стандартное отклонение')
ax.set_title('Робастность характеристик (Cauchy)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# График 4-6: Боксплоты характеристик для разных n
for idx, n in enumerate(sample_sizes):
    ax = axes[1, idx]
    data_to_plot = []
    labels = []
    
    for char_name in characteristics.keys():
        if char_name in results['Normal'][n]:
            values = []
            for _ in range(100):
                sample = stats.norm.rvs(0, 1, size=n)
                values.append(characteristics[char_name](sample))
            data_to_plot.append(values)
            labels.append(char_name)
    
    ax.boxplot(data_to_plot, labels=labels)
    ax.set_title(f'Распределение характеристик (n={n})')
    ax.set_ylabel('Значение')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('results/lab2/convergence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nLab 2 completed! Results saved to results/lab2/")