# lab4_edf_kernel.py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.neighbors import KernelDensity

os.makedirs('results/lab4', exist_ok=True)

# Параметры распределений (только непрерывные + Пуассон)
distributions = {
    'Normal': {'func': stats.norm, 'params': (0, 1), 'label': 'N(0, 1)', 'range': [-4, 4]},
    'Cauchy': {'func': stats.cauchy, 'params': (0, 1), 'label': 'C(0, 1)', 'range': [-4, 4]},
    'Laplace': {'func': stats.laplace, 'params': (0, 1/np.sqrt(2)), 'label': 'L(0, 1/√2)', 'range': [-4, 4]},
    'Poisson': {'func': stats.poisson, 'params': (10,), 'label': 'P(10)', 'range': [6, 14], 'discrete': True},
    'Uniform': {'func': stats.uniform, 'params': (-np.sqrt(3), 2*np.sqrt(3)), 'label': 'U(-√3, √3)', 'range': [-4, 4]}
}

sample_sizes = [20, 60, 100]

# Функция для расчёта эмпирической функции распределения
def empirical_cdf(sample, x_values):
    n = len(sample)
    return np.array([np.sum(sample <= x) / n for x in x_values])

# Функция для ядерной оценки плотности
def kernel_density_estimate(sample, x_values, bandwidth=0.5):
    sample = sample.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sample)
    log_density = kde.score_samples(x_values.reshape(-1, 1))
    return np.exp(log_density)

# Построение графиков
for dist_name, dist_info in distributions.items():
    x_range = dist_info['range']
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    
    # Теоретические функции
    if dist_info.get('discrete', False):
        theoretical_pmf = dist_info['func'].pmf(x_values.astype(int), *dist_info['params'])
        theoretical_cdf = dist_info['func'].cdf(x_values.astype(int), *dist_info['params'])
    else:
        theoretical_pdf = dist_info['func'].pdf(x_values, *dist_info['params'])
        theoretical_cdf = dist_info['func'].cdf(x_values, *dist_info['params'])
    
    # Создание фигуры для этого распределения
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, n in enumerate(sample_sizes):
        # Генерация выборки
        np.random.seed(42 + n)
        if dist_info.get('discrete', False):
            sample = dist_info['func'].rvs(*dist_info['params'], size=n)
        else:
            sample = dist_info['func'].rvs(*dist_info['params'], size=n)
        
        # Эмпирическая CDF
        emp_cdf = empirical_cdf(sample, x_values)
        
        # Ядерная оценка плотности
        if not dist_info.get('discrete', False):
            kde_density = kernel_density_estimate(sample, x_values, bandwidth=0.5)
        
        # График 1: EDF vs теоретическая CDF (верхний ряд)
        ax = axes[0, idx]
        ax.plot(x_values, theoretical_cdf, 'r-', linewidth=2, label='Теоретическая CDF')
        ax.step(x_values, emp_cdf, 'b-', where='post', linewidth=1.5, label='Эмпирическая CDF', alpha=0.7)
        ax.set_title(f'Функция распределения, n={n}')
        ax.set_xlabel('Значение')
        ax.set_ylabel('F(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # График 2: Ядерная оценка vs теоретическая плотность (нижний ряд)
        ax = axes[1, idx]
        if not dist_info.get('discrete', False):
            ax.plot(x_values, theoretical_pdf, 'r-', linewidth=2, label='Теоретическая плотность')
            ax.plot(x_values, kde_density, 'b-', linewidth=1.5, label='Ядерная оценка', alpha=0.7)
            ax.hist(sample, bins='auto', density=True, alpha=0.3, color='gray', label='Гистограмма')
        else:
            # Для Пуассона
            unique_vals = np.unique(sample)
            hist_vals, hist_bins = np.histogram(sample, bins=np.arange(min(sample)-0.5, max(sample)+1.5))
            bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
            ax.bar(bin_centers, hist_vals / n, width=0.8, alpha=0.5, label='Гистограмма')
            ax.vlines(x_values.astype(int), 0, theoretical_pmf, colors='red', 
                     linestyles='dashed', linewidth=2, label='Теоретическая')
        
        ax.set_title(f'Оценка плотности, n={n}')
        ax.set_xlabel('Значение')
        ax.set_ylabel('Плотность')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/lab4/{dist_name}_edf_kde.png', dpi=300, bbox_inches='tight')
    plt.close()

# Сводный график: сравнение гистограммы и ядерной оценки
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

for idx, n in enumerate(sample_sizes):
    np.random.seed(42)
    sample = stats.norm.rvs(0, 1, size=n)
    x_values = np.linspace(-4, 4, 1000)
    
    # Гистограмма
    ax = axes[0, idx]
    ax.hist(sample, bins='auto', density=True, alpha=0.6, color='skyblue', 
            edgecolor='black', label='Гистограмма')
    ax.plot(x_values, stats.norm.pdf(x_values), 'r-', linewidth=2, label='Теоретическая')
    ax.set_title(f'Гистограмма, n={n}')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Плотность')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ядерная оценка
    ax = axes[1, idx]
    kde_density = kernel_density_estimate(sample, x_values, bandwidth=0.5)
    ax.plot(x_values, stats.norm.pdf(x_values), 'r-', linewidth=2, label='Теоретическая')
    ax.plot(x_values, kde_density, 'b-', linewidth=2, label='Ядерная оценка')
    ax.set_title(f'Ядерная оценка, n={n}')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Плотность')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/lab4/histogram_vs_kde_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Расчёт ошибки приближения EDF к теоретической CDF
print("=" * 80)
print("LAB 4: ЭМПИРИЧЕСКАЯ ФУНКЦИЯ РАСПРЕДЕЛЕНИЯ И ЯДЕРНЫЕ ОЦЕНКИ")
print("=" * 80)

edf_errors = {}

for dist_name, dist_info in distributions.items():
    if dist_info.get('discrete', False):
        continue
    
    edf_errors[dist_name] = {}
    x_range = dist_info['range']
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    theoretical_cdf = dist_info['func'].cdf(x_values, *dist_info['params'])
    
    for n in sample_sizes:
        max_errors = []
        
        for _ in range(100):
            sample = dist_info['func'].rvs(*dist_info['params'], size=n)
            emp_cdf = empirical_cdf(sample, x_values)
            max_error = np.max(np.abs(emp_cdf - theoretical_cdf))
            max_errors.append(max_error)
        
        edf_errors[dist_name][n] = {
            'mean_max_error': np.mean(max_errors),
            'std_max_error': np.std(max_errors)
        }
    
    print(f"\n{dist_name} ({dist_info['label']}):")
    for n in sample_sizes:
        data = edf_errors[dist_name][n]
        print(f"  n={n}: Максимальная ошибка EDF = {data['mean_max_error']:.4f} ± {data['std_max_error']:.4f}")

# Сохранение результатов
with open('results/lab4/edf_errors.txt', 'w', encoding='utf-8') as f:
    f.write("LAB 4: ОШИБКИ ПРИБЛИЖЕНИЯ EDF\n")
    f.write("=" * 80 + "\n\n")
    
    for dist_name, dist_info in distributions.items():
        if dist_name in edf_errors:
            f.write(f"{dist_name} ({dist_info['label']}):\n")
            for n in sample_sizes:
                data = edf_errors[dist_name][n]
                f.write(f"  n={n}: Максимальная ошибка EDF = {data['mean_max_error']:.4f} ± {data['std_max_error']:.4f}\n")
            f.write("\n")

print("\nLab 4 completed! Results saved to results/lab4/")