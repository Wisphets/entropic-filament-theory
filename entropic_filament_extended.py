#!/usr/bin/env python3
"""
entropic_filament_extended.py

Проверка Теории Энтропийных Нитей с повышенной статистикой и масштабом:
- NODES = 100
- P_EDGE = 0.05
- MASS = 5.0
- M_RUNS = 150
- α = 0.01 (цель p<0.01)

Сохраняет:
 - C:\PY\niti\entropic_corrs_extended.csv
 - C:\PY\niti\entropic_hist_extended.png

Запуск:
    pip install numpy scipy pandas networkx matplotlib
    python entropic_filament_extended.py
"""
import os, math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import t

# === Параметры ===
OUTPUT_DIR = r"C:\PY\niti"
CSV_FILE   = "entropic_corrs_extended.csv"
PNG_FILE   = "entropic_hist_extended.png"

NODES   = 100      # число узлов в графе
P_EDGE  = 0.05     # вероятность ребра
M_RUNS  = 150      # число успешных прогонов
MASS    = 5.0      # коэффициент «массы» (усиление весов)
ALPHA   = 0.01     # порог для p-value

# Создать выходную папку, если нужно
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_once(seed):
    """Один прогон: строит взвешенный граф, решает Laplace для E, возвращает r."""
    np.random.seed(seed)
    # 1) Случайный граф Эрдеёша-Реньи
    G = nx.erdos_renyi_graph(NODES, P_EDGE)
    if not nx.is_connected(G):
        # если граф не связен — пропустить
        raise ValueError("Graph not connected")
    # 2) Вес 1.0 по умолчанию
    for u, v in G.edges():
        G[u][v]['w'] = 1.0
    # 3) Усилить веса у центрального узла 0
    for nbr in G.neighbors(0):
        G[0][nbr]['w'] *= MASS
    # 4) Взвешенный лапласиан
    L = nx.laplacian_matrix(G, weight='w').astype(float).toarray()
    # 5) Dirichlet: E(0)=1
    b = np.zeros(NODES); b[0] = 1.0
    A = np.delete(np.delete(L, 0, axis=0), 0, axis=1)
    bb = np.delete(b, 0)
    sol = np.linalg.solve(A, bb)
    E = np.zeros(NODES)
    idx = list(range(NODES)); idx.pop(0)
    E[idx] = sol; E[0] = 1.0
    # 6) Расстояния по графу
    dist = np.array([nx.shortest_path_length(G, 0, i) for i in range(NODES)])
    dE   = 1.0 - E
    # 7) Корреляция
    return np.corrcoef(dist, dE)[0,1]

# --- Собираем выборку r ---
corrs = []
seed = 0
while len(corrs) < M_RUNS:
    try:
        r = run_once(seed)
        corrs.append(r)
    except Exception:
        pass
    seed += 1

df = pd.DataFrame({'r': corrs})
# Сохранить CSV
csv_path = os.path.join(OUTPUT_DIR, CSV_FILE)
df.to_csv(csv_path, index=False)

# Нарисовать гистограмму
plt.figure(figsize=(6,4))
plt.hist(df['r'], bins=15, edgecolor='black')
plt.xlabel('Correlation dist vs ΔE')
plt.ylabel('Frequency')
plt.title(f'Entropic Filament (N={NODES}, runs={M_RUNS})')
plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, PNG_FILE)
plt.savefig(png_path)
plt.close()

# --- Статистика ---
mean_r = df['r'].mean()
n      = len(corrs)
t_stat = mean_r * math.sqrt((n-1)/(1-mean_r**2))
p_val  = 2 * (1 - t.cdf(abs(t_stat), df=n-1))

print(f"Results saved to:\n  {csv_path}\n  {png_path}")
print(f"Mean r    = {mean_r:.4f}")
print(f"t-stat    = {t_stat:.3f}")
print(f"p-value   = {p_val:.5f}")
if p_val < ALPHA:
    print("=> Теория Энтропийных Нитей подтверждена на уровне >99% (p<0.01).")
else:
    print("=> Теория НЕ подтвердилась на уровне 99% (p>=0.01).")
