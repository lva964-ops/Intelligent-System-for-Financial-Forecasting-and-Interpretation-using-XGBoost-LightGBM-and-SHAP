# generate_initial_dataset.py (оновлений)
import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SAMPLES = 1000        # кількість спостережень

# Генеруємо 5 незалежних фінансових індикаторів (нормальний розподіл)
x1 = np.random.normal(0, 1, N_SAMPLES)   # ліквідність
x2 = np.random.normal(0, 1, N_SAMPLES)   # рентабельність
x3 = np.random.normal(0, 1, N_SAMPLES)   # фінансова стійкість
x4 = np.random.normal(0, 1, N_SAMPLES)   # ділова активність
x5 = np.random.normal(0, 1, N_SAMPLES)   # грошовий потік

# Цільова змінна (прогнозований показник) – нелінійна функція + взаємодії + невеликий шум
y = (0.3 * x1
     + 0.5 * np.tanh(x2)          # нелінійність
     + 0.2 * (x3 ** 2)            # квадратичний ефект
     + 0.15 * x4
     + 0.1 * x5
     + 0.1 * x1 * x2              # взаємодія
     + np.random.normal(0, 0.1, N_SAMPLES))   # шум (σ=0.1)

# Формуємо DataFrame
data = pd.DataFrame({
    'x1_liquidity': x1,
    'x2_roa': x2,
    'x3_equity_ratio': x3,
    'x4_asset_turnover': x4,
    'x5_operating_cf': x5,
    'y': y
})

# Збереження
os.makedirs('Data', exist_ok=True)
data.to_csv('Data/financial_data.csv', index=False)

print("✅ Новий датасет згенеровано: Data/financial_data.csv")
print(f"Розмір: {data.shape}")
print(data.describe())