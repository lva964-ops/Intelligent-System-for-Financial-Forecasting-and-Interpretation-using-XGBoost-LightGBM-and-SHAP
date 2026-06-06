# run_experiment_and_plot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Завантаження даних
# -------------------------------
df = pd.read_csv('Data/financial_data.csv')

feature_cols = ['x1_liquidity', 'x2_roa', 'x3_equity_ratio', 'x4_asset_turnover', 'x5_operating_cf']
X = df[feature_cols]
y = df['y']

# Перемішування (для демонстрації методу – за умови, що дані незалежні)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# -------------------------------
# 2. Навчання LightGBM з регуляризацією
# -------------------------------
model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric='l2',
          callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])

# Прогнозування
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

# -------------------------------
# 3. SHAP аналіз
# -------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# -------------------------------
# 4. Побудова графіків
# -------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Графік 1: Фактичні vs Прогноз
plt.figure()
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', c='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted (Test set)\nMSE={test_mse:.3f}, R²={test_r2:.3f}")
plt.tight_layout()
plt.show()

# Графік 2: Гістограма похибок
residuals = y_test - y_test_pred
plt.figure()
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='green')
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Residuals")
plt.axvline(x=0, color='red', linestyle='--')
plt.tight_layout()
plt.show()

# Графік 3: Глобальна важливість за SHAP (bar plot)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, plot_type="bar")
plt.title("Global Feature Importance (SHAP) - Test set")
plt.tight_layout()
plt.show()

# Графік 4: Теплова карта кореляцій SHAP-значень (взаємодії)
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
corr_shap = shap_df.corr()
plt.figure()
sns.heatmap(corr_shap, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title("Correlation of SHAP values between features (interactions)")
plt.tight_layout()
plt.show()

# Графік 5: Waterfall для першого тестового зразка
plt.figure()
shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                     base_values=explainer.expected_value,
                                     data=X_test.iloc[0].values,
                                     feature_names=feature_cols), show=False)
plt.title(f"SHAP Waterfall Plot - First test sample\nPrediction = {y_test_pred[0]:.3f}, Actual = {y_test.iloc[0]:.3f}")
plt.tight_layout()
plt.show()

# Графік 6: SHAP summary dot plot (розподіл впливу ознак)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot (impact distribution)")
plt.tight_layout()
plt.show()