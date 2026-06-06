# real_data_experiment.py (остаточна версія)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import shap
import openml
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Завантаження даних
# -------------------------------
print("📥 Loading Corporate Credit Rating dataset...")
dataset = openml.datasets.get_dataset(46525)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format="dataframe"
)
df = X.copy()
df['rating'] = y
print(f"✅ Data loaded. Shape: {df.shape}")

# -------------------------------
# 2. Попередня обробка
# -------------------------------
print("\n🛠️ Preprocessing...")
cols_to_drop = ['Name', 'Symbol', 'Date']
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
if existing_cols_to_drop:
    df = df.drop(columns=existing_cols_to_drop)
    print(f"   Dropped: {existing_cols_to_drop}")

# Заповнення пропусків медіаною
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)

# Кодування категоріальних змінних
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
label_encoders = {}
for col in categorical_columns:
    if col != 'rating':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X = df.drop('rating', axis=1)
y = df['rating']
print(f"   Features: {X.shape[1]}, Classes: {len(np.unique(y))}")

# -------------------------------
# 3. Розділення та навчання моделі
# -------------------------------
print("\n🧠 Training LightGBM...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'num_leaves': [15, 31],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [50, 100],
    'reg_alpha': [0.0, 0.1],
    'reg_lambda': [0.0, 0.1],
}
lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"🏆 Best params: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Test accuracy: {accuracy:.4f}")
print("\n🔍 Classification report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 4. Матриця помилок
# -------------------------------
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -------------------------------
# 5. SHAP аналіз (багатокласовий)
# -------------------------------
print("\n📐 Computing SHAP values...")
X_train_sample = X_train.sample(n=min(200, len(X_train)), random_state=42)  # менше для швидкості
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train_sample)

# Визначаємо структуру shap_values
print(f"   Type of shap_values: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"   Number of elements in list: {len(shap_values)}")
    print(f"   Shape of first element: {shap_values[0].shape}")
    # Для multiclass: список довжиною n_classes, кожен елемент (n_samples, n_features)
    n_classes = len(shap_values)
    n_samples = shap_values[0].shape[0]
    n_features = shap_values[0].shape[1]
    # Об'єднуємо в один масив (n_samples * n_classes, n_features)
    all_shap = np.vstack(shap_values)
    # Повторюємо X для кожного класу
    X_repeated = np.vstack([X_train_sample.values] * n_classes)
else:
    # Якщо shap_values - це масив (n_samples, n_features, n_classes)
    n_samples, n_features, n_classes = shap_values.shape
    all_shap = shap_values.reshape(-1, n_features)
    X_repeated = np.tile(X_train_sample.values, (n_classes, 1))

print(f"   Combined SHAP shape: {all_shap.shape}")

# Глобальна важливість ознак (середнє абсолютне значення)
mean_abs_shap = np.abs(all_shap).mean(axis=0)
shap_feature_importance = pd.DataFrame({'feature': X.columns, 'importance': mean_abs_shap}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=shap_feature_importance.head(15), x='importance', y='feature', palette='viridis')
plt.title('Global SHAP Feature Importance (averaged over classes and samples)')
plt.tight_layout()
plt.show()

# SHAP summary dot plot (об'єднані всі класи)
plt.figure(figsize=(10, 8))
shap.summary_plot(all_shap, X_repeated, feature_names=X.columns, show=False)
plt.title("SHAP Summary Plot (all classes combined)")
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Waterfall plot для першого тестового зразка
# -------------------------------
print("\n💧 Waterfall plot for first test sample...")
shap_single = explainer.shap_values(X_test.iloc[0:1])
predicted_class = best_model.predict(X_test.iloc[0:1])[0]
class_idx = list(best_model.classes_).index(predicted_class)

if isinstance(shap_single, list):
    shap_vals_for_class = shap_single[class_idx][0]  # вектор n_features
    base_value = explainer.expected_value[class_idx]
else:
    # Якщо shap_single - масив (1, n_features, n_classes)
    shap_vals_for_class = shap_single[0, :, class_idx]
    base_value = explainer.expected_value[class_idx]

plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals_for_class,
        base_values=base_value,
        data=X_test.iloc[0].values,
        feature_names=X.columns
    ),
    show=False
)
plt.title(f"Waterfall plot (predicted rating: {predicted_class})")
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Теплова карта кореляції між ознаками на основі SHAP
# -------------------------------
shap_df = pd.DataFrame(all_shap, columns=X.columns)
corr_shap = shap_df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr_shap, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title("Correlation of SHAP values between features")
plt.tight_layout()
plt.show()