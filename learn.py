import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

def prepare_all_features(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

    X_processed = X[numeric_cols].copy()
    
    for col in categorical_cols:
        if X[col].nunique() <= 10:
            dummies = pd.get_dummies(X[col], prefix=col)
            X_processed = pd.concat([X_processed, dummies], axis=1)
        else:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col])
    
    return X_processed

def fill_missing_values(X):
    X_filled = X.copy()
    
    numeric_cols = X_filled.select_dtypes(include=[np.number]).columns
    X_filled[numeric_cols] = X_filled[numeric_cols].fillna(X_filled[numeric_cols].mean())
    
    binary_cols = []
    for col in X_filled.columns:
        if X_filled[col].nunique() == 2 and set(X_filled[col].dropna().unique()).issubset({0, 1}):
            binary_cols.append(col)
    
    if binary_cols:
        X_filled[binary_cols] = X_filled[binary_cols].fillna(0)
    
    return X_filled

data = fetch_openml(name="house_prices", as_frame=True, parser='auto')
X = data.data
y = data.target

X = prepare_all_features(X)

X = fill_missing_values(X)

missing_values = X.isnull().sum().sum()


X = X.select_dtypes(include=[np.number])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ЛИНЕЙНОЙ РЕГРЕССИИ")
print("="*50)
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Коэффициенты: {model.weights[:5]}...")
print(f"Свободный член: {model.bias:.4f}")

sklearn_model = SklearnLinearRegression()
sklearn_model.fit(X_train_scaled, y_train)

y_test_pred_sklearn = sklearn_model.predict(X_test_scaled)
test_r2_sklearn = r2_score(y_test, y_test_pred_sklearn)
test_mse_sklearn = mean_squared_error(y_test, y_test_pred_sklearn)

print("\n" + "="*50)
print("СРАВНЕНИЕ С SKLEARN")
print("="*50)
print(f"Наша модель R²: {test_r2:.4f}")
print(f"Sklearn R²: {test_r2_sklearn:.4f}")
print(f"Разница: {abs(test_r2 - test_r2_sklearn):.4f}")
print(f"Наша модель MSE: {test_mse:.4f}")
print(f"Sklearn MSE: {test_mse_sklearn:.4f}")
print(f"Разница: {abs(test_mse - test_mse_sklearn):.4f}")