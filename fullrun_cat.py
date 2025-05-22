# fullrun_cat.py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载全量数据
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# 模型训练
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 指标输出
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("✅ CatBoost 全数据评估结果：")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")

# 📊 特征重要性
importances = model.get_feature_importance(prettified=True)
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature Id'], importances['Importances'])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance (Full Data)")
plt.tight_layout()
plt.savefig("models/cat_feature_importance_full.png")
plt.show()
