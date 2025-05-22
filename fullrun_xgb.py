# fullrun_xgb.py
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载全数据集
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# 模型训练
model = xgb.XGBRegressor(random_state=42, verbosity=0)
model.fit(X, y)

# 模型预测
y_pred = model.predict(X)

# 输出评估指标
print("✅ 全数据训练评估：")
print(f"R²   = {r2_score(y, y_pred):.4f}")
print(f"MAE  = {mean_absolute_error(y, y_pred):.2f}")
print(f"MSE  = {mean_squared_error(y, y_pred):.2f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_pred)):.2f}")

# 可视化：特征重要性图（全数据）
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type="gain", xlabel="Gain")
plt.title("XGBoost Feature Importance (Full Data)")
plt.tight_layout()
plt.savefig("models/xgb_feature_importance_full.png")
plt.show()
