# validate_mlp.py
import pandas as pd
import os
import joblib
import json
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载验证数据
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 加载模型
model = joblib.load("models/mlp_model.joblib")

# 预测
y_pred = model.predict(X_val)

# 评估指标
metrics = {
    "R2": round(r2_score(y_val, y_pred), 4),
    "MAE": round(mean_absolute_error(y_val, y_pred), 2),
    "MSE": round(mean_squared_error(y_val, y_pred), 2),
    "RMSE": round(np.sqrt(mean_squared_error(y_val, y_pred)), 2)
}

# 保存指标
os.makedirs("models", exist_ok=True)
with open("models/mlp_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ MLP 验证指标：")
for k, v in metrics.items():
    print(f"{k}: {v}")
