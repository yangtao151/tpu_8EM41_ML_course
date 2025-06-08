import pandas as pd
import os
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========== 1. 加载验证数据 ==========
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# ========== 2. 加载 scaler 并标准化 ==========
scaler = joblib.load("models/mlp_scaler.joblib")
X_val_scaled = scaler.transform(X_val)

# ========== 3. 加载模型并预测 ==========
model = joblib.load("models/mlp_model.pkl")
y_pred = model.predict(X_val_scaled)

# ========== 4. 计算评估指标 ==========
metrics = {
    "R2": float(r2_score(y_val, y_pred)),
    "MAE": float(mean_absolute_error(y_val, y_pred)),
    "MSE": float(mean_squared_error(y_val, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred)))
}

# ========== 5. 保存指标 ==========
os.makedirs("metrics", exist_ok=True)
with open("metrics/mlp_val_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ========== 6. 输出指标 ==========
print("✅ MLP 验证指标如下：")
for k, v in metrics.items():
    print(f"{k}: {v}")
