import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

# ========== 1. 加载全数据 ==========
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# ========== 2. 加载 scaler 并标准化 ==========
scaler = joblib.load("models/mlp_scaler.joblib")
X_scaled = scaler.transform(X)

# ========== 3. 加载已训练模型 ==========
model = joblib.load("models/mlp_model.pkl")

# ========== 4. 模型预测 ==========
y_pred = model.predict(X_scaled)

# ========== 5. 输出指标 ==========
metrics = {
    "R2": float(r2_score(y, y_pred)),
    "MAE": float(mean_absolute_error(y, y_pred)),
    "MSE": float(mean_squared_error(y, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y, y_pred)))
}
os.makedirs("metrics", exist_ok=True)
with open("metrics/mlp_full_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ MLP 全数据评估指标如下：")
for k, v in metrics.items():
    print(f"{k}: {v}")

# ========== 6. Loss 曲线图 ==========
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("MLP Loss Curve (Full Data)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/mlp_loss_curve_full.png")
plt.close()

# ========== 7. 预测 vs 实际图 ==========
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y)), y, label="Actual", alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.7)
plt.title("Prediction vs Actual (MLP Full Data)")
plt.xlabel("Sample")
plt.ylabel("Target")
plt.legend()
plt.tight_layout()
plt.savefig("models/mlp_pred_vs_actual_full.png")
plt.close()
