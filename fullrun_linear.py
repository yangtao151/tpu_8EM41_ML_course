import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# 创建目录
os.makedirs("metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 加载完整数据
X_full = pd.read_csv("prepare/X_full.csv")
y_full = pd.read_csv("prepare/y_full.csv")

# 加载模型
model = joblib.load("models/linear_model.pkl")

# 预测
y_pred = model.predict(X_full)

# 评估指标
metrics = {
    "R2": float(r2_score(y_full, y_pred)),
    "MAE": float(mean_absolute_error(y_full, y_pred)),
    "MSE": float(mean_squared_error(y_full, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_full, y_pred)))
}

# 保存指标
with open("metrics/linear_full_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 全数据评估结果：")
print(metrics)

# 📉 残差图
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, y_full - y_pred, alpha=0.6)
plt.axhline(y=0, color="r", linestyle="--")
plt.title("Residual Plot (Full Data)")
plt.xlabel("Predicted Value")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/residual_linear_full.png")
plt.show()

# 📈 真实 vs 预测图
plt.figure(figsize=(8, 5))
plt.scatter(y_full, y_pred, alpha=0.6)
plt.plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'r--')
plt.title("Predicted vs Actual (Full Data)")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/pred_vs_actual_linear_full.png")
plt.show()
