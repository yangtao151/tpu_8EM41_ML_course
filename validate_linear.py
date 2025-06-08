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

# 加载验证集数据
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv")

# 加载模型（注意使用.pkl）
model = joblib.load("models/linear_model.pkl")

# 预测
y_pred = model.predict(X_val)

# 计算验证指标
metrics = {
    "R2": float(r2_score(y_val, y_pred)),
    "MAE": float(mean_absolute_error(y_val, y_pred)),
    "MSE": float(mean_squared_error(y_val, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred)))
}

# 保存验证指标
with open("metrics/linear_val_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 验证指标如下：")
print(metrics)

# 📊 残差图
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, y_val - y_pred, alpha=0.6)
plt.axhline(y=0, color="r", linestyle="--")
plt.title("Residual Plot (Validation Set)")
plt.xlabel("Predicted Value")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/residual_linear.png")
plt.show()

# 📈 真实 vs 预测图
plt.figure(figsize=(8, 5))
plt.scatter(y_val, y_pred, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title("Predicted vs Actual (Validation Set)")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/pred_vs_actual_linear.png")
plt.show()
