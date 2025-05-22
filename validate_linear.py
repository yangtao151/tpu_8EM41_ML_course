# validate_linear.py
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# 加载验证集
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv")

# 加载模型
model = joblib.load("models/linear_model.joblib")

# 预测
y_pred = model.predict(X_val)

# 指标计算
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

metrics = {
    "R2": round(r2, 4),
    "MAE": round(mae, 2),
    "MSE": round(mse, 2),
    "RMSE": round(rmse, 2)
}

# 保存指标
os.makedirs("models", exist_ok=True)
with open("models/linear_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 验证指标：")
print(metrics)

# 📊 残差图
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, y_val - y_pred, alpha=0.6)
plt.axhline(y=0, color="r", linestyle="--")
plt.title("Residual Plot (Validation Set)")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/residual_linear.png")
plt.show()

# 📈 真实 vs 预测 散点图
plt.figure(figsize=(8, 5))
plt.scatter(y_val, y_pred, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.title("Predicted vs Actual (Validation Set)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/pred_vs_actual_linear.png")
plt.show()
