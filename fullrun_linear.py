# fullrun_linear.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载全量数据
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv")

# 模型训练
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 评估
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("✅ 全数据评估结果：")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")

# 输出图像
os.makedirs("models", exist_ok=True)

# 📉 残差图
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, y - y_pred, alpha=0.6)
plt.axhline(y=0, color="r", linestyle="--")
plt.title("Residual Plot (Full Data)")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/residual_linear_full.png")
plt.show()

# 📈 真实 vs 预测图
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Predicted vs Actual (Full Data)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/pred_vs_actual_linear_full.png")
plt.show()
