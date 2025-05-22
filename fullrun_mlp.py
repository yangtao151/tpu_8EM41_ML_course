# fullrun_mlp.py
import pandas as pd
import os
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 加载全量数据
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# 模型训练
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                     max_iter=1000, random_state=42, verbose=False)
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 输出评估指标
print("✅ MLP 全数据训练评估：")
print(f"R²   = {r2_score(y, y_pred):.4f}")
print(f"MAE  = {mean_absolute_error(y, y_pred):.2f}")
print(f"MSE  = {mean_squared_error(y, y_pred):.2f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_pred)):.2f}")

# 可视化训练过程损失收敛
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("MLP Loss Curve (Full Data)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/mlp_loss_curve_full.png")
plt.show()
