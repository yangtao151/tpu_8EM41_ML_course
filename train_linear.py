import pandas as pd
import joblib
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 创建输出目录（如果不存在）
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# 加载训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 保存模型为 .pkl 格式
joblib.dump(model, "models/linear_model.pkl")

# 进行预测并计算指标
y_pred = model.predict(X_train)

metrics = {
    "R2": float(r2_score(y_train, y_pred)),
    "MAE": float(mean_absolute_error(y_train, y_pred)),
    "MSE": float(mean_squared_error(y_train, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_train, y_pred)))
}

# 保存指标为 JSON
with open("metrics/linear_train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 线性回归模型训练完成，模型已保存至 models/linear_model.pkl")
print("📉 训练指标已保存至 metrics/linear_train_metrics.json")
