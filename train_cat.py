import pandas as pd
import joblib
import os
import yaml
import json
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()

# 加载超参数
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["cat"]

# 创建并训练模型
model = CatBoostRegressor(
    depth=params["depth"],
    learning_rate=params["learning_rate"],
    iterations=params["iterations"],
    random_state=params["random_state"],
    verbose=0
)
model.fit(X_train, y_train)

# 保存模型为 .pkl
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/catboost_model.pkl")

# 训练预测并计算指标
y_pred = model.predict(X_train)
metrics = {
    "R2": float(r2_score(y_train, y_pred)),
    "MAE": float(mean_absolute_error(y_train, y_pred)),
    "MSE": float(mean_squared_error(y_train, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_train, y_pred)))
}

# 保存指标
os.makedirs("metrics", exist_ok=True)
with open("metrics/cat_train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ CatBoost 模型训练完成，模型已保存至 models/catboost_model.pkl")
print("📉 训练指标保存至 metrics/cat_train_metrics.json")
