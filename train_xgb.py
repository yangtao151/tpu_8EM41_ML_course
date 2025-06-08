import pandas as pd
import joblib
import os
import yaml
import json
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 读取超参数
with open("params.yaml") as f:
    params = yaml.safe_load(f)["xgb"]

# 加载训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()

# 创建并训练模型
model = XGBRegressor(
    max_depth=params["max_depth"],
    learning_rate=params["learning_rate"],
    n_estimators=params["n_estimators"],
    random_state=params["random_state"]
)
model.fit(X_train, y_train)

# 保存模型为 .pkl
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")

# 训练预测与指标
y_pred = model.predict(X_train)
metrics = {
    "R2": float(r2_score(y_train, y_pred)),
    "MAE": float(mean_absolute_error(y_train, y_pred)),
    "MSE": float(mean_squared_error(y_train, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_train, y_pred)))
}

# 保存指标为 JSON
os.makedirs("metrics", exist_ok=True)
with open("metrics/xgb_train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ XGBoost 模型训练完成，模型已保存至 models/xgb_model.pkl")
print("📉 训练指标保存至 metrics/xgb_train_metrics.json")
