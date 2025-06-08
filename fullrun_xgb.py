import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib
from xgboost import plot_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载全数据
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# 加载已训练模型
model = joblib.load("models/xgb_model.pkl")

# 预测
y_pred = model.predict(X)

# 评估指标
metrics = {
    "R2": float(r2_score(y, y_pred)),
    "MAE": float(mean_absolute_error(y, y_pred)),
    "MSE": float(mean_squared_error(y, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y, y_pred)))
}

# 保存指标
os.makedirs("metrics", exist_ok=True)
with open("metrics/xgb_full_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ XGBoost 全数据评估指标如下：")
print(metrics)

# 📊 特征重要性图
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type="gain", xlabel="Gain")
plt.title("XGBoost Feature Importance (Full Data)")
plt.tight_layout()
os.makedirs("models", exist_ok=True)
plt.savefig("models/xgb_feature_importance_full.png")
plt.show()
