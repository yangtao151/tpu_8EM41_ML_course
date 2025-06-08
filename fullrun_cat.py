import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载全量数据
X = pd.read_csv("prepare/X_full.csv")
y = pd.read_csv("prepare/y_full.csv").squeeze()

# 加载已训练好的 CatBoost 模型
model = joblib.load("models/catboost_model.pkl")

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
with open("metrics/cat_full_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ CatBoost 全数据评估指标如下：")
for k, v in metrics.items():
    print(f"{k}: {v}")

# 特征重要性图
importances = model.get_feature_importance(prettified=True)

plt.figure(figsize=(10, 6))
plt.barh(importances['Feature Id'], importances['Importances'])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance (Full Data)")
plt.tight_layout()
os.makedirs("models", exist_ok=True)
plt.savefig("models/cat_feature_importance_full.png")
plt.show()
