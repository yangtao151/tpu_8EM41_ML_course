import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载验证集
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 加载模型 (.pkl)
model = joblib.load("models/catboost_model.pkl")

# 预测
y_pred = model.predict(X_val)

# 计算指标
metrics = {
    "R2": float(r2_score(y_val, y_pred)),
    "MAE": float(mean_absolute_error(y_val, y_pred)),
    "MSE": float(mean_squared_error(y_val, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred)))
}

# 保存指标
os.makedirs("metrics", exist_ok=True)
with open("metrics/cat_val_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ CatBoost 验证指标如下：")
print(metrics)

# 📊 特征重要性图
importances = model.get_feature_importance(prettified=True)

plt.figure(figsize=(10, 6))
plt.barh(importances['Feature Id'], importances['Importances'])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance (Validation Set)")
plt.tight_layout()
plt.savefig("models/cat_feature_importance.png")
plt.show()
