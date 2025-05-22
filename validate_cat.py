# validate_cat.py
import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载验证集
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 加载模型
model = CatBoostRegressor()
model.load_model("models/cat_model.cbm")

# 预测
y_pred = model.predict(X_val)

# 评估指标
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
with open("models/cat_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ CatBoost 验证指标：")
print(metrics)

# 📊 特征重要性可视化
importances = model.get_feature_importance(prettified=True)
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature Id'], importances['Importances'])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance (Validation Set)")
plt.tight_layout()
plt.savefig("models/cat_feature_importance.png")
plt.show()
