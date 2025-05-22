# validate_xgb.py
import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 加载验证数据
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 加载已训练模型
model = xgb.XGBRegressor()
model.load_model("models/xgb_model.json")

# 模型预测
y_pred = model.predict(X_val)

# 评估指标
metrics = {
    "R2": round(r2_score(y_val, y_pred), 4),
    "MAE": round(mean_absolute_error(y_val, y_pred), 2),
    "MSE": round(mean_squared_error(y_val, y_pred), 2),
    "RMSE": round(np.sqrt(mean_squared_error(y_val, y_pred)), 2)
}

# 保存指标
os.makedirs("models", exist_ok=True)
with open("models/xgb_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ 验证集评估指标：")
for k, v in metrics.items():
    print(f"{k}: {v}")

# 可视化：特征重要性图（验证集）
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type="gain", xlabel="Gain")
plt.title("XGBoost Feature Importance (Validation Set)")
plt.tight_layout()
plt.savefig("models/xgb_feature_importance.png")
plt.show()
