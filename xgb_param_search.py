# xgb_param_search.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# 加载数据
X = pd.read_csv("prepare/X_train.csv")
y = pd.read_csv("prepare/y_train.csv").squeeze()

# 拆分验证集（如果没有 X_val）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("📊 搜索最优 XGBoost 超参数组合：\n")

# 参数网格
depths = [3, 4, 6]
etas = [0.03, 0.1]
n_estimators = [300, 500]

# 网格搜索
for d in depths:
    for lr in etas:
        for n in n_estimators:
            model = XGBRegressor(max_depth=d, learning_rate=lr, n_estimators=n, random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = mean_squared_error(y_val, y_pred) ** 0.5
            print(f"🧪 depth={d}, lr={lr}, iter={n} => R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
