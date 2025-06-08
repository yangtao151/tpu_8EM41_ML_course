import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import itertools
import numpy as np

# 加载数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 定义要搜索的超参数
depth_list = [4, 6, 8]
learning_rate_list = [0.03, 0.1]
iterations_list = [300, 500]

# 枚举所有组合
search_space = list(itertools.product(depth_list, learning_rate_list, iterations_list))

# 搜索
print("📊 搜索最优 CatBoost 超参数组合：\n")
for depth, lr, iters in search_space:
    model = CatBoostRegressor(
        depth=depth,
        learning_rate=lr,
        iterations=iters,
        verbose=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
  
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"🧪 depth={depth}, lr={lr}, iter={iters} => R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
