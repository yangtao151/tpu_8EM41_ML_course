import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# 忽略收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 加载数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").squeeze()
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").squeeze()

# 标准化（MLP 对特征尺度非常敏感）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 搜索空间
hidden_layers = [(100,), (50, 50), (100, 50), (64, 32)]
activations = ["relu", "tanh"]
solvers = ["adam"]
max_iters = [500, 1000]

print("📊 搜索最优 MLP 超参数组合：\n")

best_score = -np.inf
best_params = {}

for h in hidden_layers:
    for act in activations:
        for solver in solvers:
            for iters in max_iters:
                try:
                    model = MLPRegressor(
                        hidden_layer_sizes=h,
                        activation=act,
                        solver=solver,
                        max_iter=iters,
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)

                    r2 = r2_score(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

                    print(f"🧪 hidden={h}, act={act}, solver={solver}, max_iter={iters} => "
                          f"R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

                    if r2 > best_score:
                        best_score = r2
                        best_params = {
                            "hidden_layer_sizes": h,
                            "activation": act,
                            "solver": solver,
                            "max_iter": iters,
                            "R2": round(r2, 4),
                            "MAE": round(mae, 2),
                            "RMSE": round(rmse, 2)
                        }

                except Exception as e:
                    print(f"❌ Failed: hidden={h}, act={act}, solver={solver}, max_iter={iters} -- {e}")

print("\n✅ 最佳参数组合：")
for k, v in best_params.items():
    print(f"{k}: {v}")
