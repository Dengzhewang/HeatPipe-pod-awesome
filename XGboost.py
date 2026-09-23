# 但愿感君一回顾 使我思君朝与暮
# XGBoost 示例：2个输入 -> 5个输出
# 提示：运行前需要安装 xgboost 库： pip install xgboost scikit-learn

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import Dataloader
import joblib
from sklearn.model_selection import GridSearchCV
import os

dataloader = Dataloader.DataLine(
    csv_path="P_pod_data/An_data.CSV",
    test_size=0.1,
    batch_size=1,
    random_state=12,
    Z_score=True
)
X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = dataloader.load_and_split()

# -------------------------
# 2) 建立 XGBoost 模型
# -------------------------
# 先定义一个单输出的 XGBoost 回归器
# xgb_single = XGBRegressor(
#     n_estimators=400,      # 树的数量（接力赛有多少个人参加，100~500常见）
#     learning_rate=0.05,     # 学习率（每个人纠错的步子迈多大。步子越小越稳，但需要更多的树）
#     max_depth=5,           # 每棵树的最大深度（树越深，能学的规律越复杂，但也容易过拟合。3~7常见）
#     random_state=42
# )

# 用 MultiOutputRegressor 把它包装成多输出模型 (同时预测5个系数)
# 注意：树模型不需要像 BPNN 或 SVR 那样必须做输入标准化 (StandardScaler)
# 因为树是靠“切分点”工作的，特征的绝对数值大小不影响树的切分。
# model = MultiOutputRegressor(xgb_single)

# -------------------------
# 3) 训练模型
# -------------------------
# model.fit(X_train_normalized, y_train_normalized)

# -------------------------
# 4) 测试模型
# -------------------------
# Y_pred = model.predict(X_test_normlized)
#
# print("总体均方误差 (MSE) =", mean_squared_error(y_test_normaliezd, Y_pred))
#
# # 看看每个 POD 系数的误差
# mse_each = ((y_test_normaliezd - Y_pred) ** 2).mean(axis=0)
# print("每个系数(a1~a5)的 MSE:", mse_each)



param_grid = {
    "n_estimators": [100, 200, 300,400,500],
    "learning_rate": [0.01,0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
n_modes = y_train_normalized.shape[1]

models = []

os.makedirs("xgb_models",exist_ok=True)
for i in range(n_modes):
    print(f"\n===== Training mode {i + 1} =====")

    y_train_i = y_train_normalized[:,i]
    print(y_train_i.shape)

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=12
    )

    grid = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1
    )

    grid.fit(X_train_normalized, y_train_i)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    best_model = grid.best_estimator_

    models.append(best_model)

    # 保存模型
    joblib.dump(best_model, f"P_models/xgb_models/xgb_mode{i + 1}.pkl")
