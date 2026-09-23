# 但愿感君一回顾 使我思君朝与暮

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import Dataloader
from sklearn.model_selection import GridSearchCV
import joblib
import os
# ==================================================
# =======================模型加载====================
# ==================================================


models = []
dataloader = Dataloader.DataLine(
    csv_path="T_pod_data/Dataset.CSV",
    test_size=0.1,
    batch_size=1,
    random_state=12,
    Z_score=True
)
X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = dataloader.load_and_split()

for i in range(6):

    model = joblib.load(f"svr_models/svr_mode{i+1}.pkl")

    models.append(model)

# for i in range(6):
#     model1=models[i]
#
#     Y_pred_norm=model1.predict(X_test_normalized)
#     mse = mean_squared_error(y_test_normalized[:,[i]], Y_pred_norm)
#     print(f"Test MSE_pod{i+1} =", mse)



datanormalizer=Dataloader.DataNormalizer(x_means=np.array([9.894037,1106.884]),x_stds=[5.776581,136.84344 ],
                      y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02]),
                      y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00]))

# x = np.array([13.005,1223.15]).reshape(1,-1)
# x_norm=datanormalizer.transform_x(x)

# coeffs = []
#
# for model in models:
#
#     a = model.predict(X_test_normalized)
#
#     coeffs.append(a)
#
# coeffs=np.array(coeffs)
# print(coeffs.shape)
# for i in range(10):
#     coeffs[:,i]=datanormalizer.inverse_transform(coeffs[:,i])
# # coeffs=datanormalizer.inverse_transform(coeffs)
# coeffs=np.array(coeffs).T
# y_test=datanormalizer.inverse_transform(y_test_normalized)
# mse = mean_squared_error(coeffs, y_test)
# print("Total MSE:", mse)
#
# mse_each = ((y_test - coeffs) ** 2).mean(axis=0)
# print("Each mode MSE:", mse_each)
# erro=coeffs-y_test
# print(f"erro_mean:{erro.mean()},erro_max:{erro.max()},erro_min:{erro.min()}")
# print(erro)






# print(coeffs.shape)
# U_phi=np.loadtxt("T_pod_data/U_phi.csv",delimiter=",")
# U_phi=U_phi[:,:6]
# U_mean=np.loadtxt("T_pod_data/U_mean.csv",delimiter=",")
# U_pred=U_mean.reshape(1600,1)+U_phi@coeffs
#
# U_pred=U_pred.reshape(8,200)
# np.savetxt("pod_svr_T_predict_13.005KW_1223.15K",U_pred,delimiter=",")
y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02])
y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00])
for i in range(6):
    if i>=2:
        model=models[i]
        a=model.predict(X_train_normalized)
        coeffs=a*y_stds[i]+y_means[i]
        print(coeffs.shape)
        np.savetxt(f"coeffs_predict/SVR_pod{i+1}_train_pre.csv",coeffs,delimiter=",")
