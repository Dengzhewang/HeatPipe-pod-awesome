# 但愿感君一回顾 使我思君朝与暮

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import Dataloader
import joblib
from sklearn.model_selection import GridSearchCV
import os


models = []
dataloader = Dataloader.DataLine(
    csv_path="T_pod_data/Dataset.CSV",
    test_size=0.1,
    batch_size=1,
    random_state=12,
    Z_score=True
)
X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = dataloader.load_and_split()

n_modes=y_train_normalized.shape[1]
for i in range(6):
    model = joblib.load(f"xgb_models/xgb_mode{i+1}.pkl")
    models.append(model)

# Y_pred = []
#
# x=np.array([[13.005,1073.15]])

datanormalize=Dataloader.DataNormalizer(x_means=np.array([9.894037,1106.884]),x_stds=[5.776581,136.84344 ],
                      y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02]),
                      y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00]))
# x_norm=datanormalize.transform_x(x)



# for model in models:
#
#     y_pred = model.predict(x_norm)
#     Y_pred.append(y_pred)
#
#
# Y_pred = np.array(Y_pred).T
# print(Y_pred.shape)
# coeffs=datanormalize.inverse_transform(Y_pred)
# print(coeffs.shape)
# print(coeffs)
# coeffs=coeffs.T
# U_phi=np.loadtxt("T_pod_data/U_phi.csv",delimiter=",")
# U_phi=U_phi[:,:6]
# U_mean=np.loadtxt("T_pod_data/U_mean.csv",delimiter=",")
#
#
# U_pred=U_mean.reshape(1600,1)+U_phi@coeffs
#
# U_pred=U_pred.reshape(8,200)
# np.savetxt("pod_xgb_T_predict_13.005KW_1073.15K",U_pred,delimiter=",")
# for i in range(10):
#     Y_pred_i=Y_pred[:,i]
#     Y_pred[:,i]=datanormalize.inverse_transform(Y_pred_i)
# Y_pred=Y_pred.T
# Y_test=datanormalize.inverse_transform(y_test_normalized)
# mse = mean_squared_error(Y_test, Y_pred)
# print("Total MSE:", mse)
#
# mse_each = ((Y_test - Y_pred) ** 2).mean(axis=0)
# print("Each mode MSE:", mse_each)
# erro=Y_pred-Y_test
# print(f"erro_mean:{erro.mean()},erro_max:{erro.max()},erro_min:{erro.min()}")
# print(erro)
# np.savetxt('test_true.csv',Y_test,delimiter=",")
# np.savetxt("test_pred.csv",Y_pred,delimiter=",")

y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02])
y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00])
for i in range(6):
    if i>=2:
        model=models[i]
        a=model.predict(X_train_normalized)
        coeffs=a*y_stds[i]+y_means[i]
        print(coeffs.shape)
        np.savetxt(f"coeffs_predict/XGB_pod{i+1}_train_pre.csv",coeffs,delimiter=",")
