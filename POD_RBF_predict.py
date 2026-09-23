# 但愿感君一回顾 使我思君朝与暮

from RBFNN import RBFNN
import os
import numpy as np
from Dataloader import DataNormalizer
from Dataloader import DataLine
def load_rbb_model(name):
    model=RBFNN(n_centers=1,sigma_scale=1)

    model.centers=np.load(f"rbf_models/{name}_centers.npy")
    model.sigma = np.load(f"rbf_models/{name}_sigma.npy")
    model.sigma_scale = np.load(f"rbf_models/{name}_sigma_scale.npy")
    model.W= np.load(f"rbf_models/{name}_w.npy")

    model.n_centers=model.centers[0]
    return model
models=[]
for i in range(6):
    model=load_rbb_model(f"pod{i+1}")
    models.append(model)



# x=np.array([[13.005,1073.15]])
# datanormalize=DataNormalizer(x_means=np.array([9.894037,1106.884]),x_stds=[5.776581,136.84344 ],
#                       y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02]),
#                       y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00]))
# x_norm=datanormalize.transform_x(x)
# coeffs_norm=[]
# for model in models:
#     a=model.predict(x_norm)
#     coeffs_norm.append(a.item())
#     print(coeffs_norm)
#
# coeffs=datanormalize.inverse_transform(coeffs_norm)
# print(coeffs)
# coeffs=np.array(coeffs).reshape(6,1)
# print(coeffs.shape)
# U_phi=np.loadtxt("T_pod_data/U_phi.csv",delimiter=",")
# U_phi=U_phi[:,:6]
# U_mean=np.loadtxt("T_pod_data/U_mean.csv",delimiter=",")
# U_pred=U_mean.reshape(1600,1)+U_phi@coeffs
# print(U_mean.shape)
# print(U_phi.shape)
# print(coeffs.shape)
# print(U_pred.shape)
#
# U_pred=U_pred.reshape(8,200)
# np.savetxt("pod_rbf_T_predict_13.005KW_1073.15K",U_pred,delimiter=",")
dataloader = DataLine(
        csv_path="T_pod_data/Dataset.CSV",
        test_size=0.1,
         batch_size=1,
         random_state=12,
          Z_score=True
     )
X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normalized, y_train_normalized, y_test_normaliezd=dataloader.load_and_split()
train_loader, test_loader = dataloader.creat_dataloaders()
print(X_means,X_stds,Y_means,Y_stds)
y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02])
y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00])
for i in range(6):
    if i>=2:
        model=models[i]
        a=model.predict(X_test_normalized)
        coeffs=a*y_stds[i]+y_means[i]
        print(coeffs.shape)
        np.savetxt(f"coeffs_predict/RBF_pod{i+1}_test_pre.csv",coeffs,delimiter=",")