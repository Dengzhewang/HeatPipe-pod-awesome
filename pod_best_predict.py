# 但愿感君一回顾 使我思君朝与暮
import torch
import numpy as np
from BPNN import BPNet
from Dataloader import DataNormalizer
from RBFNN import RBFNN
import Dataloader
import time



x_min=np.array([1.035,823.15])
x_max= np.array([19.035, 1323.15])
y_T_min=np.array([-1703.7465,-626.8481,-189.90326,-39.54724,-19.978724])
y_T_max=np.array([4.3377686e+02,4.2277777e+02,1.0848428e+02,5.5262806e+01,8.8568058e+00])
y_T_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02])
y_T_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00])
y_P_min=np.array([-3.7503003e+03,-9.0084943e+02,-6.2148779e+02,-2.5432315e+02])
y_P_max=np.array([1.85640684e+04,9.26065369e+02,1.02172437e+03,4.10924103e+02])
T_phi=np.loadtxt("T_pod_data/U_phi.csv",delimiter=",")
T_mean=np.loadtxt("T_pod_data/U_mean.csv",delimiter=",")
P_phi=np.loadtxt("P_pod_data/U_phi.csv",delimiter=",")
P_mean=np.loadtxt("P_pod_data/U_mean.csv",delimiter=",")
datanormalize=DataNormalizer(x_means=np.array([9.894037,1106.884]),x_stds=[5.776581,136.84344 ],
                      y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02]),
                      y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00]))

class POD_pre:
    def __init__(self,object):
        self.models=[]
        self.object=object
        if object=="T":
            for i in range(6):
                if i>=1:
                    model=BPNet()
                    model.load_state_dict(
                        torch.load(f"POD_bp_best_model/pod{i+1}.pth"),

                    )
                    model.eval()
                    self.models.append(model)

            self.x_min=x_min
            self.x_max=x_max
            self.y_min=y_T_min
            self.y_max=y_T_max
        elif object=="P":
            for i in range(5):
                if i >= 1:
                    model = BPNet()
                    model.load_state_dict(
                        torch.load(f"POD_bp_best_model/model_P/pod{i + 1}.pth"),

                    )
                    model.eval()
                    self.models.append(model)

            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_P_min
            self.y_max = y_P_max


    def load_rbf_model(self,name,object):
        model = RBFNN(n_centers=1, sigma_scale=1)
        if object=="T":

            model.centers = np.load(f"rbf_models/{name}_centers.npy")
            model.sigma = np.load(f"rbf_models/{name}_sigma.npy")
            model.sigma_scale = np.load(f"rbf_models/{name}_sigma_scale.npy")
            model.W = np.load(f"rbf_models/{name}_w.npy")
            model.n_centers = model.centers[0]
            return model
        elif object=="P":
            model.centers = np.load(f"P_models/rbf_models/{name}_centers.npy")
            model.sigma = np.load(f"P_models/rbf_models/{name}_sigma.npy")
            model.sigma_scale = np.load(f"P_models/rbf_models/{name}_sigma_scale.npy")
            model.W = np.load(f"P_models/rbf_models/{name}_w.npy")
            model.n_centers = model.centers[0]
            return model



    def predict_coeff_pod_other(self,x):
        x=np.array(x)
        x_norm = (x-self.x_min)/(self.x_max-self.x_min)
        x_norm = torch.tensor(x_norm,dtype=torch.float32)
        coeffs=[]
        with torch.no_grad():
            for model in self.models:
                a=model(x_norm)
                coeffs.append(a.item())
        coeffs=np.array(coeffs)
        coeffs=coeffs*(self.y_max-self.y_min)+self.y_min
        return coeffs
    def predict_coeff_pod1(self,x,object):
        if object=="P":
            x_means = np.array([9.5265,1124.3992])
            x_stds =np.array([5.8709874,129.50112])
            x_norm=(x-x_means)/x_stds
            model = self.load_rbf_model("pod1",object="P")
            coeff_norm=model.predict(x_norm)
            coeff_norm=np.array(coeff_norm)
            coeff=coeff_norm*2199223.5+29093.656
            return coeff
        if object=="T":
            x_means = np.array([9.894037,1106.884])
            x_stds = np.array([5.776581,136.84344])
            x_norm = (x - x_means) / x_stds
            model = self.load_rbf_model("pod1",object="T")
            coeff_norm = model.predict(x_norm)
            coeff_norm = np.array(coeff_norm)
            coeff = coeff_norm * 5.5085073e+03 + 2.6540762e+02
            return coeff


    def construct(self,x,phi,U_mean):
        coeffs_pod1=self.predict_coeff_pod1(x,object=self.object)
        coeffs_pod_other=self.predict_coeff_pod_other(x)
        coeffs=np.append(coeffs_pod1,coeffs_pod_other)
        U_construct=U_mean+phi@coeffs
        return U_construct




# x=[13.005,1073.15]

try:
    target=float(input("1-----------温度场\n2-----------压力场\n请输入预测目标："))
    P = float(input("输入功率："))
    Tc = float(input("输入冷凝段边界温度："))
except ValueError:
    print("请输入数值！")
start_time = time.time()
x=[[P,Tc]]
x=np.array(x)
# x=np.array([[13.005,1073.15]])
if target==1:

    U_phi = T_phi[:, :6]
    pod=POD_pre(object="T")
    U_construct=pod.construct(x,U_phi,T_mean)
    U_construct=U_construct.reshape(8,200)
    end_time=time.time()
    # np.savetxt(f"RESULT_T_predict/pod_best_T_predict_{P}KW_{Tc}K", U_construct, delimiter=",")
    print(f"计算用时：{end_time-start_time}")
elif target==2:
    U_phi = P_phi[:, :5]
    pod = POD_pre(object="P")
    U_construct = pod.construct(x, U_phi, P_mean)
    U_construct = U_construct.reshape(1, 200)
    end_time = time.time()
    print(f"计算用时：{end_time - start_time}")

    # np.savetxt(f"RESULT_P_predict/pod_best_P_predict_{P}KW_{Tc}K", U_construct, delimiter=",")


