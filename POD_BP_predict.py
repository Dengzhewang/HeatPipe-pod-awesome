# 但愿感君一回顾 使我思君朝与暮

import torch
import numpy as np
from BPNN import BPNet
import Dataloader


x_min=np.array([1.035,823.15])
x_max= np.array([19.035, 1323.15])
y_T_min=np.array([-8889.074,-1703.7465,-626.8481,-189.90326,-39.54724,-19.978724])
y_T_max=np.array([1.2157947e+04,4.3377686e+02,4.2277777e+02,1.0848428e+02,5.5262806e+01,8.8568058e+00])

y_p_min=np.array([-5.3933555e+06,-3.7503003e+03,-9.0084943e+02,-6.2148779e+02,-2.5432315e+02])
y_p_max=np.array([2.28903050e+06,1.85640684e+04,9.26065369e+02,1.02172437e+03,4.10924103e+02])

y_U_min=np.array([-634.5996,-79.77345,-50.853672,-12.475959,-5.210249,-2.955774])
y_U_max=np.array([3.5075288e+03,2.8521164e+02,1.6015042e+02,4.3319473e+00,5.5213451e+00,2.0318761e+00])

y_uxy_min=np.array([-0.7185506,-0.08387373,-0.05890597,-0.05906147,-0.05144239,-0.01959782,-0.007846,-0.01445069])
y_uxy_max=np.array([0.5309019,0.26047018,0.03821242,0.04857168,0.01409886,0.04496875,0.01426748,0.00650996])
T_phi=np.loadtxt("T_pod_data/U_phi.csv",delimiter=",")
T_mean=np.loadtxt("T_pod_data/U_mean.csv",delimiter=",")
P_phi=np.loadtxt("P_pod_data/U_phi.csv",delimiter=",")
P_mean=np.loadtxt("P_pod_data/U_mean.csv",delimiter=",")
u_phi=np.loadtxt("U_pod_data/U_phi.csv",delimiter=",")
u_mean=np.loadtxt("U_pod_data/U_mean.csv",delimiter=",")
uxy_phi=np.loadtxt("uxy_pod_data/U_phi.csv",delimiter=",")
uxy_mean=np.loadtxt("uxy_pod_data/U_mean.csv",delimiter=",")
class POD_pre:
    def __init__(self,object):
        self.models=[]
        self.object=object
        if self.object=="T":
            for i in range(6):
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
        if self.object=="P":
            for i in range(5):
                model = BPNet()
                model.load_state_dict(
                    torch.load(f"POD_bp_best_model/model_P/pod{i + 1}.pth"),

                )
                model.eval()
                self.models.append(model)
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_p_min
            self.y_max = y_p_max
        if self.object=="U":
            for i in range(6):
                model = BPNet()
                model.load_state_dict(
                    torch.load(f"POD_bp_best_model/model_U/pod{i + 1}.pth"),

                )
                model.eval()
                self.models.append(model)
            self.x_min = x_min
            self.x_max = x_max
            self.y_min =np.array([-634.5996,-79.77345,-50.853672,-12.475959,-5.210249,-2.955774])
            self.y_max =np.array([3.5075288e+03,2.8521164e+02,1.6015042e+02,4.3319473e+00,5.5213451e+00,2.0318761e+00])

        if self.object=="uxy":
            for i in range(8):
                model = BPNet()
                model.load_state_dict(
                    torch.load(f"POD_bp_best_model/model_uxy/pod{i + 1}.pth"),

                )
                model.eval()
                self.models.append(model)

            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_uxy_min
            self.y_max = y_uxy_max









    def predict_coeff(self,x):
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

    def construct(self,x,phi,U_mean):
        A=self.predict_coeff(x)
        U_construct=U_mean+phi@A
        return U_construct
    def pod_pre(self,x):
        x=torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            model=self.models[5]
            a=model(x)
        coeffs=a*(8.8568058e+00+19.978724)-19.978724
        return coeffs
# pod=POD_pre(object="uxy")
# U_phi=uxy_phi[:,:8]
# x=[13.005,1073.15]
#
#
# coeffs=pod.predict_coeff(x)
#
# U_reconstruct=pod.construct(x,U_phi,uxy_mean)
#
# U_reconstruct=U_reconstruct.reshape(4,200)
# np.savetxt("pod_bp_uxy_predict_13.005KW_1073.15K",U_reconstruct,delimiter=",")

# dataloader=Dataloader.DataLine(
#         csv_path="T_pod_data/Dataset.CSV",
#         test_size=0.1,
#          batch_size=1,
#          random_state=12,
#           Z_score=False
#      )
# X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normalized, y_train_normalized, y_test_normaliezd=dataloader.load_and_split()
# train_loader, test_loader = dataloader.creat_dataloaders()
# print(X_means,X_stds,Y_means,Y_stds)
# coeffs=pod.pod_pre(X_train_normalized)
# print(coeffs.shape)
# print(coeffs)
# np.savetxt("coeffs_predict/bp_pod6_train_pre.csv",coeffs,delimiter=",")
try:
    target=float(input("1-----------温度场\n2-----------压力场\n3-------------蒸汽区速度场\n4----------------吸液芯速度场(标量）\n请输入预测目标："))
    P = float(input("输入功率："))
    Tc = float(input("输入冷凝段边界温度："))
except ValueError:
    print("请输入数值！")
x=[[P,Tc]]
x=np.array(x)
if target==1:
    U_phi = T_phi[:, :6]
    pod=POD_pre(object="T")
    U_construct=pod.construct(x,U_phi,T_mean)
    U_construct=U_construct.reshape(8,200)
    np.savetxt(f"RESULT_T_predict/pod_best_T_predict_{P}KW_{Tc}K", U_construct, delimiter=",")
elif target==2:
    U_phi = P_phi[:, :5]
    pod = POD_pre(object="P")
    U_construct = pod.construct(x, U_phi, P_mean)
    U_construct = U_construct.reshape(1, 200)
    np.savetxt(f"RESULT_P_predict/pod_best_P_predict_{P}KW_{Tc}K", U_construct, delimiter=",")

elif target==3:
    U_phi = u_phi[:, :6]
    pod = POD_pre(object="U")
    U_construct = pod.construct(x, U_phi, u_mean)
    U_construct = U_construct.reshape(1, 200)
    np.savetxt(f"RESULT_U_predict/pod_best_U_predict_{P}KW_{Tc}K", U_construct, delimiter=",")

elif target==4:
    U_phi = uxy_phi[:, :8]
    pod = POD_pre(object="uxy")
    U_construct = pod.construct(x, U_phi, uxy_mean)
    U_construct = U_construct.reshape(4, 200)
    np.savetxt(f"RESULT_uxy_predict/pod_best_uxy_predict_{P}KW_{Tc}K", U_construct, delimiter=",")





