import numpy as np
from scipy.optimize import fsolve

points1 = np.load('A_photoCor.npy')
points2 = np.load('B_photoCor.npy')

#求旋轉矩陣的Fuction
def FindM(omegao, phio, kappao):
    omega = omegao * np.pi / 180
    phi = phio * np.pi / 180
    kappa = kappao * np.pi / 180
    M_omega = np.array([[1, 0, 0],
                        [0, np.cos(omega), np.sin(omega)],
                        [0, -np.sin(omega), np.cos(omega)]])
    M_phi = np.array([[np.cos(phi), 0, -np.sin(phi)],
                      [0, 1, 0],
                      [np.sin(phi), 0, np.cos(phi)]])
    M_kappa = np.array([[np.cos(kappa), np.sin(kappa), 0],
                        [-np.sin(kappa), np.cos(kappa), 0],
                        [0, 0, 1]])
    return np.dot(np.dot(M_kappa, M_phi), M_omega)

def objective(params, *args):
    X,Y,Z= params   ##x,y,z代表三個物點座標
    M, point,focal_lenght,XL,YL,ZL= args  ##points代表一個點的相片座標(x,y)
    P_array=np.array([X-XL,Y-YL,Z-ZL])
    p_array=np.array([point[0],point[1],-focal_lenght])
    # Evaluate the objective functi>>
    Function = np.dot(M,P_array)-p_array
    return Function

A_Point = np.empty(shape=(len(points1[:,0]),2))
B_Point=np.empty(shape=(len(points2[:,0]),2))

def img_to_photo(points,M,focal_lenght,XL,YL,ZL):
    P_Cor = np.empty(shape=(len(points[:,0]),3))
    initial_guess=[0,0,0]
    for i in range(len(points1[:,0])):
        point=points[i]
        result=fsolve(objective, initial_guess, args=(M, point,focal_lenght,XL,YL,ZL))
        P_Cor[i]=result
    return P_Cor

##A和B的相對方位
M1=FindM(0,0,0)
M2=FindM(5.9671801,4.89610752,-8.22253956)
focal_length = 10.26

np.save('A_ObjectCor(P).npy',img_to_photo(points1,M1,focal_length,0,0,0) )
np.save('B_ObjectCor(P).npy',img_to_photo(points2,M2,focal_length,-9.38,-1.7431715,7.88361065) )
print(img_to_photo(points2,M2,focal_length,-9.38,-1.7431715,7.88361065))