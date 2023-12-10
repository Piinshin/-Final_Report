import numpy as np
from scipy.optimize import leastsq

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

def equations(params, *args):
    X, Y, Z = params  ##x,y,z代表三個物點座標
    M1, M2,x,y,xx,yy,f, XL, YL, ZL, XLL, YLL, ZLL = args
    equation1=x+f*(M1[0,0]*(X-XL)+M1[0,1]*(Y-YL)+M1[0,2]*(Z-ZL))/(M1[2,0]*(X-XL)+M1[2,1]*(Y-YL)+M1[2,2]*(Z-ZL))
    equation2=y+f*(M1[1,0]*(X-XL)+M1[1,1]*(Y-YL)+M1[1,2]*(Z-ZL))/(M1[2,0]*(X-XL)+M1[2,1]*(Y-YL)+M1[2,2]*(Z-ZL))
    equation3=xx+f*(M2[0,0]*(X-XLL)+M2[0,1]*(Y-YLL)+M2[0,2]*(Z-ZLL))/(M2[2,0]*(X-XLL)+M2[2,1]*(Y-YLL)+M2[2,2]*(Z-ZLL))
    equation4=yy+f*(M2[1,0]*(X-XLL)+M2[1,1]*(Y-YLL)+M2[1,2]*(Z-ZLL))/(M2[2,0]*(X-XLL)+M2[2,1]*(Y-YLL)+M2[2,2]*(Z-ZLL))
    return [equation1, equation2, equation3, equation4]

##A和B的相對方位
M1=FindM(0,0,0)
M2=FindM(4.10291277 ,18.70862634 ,-7.51208004) ##
focal_length = 10.26
XL,YL,ZL=0,0,0
XLL,YLL,ZLL=-9.38,-0.62539264,-1.73236062  ##
point1 = np.load('A_photoCor.npy')
point2 = np.load('B_photoCor.npy')
initial_guess=[-1,-1,-1]

print(len(point1[:,0]))
P_Cor = np.empty(shape=(len(point1[:,0]),3))
for i in range(len(point1[:,0])):
    x,y=point1[i,0],point1[i,1]
    xx,yy=point2[i,0],point2[i,1]
    result,_ = leastsq(equations, initial_guess, args=(M1, M2,x,y,xx,yy, focal_length, XL, YL, ZL, XLL, YLL, ZLL))
    print(result)
    P_Cor[i] = result
np.save('ObjectCor(P).npy',P_Cor)