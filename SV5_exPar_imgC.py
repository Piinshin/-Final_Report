import numpy as np
from scipy.optimize import fsolve

P1=[-16.53576115 ,-21.42264382,  30.98636507]
P2= [ -19.35470099 ,-13.70082442,  31.77934703]
P3= [-20.31460925 ,-16.31784169 , 24.51734023]

p1=(5.513, 6.832)
p2=(6.629, 4.274)
p3=(8.405, 6.905)

##求M矩陣
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
    XL, YL, ZL,omega, phi, kappa= params   ##待求值
    f= 10.26  ##相機焦距
    X1, Y1, Z1 = P1[0], P1[1], P1[2]
    X2, Y2, Z2 = P2[0], P2[1], P2[2]
    X3, Y3, Z3 = P3[0], P3[1], P3[2]  ##三控制點物點坐標
    x1, x2, x3, y1, y2, y3 = p1[0], p2[0], p3[0], p1[1], p2[1], p3[1] ##三控制點的相片坐標
    M=FindM(omega,phi,kappa)
    equation1=x1+f*(M[0,0]*(X1-XL)+M[0,1]*(Y1-YL)+M[0,2]*(Z1-ZL))/(M[2,0]*(X1-XL)+M[2,1]*(Y1-YL)+M[2,2]*(Z1-ZL))
    equation2=y1+f*(M[1,0]*(X1-XL)+M[1,1]*(Y1-YL)+M[1,2]*(Z1-ZL))/(M[2,0]*(X1-XL)+M[2,1]*(Y1-YL)+M[2,2]*(Z1-ZL))
    equation3=x2+f*(M[0,0]*(X2-XL)+M[0,1]*(Y2-YL)+M[0,2]*(Z2-ZL))/(M[2,0]*(X2-XL)+M[2,1]*(Y2-YL)+M[2,2]*(Z2-ZL))
    equation4=y2+f*(M[1,0]*(X2-XL)+M[1,1]*(Y2-YL)+M[1,2]*(Z2-ZL))/(M[2,0]*(X2-XL)+M[2,1]*(Y2-YL)+M[2,2]*(Z2-ZL))
    equation5=x3+f*(M[0,0]*(X3-XL)+M[0,1]*(Y3-YL)+M[0,2]*(Z3-ZL))/(M[2,0]*(X3-XL)+ M[2,1]*(Y3-YL)+M[2,2]*(Z3-ZL))
    equation6=y3+f*(M[1,0]*(X3-XL)+M[1,1]*(Y3-YL)+M[1,2]*(Z3-ZL))/(M[2,0]*(X3-XL)+M[2,1] *(Y3-YL)+M[2,2]*(Z3- ZL))
    return [equation1, equation2, equation3, equation4,equation5,equation6]

initial_guess=[0,0,0,0,0,0]
result = fsolve(equations, initial_guess)
np.set_printoptions(suppress=True)
print('照片C的外方為參數數(XL,YL,ZL,omega,phi,kappa)=',result)