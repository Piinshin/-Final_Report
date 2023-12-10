import numpy as np
from scipy.optimize import fsolve,leastsq

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

##求出旋轉矩陣的逆矩陣Mt
def Mt(omegao, phio, kappao):
    return np.linalg.inv(FindM(omegao,phio,kappao))

def objective(params, *args):
    x, y, z ,by,bz= params   ##x,y,z代表omega, phi, kappa
    M0, points1, points2, focal_length, bx= args
    Munknown = Mt(x, y, z)
    A_matrix =[]             ##所有條件參數存在這
    b=np.array([bx,by,bz])
    for i in range(len(points1[:, 0])):
        v1 = np.append(points1[i, :], -focal_length)
        A1 = np.dot(M0, v1  )
        v2 = np.append(points2[i, :], -focal_length)
        A2 = np.dot(Munknown, v2)
        # Construct the matrix for the system of equations
        b = np.array([bx, by, bz])
        A_matrix.append(np.cross(A1, A2))
    A_matrix = np.array(A_matrix)
    A_matrix=np.transpose(A_matrix)
    # Evaluate the objective function>>b dot.(A1xA2)=0
    Function = np.dot(b,A_matrix)
    return Function

# 已知值Given values
M0 = Mt(0, 0, 0)
focal_length = 10.26  # 焦距
points1 = np.load('A_photoCor.npy')
points2 = np.load('B_photoCor.npy')
bx=-9.38
initial_guess = [26,20, -20,-8,30]

##用最小二乘法求解
result, _ = leastsq(objective, initial_guess, args=(M0, points1, points2, focal_length, bx))

# Print the result
print("  Solution for omg,phi, kappa:", result[0:3],'\n',"Solution for by bz",result[3:5])

residuals_at_optimized_params = objective(result, M0, points1, points2, focal_length, bx)
cost_at_optimized_params = np.sum(residuals_at_optimized_params ** 2)
print("Residuals at optimized parameters:", residuals_at_optimized_params)
print("Cost at optimized parameters:", cost_at_optimized_params)