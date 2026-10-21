import numpy as np
from string_fem import *
import matplotlib.pyplot as plt



L = 0.33        # longueur vibrante [m]
N_ELEM = 11     # nombre d'éléments
MU = 3.1e-3     # masse linéique [kg/m]
T = 51.9        # tension corde sol [N]
#X_ARCH = 0.03   # abscisse archet [m]

m = mat_masse_elem(MU, 0.03)
M = assemblage(m, N_ELEM)
M_red = M[1:-1, 1:-1]

k = mat_raideur_elem(T, 0.03)
K = assemblage(k, N_ELEM)
K_red = K[1:-1, 1:-1]

#plt.spy(M)
# plt.spy(M_red)
# plt.show()

F = np.array([0 for _ in range(N_ELEM+1)]) #check indices
F[1] = 1
F_red = F[1:-1]

U_red = np.linalg.solve(K_red, F_red)
U = [0 for _ in range(N_ELEM+1)]
for i in range(1, N_ELEM) :
    U[i] = U_red[i-1]

U_mm = [u * 1e3 for u in U]
x = list(mesh_1d(L, N_ELEM)[0])
plt.plot(x, U_mm)
plt.xlabel("x (m)")
plt.ylabel("U(x) (mm)")
plt.title("Résolution statique : KU = F")
plt.show()