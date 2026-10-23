import numpy as np
from string_fem import *
import matplotlib.pyplot as plt
import scipy as sp
from math import cos

L = 0.33        # longueur vibrante [m]
N_ELEM = 329     # nombre d'éléments  # multiple de 11, mais on pense que -1
MU = 3.1e-3     # masse linéique [kg/m]
T = 51.9        # tension corde sol [N]
#X_ARCH = 0.03   # abscisse archet [m]

m = mat_masse_elem(MU, 0.03)
M = assemblage(m, N_ELEM)
M_red = M[1:-1, 1:-1]

k = mat_raideur_elem(T, 0.03)
K = assemblage(k, N_ELEM)
K_red = K[1:-1, 1:-1]

# plt.spy(M)
# plt.spy(M_red)
# plt.show()

F = np.zeros(N_ELEM+1) #check indices
F[1] = 1
F_red = F[1:-1]

## Résolution statique

U_red = np.linalg.solve(K_red, F_red)
U = [0 for _ in range(N_ELEM+1)]
for i in range(1, N_ELEM) :
    U[i] = U_red[i-1]

# U_mm = [u * 1e3 for u in U]
# x = list(mesh_1d(L, N_ELEM)[0])
# plt.plot(x, U_mm)
# plt.xlabel("x (m)")
# plt.ylabel("U(x) (mm)")
# plt.title("Résolution statique : KU = F")
#plt.show()

## Fréquences propres

l, v_red = sp.linalg.eigh(K_red, M_red)
omegas = np.sqrt(l)
frequences = omegas / (2*np.pi)

#v = [ [0 for _ in range(len(v_red[0]))] for _ in range(len(v_red)+2)]
v = np.zeros((len(v_red)+2, len(v_red[0])))
for i in range(1, len(v)-1) :
    v[i] = v_red[i-1]
#print(v)

colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'cyan', 'yellow', 'magenta', 'black', 'grey']
# x = list(mesh_1d(L, N_ELEM)[0])
# for i in range(4) :
#     phi = v[:, i]
#     plt.plot(x, phi, color=colors[i])
# plt.xlabel("x (m)")
# plt.ylabel("déformée modale")
# plt.title("Fréquences propres : (K - w^2*M)phi = 0")
# plt.show()


### Pré tension / fréquence

# tensions = np.linspace(10, 100, 100)
# tab_K = [ (t/T) * K for t in tensions]
# tab_Kred = [mat[1:-1, 1:-1] for mat in tab_K]
# tab_l = [sp.linalg.eigh(mat, M_red)[0] for mat in tab_Kred]

# for i in range(5) :
#     tab_freq_n = [np.sqrt(la[i]) / (2*np.pi) for la in tab_l]
#     plt.plot(tensions, tab_freq_n, color=colors[i])
# plt.xlabel("tension (N)")
# plt.ylabel("fréquence (Hz)")
# plt.title("Fréquence proportionnelle à la racine de la tension")
#plt.show()