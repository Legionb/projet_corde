import numpy as np
from string_fem import *
import matplotlib.pyplot as plt
import scipy as sp

L = 0.33        # longueur vibrante [m]
MU = 3.1e-3     # masse linéique [kg/m]
T = 51.9        # tension corde sol [N]
N_ELEM = 330    # nombre d'éléments : multiple de 11 -> h = 1 mm, archet sur le noeud 30
X_ARCH = 0.03   # abscisse archet [m] -> L/11

h = L / N_ELEM  # longueur d'un élément
colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'cyan', 'yellow', 'magenta', 'black', 'grey']

m = mat_masse_elem(MU, h)
M = assemblage(m, N_ELEM)
M_red = M[1:-1, 1:-1]

k = mat_raideur_elem(T, h)
K = assemblage(k, N_ELEM)
K_red = K[1:-1, 1:-1]

F = np.zeros(N_ELEM+1)
F[round(X_ARCH*N_ELEM / L)] = 1
F_red = F[1:-1]

## Résolution statique

U_red = np.linalg.solve(K_red, F_red)
U = [0 for _ in range(N_ELEM+1)]
for i in range(1, N_ELEM) :
    U[i] = U_red[i-1]

x = list(mesh_1d(L, N_ELEM)[0])

if __name__ == "__main__" :
    # solution exacte (triangulaire) pour vérifier
    U_exact = [ (1/(T*L)) * xi * (L - X_ARCH) if xi <= X_ARCH else (1/(T*L)) * X_ARCH * (L - xi) for xi in x]
    print("erreur max statique :", max(abs(U[i] - U_exact[i]) for i in range(N_ELEM+1)), "m")

    U_mm = [u * 1e3 for u in U]
    U_exact_mm = [u * 1e3 for u in U_exact]
    plt.figure()
    plt.plot(x, U_mm, label="EF")
    plt.xlabel("x (m)")
    plt.ylabel("U(x) (mm)")
    plt.title("Résolution statique : KU = F")
    plt.legend()
    plt.savefig("figures/fig1_statique.png")

    ## Fréquences propres

    l, v_red = sp.linalg.eigh(K_red, M_red)
    omegas = np.sqrt(l)
    frequences = omegas / (2*np.pi)

    # vérification avec f_n = n/(2L) * sqrt(T/MU)
    for i in range(5) :
        f_exact = (i+1) / (2*L) * np.sqrt(T/MU)
        print("mode", i+1, ": EF =", frequences[i], "Hz ; exact =", f_exact, "Hz")

    v = np.zeros((len(v_red)+2, len(v_red[0])))
    for i in range(1, len(v)-1) :
        v[i] = v_red[i-1]

    plt.figure()
    for i in range(4) :
        phi = v[:, i]
        plt.plot(x, phi, color=colors[i], label="mode " + str(i+1))
    plt.xlabel("x (m)")
    plt.ylabel("Déformée modale (mm)")
    plt.title("Premières déformées modales : (K - w^2*M)phi = 0")
    plt.legend()
    plt.savefig("figures/fig2_modes.png")

    ### Pré tension / fréquence

    tensions = np.linspace(0.25*T, 2*T, 50)
    # K est proportionnelle à T : on multiplie par t/T, on ne ré-assemble pas
    # (on ne calcule que les 5 premiers modes avec subset_by_index)
    tab_l = [sp.linalg.eigh((t/T) * K_red, M_red)[0][:5] for t in tensions]

    plt.figure()
    for i in range(5) :
        tab_freq_n = [np.sqrt(la[i]) / (2*np.pi) for la in tab_l]
        plt.plot(tensions, tab_freq_n, color=colors[i])
    plt.xlabel("Tension (N)")
    plt.ylabel("Fréquence (Hz)")
    plt.title("Fréquence proportionnelle à la racine de la tension")
    plt.savefig("figures/fig3_tension.png")
    plt.show()