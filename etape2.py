from etape1 import *

from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from scipy.io.wavfile import write

intervalles_temps = 101
n_free = len(U_red) # pas sur que vrm = DDL
T1 = np.sqrt(MU / T) * (2*L) # première période modale
t_0 = 0
t_f = 5 #ou 0.2, T1

XI = 0.0001
N_MODES = 20
fs = 44100

## Résolution classique sans amortissement

def integration_temporelle() :
    M_red_inv = np.linalg.inv(M_red)
    A = M_red_inv @ K_red

    def rhs(t, y) :
        u, v = y[:n_free], y[n_free:]
        return np.concatenate([v, - A @ u])

    u0 = U_red
    v0 = [0 for _ in U_red]
    y0 = np.concatenate([u0, v0])
    sol = solve_ivp(rhs, (t_0, t_f), y0, method='RK45', rtol=1e-3, atol=1e-6,
                 t_eval= np.linspace(t_0, t_f, intervalles_temps), vectorized=True)


    for i in range(5):
        vect = sol.y[:n_free, i]
        vect_agrandit = np.concatenate([[0], vect, [0]])
        plt.plot(np.linspace(0, L, N_ELEM+1), vect_agrandit * 1e3, 
                color=colors[i], label=f"t = { (floor(((t_f-t_0) * i / intervalles_temps) * 1e5) * 1e-5) :5} s")
    plt.xlabel("x (m)")
    plt.ylabel("U(x) (mm)")
    plt.title("Intégration temporelle sans amortissement")
    plt.legend()
    plt.savefig("figures/fig4_integration.png")
    plt.show()

## Résolution modale sans amortissement

def res_modale() :
    w2, phi = eigh(K_red, M_red)
    w2=w2[:N_MODES]
    phi=phi[:,:N_MODES]
    omegas = np.sqrt(w2)
    freq = omegas / (2*np.pi)

    def rhs_mod(t, y) :
        q, q_point = y[:N_MODES], y[N_MODES:]
        return np.concatenate([q_point, - np.array([w2[i] * q[i] for i in range(len(w2))])]) 

    phi_T = np.transpose(phi)
    B = phi_T @ M_red @ U_red
    v0 = [0 for _ in B]
    y0_mod = np.concatenate([B, v0])

    sol_mod = solve_ivp(rhs_mod, (t_0, t_f), y0_mod, method='RK45', rtol=1e-6, atol=1e-10,
                        t_eval= np.linspace(t_0, t_f, intervalles_temps), vectorized=True)

    l_int = [0, 13, 25, 38, 50]
    l = [0, 0.125, 0.25, 0.375, 0.50]
    for i in range(len(l)) :
        vect = phi @ (sol_mod.y[:N_MODES, l_int[i]  ])
        vect_agrandit = np.concatenate([[0], vect, [0]])
        plt.plot(np.linspace(0, L, N_ELEM+1), vect_agrandit * 1e3, 
                    color=colors[i], label=f"t = {l[i]} * T1")

    plt.xlabel("x (m)")
    plt.ylabel("U(x) (mm)")
    plt.title("Intégration base modale sans amortissement")
    plt.legend()
    plt.savefig("figures/fig5_modale.png")
    plt.show()

    F_chev = (sol_mod.y[1] - sol_mod.y[0]) / h
    plt.plot(sol_mod.t, F_chev, color='blue')
    plt.xlabel("t (s)")
    plt.ylabel("F_chev (N)")
    plt.title("Force au niveau du chevalet")
    plt.savefig("figures/fig6_fchev.png")
    plt.show()

    signal = (F_chev/np.max(np.abs(F_chev))*0.9*32767).astype(np.int16)
    fichier_son = write("sontropcool.wav", fs, signal)



if __name__ == "__main__" :
    #integration_temporelle()
    res_modale()

