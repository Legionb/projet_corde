from etape1 import *

from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from math import floor

# N_ELEM = 330 #changer à 330
# L = 0.33
# h = 0.003  -> sert à quoi ??
# x_pluck = L/5 #sinon 0.03  -> c'est X_ARCH

intervalles_temps = 101
n_free = len(U_red) # pas sur que vrm = DDL
t_0 = 0
t_f = 0.2

XI = 0.0001
N_MODES = 20
T1 = np.sqrt(T / MU) / (2*L) # première période modale

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

w2, phi = eigh(K_red, M_red)
w2=w2[:N_MODES]
phi=phi[:,:N_MODES]
omegas = np.sqrt(w2)
freq = omegas / (2*np.pi)

def rhs_mod(t, y) :
    q, q_point = y[:N_MODES], y[N_MODES:]
    return np.concatenate([q_point, - np.array([w2[i] * q[i] for i in range(len(w2))])]) 

phi_T = np.transpose(phi)
v0 = [0 for _ in U_red]
y0_mod = np.concatenate([phi_T @ M_red @ U_red, v0])

sol_mod = solve_ivp(rhs_mod, (t_0, t_f), y0_mod, method='RK45', rtol=1e-6, atol=1e-10,
                    t_eval= np.linspace(t_0, t_f, intervalles_temps), vectorized=True)
    
for i in range(6) :
    vect = phi @ (sol_mod.y[:N_MODES, 5*i])
    vect_agrandit = np.concatenate([[0], vect, [0]])
    plt.plot(np.linspace(0, L, N_ELEM+1), vect_agrandit, 
                color=colors[i], label=f"t = {5*i}e colonne")

plt.xlabel("x (m)")
plt.ylabel("U(x) (mm)")
plt.title("Intégration base modale sans amortissement")
plt.legend()
plt.savefig("figures/fig3_modale.png")
plt.show()




#if __name__ == "__main__" :
    #integration_temporelle()
    #res_modale()

