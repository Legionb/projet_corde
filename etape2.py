from etape1_v2 import *

from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
XI = 0.0001
n_free = len(U_red)

M_inv = np.linalg.inv(M_red)
def rhs(t, y) :
    u, v = y[:n_free], y[n_free:]
    return np.concatenate([v, - M_inv @ K_red @ u]) #inversion de M plus rapide possible

u0 = U_red
v0 = [0 for _ in U_red]
y0 = np.concatenate([U_red, v0])
# print()

#t0 = time.perf_counter()
#sol = solve_ivp(rhs, (0, 0.2), y0, method='RK45', rtol=1e-6, atol=1e-10, t_eval= np.linspace(0, 0.2, 101), vectorized=True)
#t1 = time.perf_counter()

#print(f"Temps passé dans solve_ivp = {t1 - t0}")

# for i in range(5):
#     plt.plot(sol.t, sol.y[i, :], color=colors[i])
# plt.plot(sol.t, sol.y[10, :], color=colors[1])
# plt.plot(sol.t, sol.y[11, :], color=colors[2])

# for t in range(1, 100, 10) :
#     plt.plot(np.linspace(0, 0.33, N_ELEM-1), sol.y[:n_free, t]*1e-3, color=colors[t // 10])

# plt.ylabel("U(x) (mm)")
# plt.xlabel("x (m)")
# plt.title("Résolution homogène")
# plt.show()

## Résolution homogène base modale, avec et sans amortissement
N_MODES=20
T1 = 0.005 # première période modale
w2, phi = eigh(K_red, M_red)
w2=w2[:N_MODES]
phi=phi[:,:N_MODES]
omegas = np.sqrt(w2)
freq = omegas / (2*np.pi)

def rhs_mod(t, y) :
    q, q_point = y[:N_MODES], y[N_MODES:]
    return np.concatenate([q_point, - np.array([w2[i] * q[i] for i in range(len(w2))])]) #-2* xi * omegas * q_point - w2 * q

# def rhs_mod_amo(t, y) :
#     q, q_point = y[:n_free], y[n_free:]
#     return np.concatenate([q_point, -2* XI * omegas * q_point - np.array([w2[i] * q[i] for i in range(len(w2))])]) #-2* xi * omegas * q_point - w2 * q

phi_T = np.transpose(phi)
y0_mod = np.concatenate([phi_T @ M_red @ u0, v0])

sol_mod = solve_ivp(rhs_mod, (0, 0.01), y0_mod, method='RK45', rtol=1e-6, atol=1e-10,
                    t_eval= np.linspace(0, 0.01, 101), vectorized=True)

# sol_mod_amo = solve_ivp(rhs_mod, (0, 0.005), y0_mod, method='RK45', rtol=1e-6, atol=1e-10,
#                     t_eval= np.linspace(0, 0.005, 101), vectorized=True)

# for k in (1, 3) :
#     plt.plot(np.linspace(0, 0.33, N_ELEM-1), phi @ (sol_mod.y[:n_free, k*T1]*1e-3), color=colors[k // 1])

plt.plot(np.linspace(0, 0.33, N_ELEM-1), phi @ (sol_mod.y[:N_MODES, 1]*1e-3), color=colors[1])
plt.plot(np.linspace(0, 0.33, N_ELEM-1), phi @ (sol_mod.y[:N_MODES, 10]*1e-3), color=colors[2])
plt.plot(np.linspace(0, 0.33, N_ELEM-1), phi @ (sol_mod.y[:N_MODES, 100]*1e-3), color=colors[3])



plt.ylabel("U(x) (mm)")
plt.xlabel("x (m)")
plt.title("Résolution homogène base modale")
plt.show()


# plt.plot(np.linspace(0,1,328), (phi@ (sol_mod.y[:n_free, 1]*1e-3)), color=colors[1])
# plt.ylabel("U1(t) (mm)")
# plt.xlabel("t (s)")
# plt.show()