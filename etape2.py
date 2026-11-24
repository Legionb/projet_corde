from etape1 import *
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt


# Paramètres
N_ELEM = 110
X_PLUCK = L / 5
U_PLUCK = 1e-3
fs = 44100
N_MODES = 20
XI = 0.001
t_0 = 0
t_f = 1
T1 = 2 * L * np.sqrt(MU / T)


# Maillage et matrices
h = L / N_ELEM
x, connect = mesh_1d(L, N_ELEM)

m = mat_masse_elem(MU, h)
M = assemblage(m, N_ELEM)
M_red = M[1:-1, 1:-1]

k = mat_raideur_elem(T, h)
K = assemblage(k, N_ELEM)
K_red = K[1:-1, 1:-1]

n_free = len(M_red)

# Déplacement initial : corde pincée
i_pluck = round(X_PLUCK / h)
F = np.zeros(N_ELEM + 1)
F[i_pluck] = 1
F_red = F[1:-1]
U_red = np.linalg.solve(K_red, F_red)
U0_red = U_red * (U_PLUCK / U_red[i_pluck - 1])
V0_red = np.zeros(n_free)

### Résolution et figures

def integration_temporelle():
    M_red_inv = np.linalg.inv(M_red)
    A = M_red_inv @ K_red
    def rhs(t, y):
        u = y[:n_free]
        v = y[n_free:]
        return np.concatenate([v, -A @ u])
    y0 = np.concatenate([U0_red, V0_red])
    temps = np.linspace(0, 0.02, int(0.02 * fs) + 1)
    sol = solve_ivp(
        rhs,
        (0, 0.02),
        y0,
        method='RK45',
        rtol=1e-3,
        atol=1e-6,
        t_eval=temps
    )
    indices = np.linspace(0, len(sol.t) - 1, 5, dtype=int)
    plt.figure()
    for i in range(len(indices)):
        vect = sol.y[:n_free, indices[i]]
        vect = np.concatenate([[0], vect, [0]])
        plt.plot(
            x,
            vect * 1e3,
            color=colors[i],
            label=f"t = {sol.t[indices[i]]:.3f} s"
        )
    plt.xlabel("Position x (m)")
    plt.ylabel("Déplacement (mm)")
    plt.title("Intégration temporelle sans amortissement")
    plt.legend()
    plt.savefig("figures/fig4_integration_temporelle.png")
    plt.close()


def res_modale():

    w2, phi = eigh(K_red, M_red)
    w2 = w2[:N_MODES]
    phi = phi[:, :N_MODES]
    omegas = np.sqrt(w2)

    def rhs_mod(t, y):
        q = y[:N_MODES]
        q_point = y[N_MODES:]
        return np.concatenate([q_point, -w2 * q])

    B = phi.T @ M_red @ U0_red
    v0 = np.zeros(N_MODES)
    y0_mod = np.concatenate([B, v0])
    temps = np.linspace(t_0, t_f, int((t_f - t_0) * fs) + 1)
    sol_mod = solve_ivp(
        rhs_mod,
        (t_0, t_f),
        y0_mod,
        method='RK45',
        rtol=1e-6,
        atol=1e-10,
        t_eval=temps
    )
    U_mod = phi @ sol_mod.y[:N_MODES, :]
    return U_mod, sol_mod.t, phi, w2, B


def figure_modes_sans_amortissement(U_mod, temps):
    instants = [0, 0.125, 0.25, 0.375, 0.50]
    plt.figure()
    for i in range(len(instants)):
        temps_voulu = instants[i] * T1
        indice = np.argmin(abs(temps - temps_voulu))
        vect = U_mod[:, indice]
        vect = np.concatenate([[0], vect, [0]])
        plt.plot(
            x,
            vect * 1e3,
            color=colors[i],
            label=f"t = {instants[i]} T1"
        )
    plt.xlabel("Position x (m)")
    plt.ylabel("Déplacement (mm)")
    plt.title("Corde sans amortissement")
    plt.legend()
    plt.savefig("figures/fig5_modes_sans_amortissement.png")
    plt.close()


def res_modale_amortie(phi, w2, B, temps):
    omegas = np.sqrt(w2)
    def rhs_mod_amorti(t, y):
        q = y[:N_MODES]
        q_point = y[N_MODES:]
        return np.concatenate([
            q_point,
            -2 * XI * omegas * q_point - w2 * q
        ])
    v0 = np.zeros(N_MODES)
    y0_mod = np.concatenate([B, v0])
    sol_amorti = solve_ivp(
        rhs_mod_amorti,
        (t_0, t_f),
        y0_mod,
        method='RK45',
        rtol=1e-6,
        atol=1e-10,
        t_eval=temps
    )
    U_mod_amorti = phi @ sol_amorti.y[:N_MODES, :]
    return U_mod_amorti


def figure_modes_avec_amortissement(U_mod_amorti, temps):
    instants = [0, 0.125, 0.25, 0.375, 0.50]
    plt.figure()
    for i in range(len(instants)):
        temps_voulu = instants[i] * T1
        indice = np.argmin(abs(temps - temps_voulu))
        vect = U_mod_amorti[:, indice]
        vect = np.concatenate([[0], vect, [0]])
        plt.plot(
            x,
            vect * 1e3,
            color=colors[i],
            label=f"t = {instants[i]} T1"
        )
    plt.xlabel("Position x (m)")
    plt.ylabel("Déplacement (mm)")
    plt.title("Corde avec amortissement")
    plt.legend()
    plt.savefig("figures/fig6_modes_avec_amortissement.png")
    plt.close()


def figure_force_chevalet(U_mod, U_mod_amortie, temps):
    F_chev = T * U_mod[0, :] / h
    F_chev_amortie = T * U_mod_amortie[0, :] / h
    plt.figure()
    plt.plot(
        temps,
        F_chev,
        label="Sans amortissement"
    )
    plt.plot(
        temps,
        F_chev_amortie,
        label="Avec amortissement"
    )
    plt.xlim(0, 0.02)
    plt.xlabel("Temps (s)")
    plt.ylabel("Force (N)")
    plt.title("Force transmise au chevalet")
    plt.legend()
    plt.savefig("figures/fig7_force_chevalet.png")
    plt.close()
    return F_chev_amortie


def figure_deplacement_pincement(U_mod, U_mod_amortie, temps):
    U_pincement = U_mod[i_pluck - 1, :]
    U_pincement_amortie = U_mod_amortie[i_pluck - 1, :]
    plt.figure()
    plt.plot(
        temps,
        U_pincement * 1e3,
        label="Sans amortissement"
    )
    plt.plot(
        temps,
        U_pincement_amortie * 1e3,
        label="Avec amortissement"
    )
    plt.xlim(0, 0.02)
    plt.xlabel("Temps (s)")
    plt.ylabel("Déplacement (mm)")
    plt.title("Déplacement au point de pincement")
    plt.legend()
    plt.savefig("figures/fig8_deplacement_pincement.png")
    plt.close()


### Programme principal


if __name__ == "__main__":

    integration_temporelle()

    U_mod, temps, phi, w2, B = res_modale()
    figure_modes_sans_amortissement(U_mod, temps)

    U_mod_amortie = res_modale_amortie(phi, w2, B, temps)
    figure_modes_avec_amortissement(U_mod_amortie, temps)

    F_chev_amortie = figure_force_chevalet(U_mod, U_mod_amortie, temps)
    figure_deplacement_pincement(U_mod, U_mod_amortie, temps)

    # Création du fichier audio
    signal = (
        F_chev_amortie /
        np.max(np.abs(F_chev_amortie)) *0.9*32767
    ).astype(np.int16)
    write("pincement.wav", fs, signal)
