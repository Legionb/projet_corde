import numpy as np


def mesh_1d(L, N_ELEM):
    x = np.linspace(0, L, N_ELEM + 1)
    connect = np.column_stack((np.arange(N_ELEM), np.arange(1, N_ELEM + 1)))
    return x, connect


def mat_masse_elem(mu, h):
    """Matrice de masse élémentaire :"""
    return mu * h / 6 * np.array([[2, 1], [1, 2]])


def mat_raideur_elem(T, h):
    """Matrice de raideur élémentaire :"""
    return T / h * np.array([[1, -1], [-1, 1]])


def assemblage(mat_elem, N_ELEM):
    """Assemble la matrice globale (N_ELEM+1) x (N_ELEM+1)"""
    A = np.zeros((N_ELEM + 1, N_ELEM + 1))
    for e in range(N_ELEM):
        A[e:e+2, e:e+2] += mat_elem
    return A
