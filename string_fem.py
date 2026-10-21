import numpy as np

def mesh_1d(L, n_elem) :
    """
    Paramètres : 
     L = taille corde
     n_elem = nb d'éléments
    Sortie :
     nodes = (array) np.linspace(0.0, L, n_elem + 1)
     mesh = #[[0, 1], [1, 2], ..., [n_elem-1, n_elem]]
    """
    nodes = np.linspace(0.0, L, n_elem + 1)   # coordonnées des noeuds
    n_nodes = len(nodes)
    #connect = np.array([[i, i+1] for i in range(0, n_nodes)])   
    mesh = np.column_stack((np.arange(n_nodes-1), np.arange(1,n_nodes))) 
    return nodes, mesh

def mat_masse_elem(mu, h) :
    """
    Paramètres :
     mu = masse linéique
     h = longueur d'un élément
    Sortie :
     matrice élémentaire de la masse (11) dans la doc
    """
    mat = np.array([[2, 1], [1,2]])
    return (1/6)* mu * h * mat

def mat_raideur_elem(T, h) :
    """
    Paramètres :
     T = tension
     h = longueur d'un élément
    Sortie :
     matrice élémentaire de la raideur (10) dans la doc
    """
    mat = np.array([[1, -1], [-1,1]])
    return (T / h) * mat

def assemblage(mat_elem, n) :
    """
    Paramètres :
     mat_elem = matrice élémentaire à assembler /!\ matrice de la forme [ [a, b], [b, a] ]
     n = nb de répétitions de la matrice élémentaire
    Sortie :
     matrice assemblée n fois à partir de mat_elem
    """
    mat = np.zeros((n+1, n+1))
    a = mat_elem[0, 0]
    b = mat_elem[0, 1]
    for i in range(1, n) :
        mat[i, i] = 2*a
        mat[i+1, i] = b
        mat[i, i+1] = b
    mat[0, 0] = a
    mat[n, n] = a
    mat[0, 1] = b
    mat[1, 0] = b
    return mat



