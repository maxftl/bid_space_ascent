import numpy as np
import sympy as sp


def random_density(n):
    r = np.random.random_sample((n + 1,)).tolist()
    r.sort()
    d = np.array([r[i + 1] - r[i] for i in range(n)])
    d = d * n / d.sum()
    return d


def L2norm(f):
    n = np.size(f)
    return np.linalg.norm(f)/np.sqrt(n)

def project_to_L2_simplex(f):
    shape = np.shape(f)
    n = np.size(f)
    pf = projection_simplex_pivot(f.flatten() / n)
    return n * pf.reshape(shape)


def projection_simplex_pivot(v, z=1, random_state=None):
    rs = np.random.RandomState(random_state)
    n_features = len(v)
    U = np.arange(n_features)
    s = 0
    rho = 0
    while len(U) > 0:
        G = []
        L = []
        k = U[rs.randint(0, len(U))]
        ds = v[k]
        for j in U:
            if v[j] >= v[k]:
                if j != k:
                    ds += v[j]
                    G.append(j)
            elif v[j] < v[k]:
                L.append(j)
        drho = len(G) + 1
        if s + ds - (rho + drho) * v[k] < z:
            s += ds
            rho += drho
            U = L
        else:
            U = G
    theta = (s - z) / float(rho)
    return np.maximum(v - theta, 0)


def get_equilibrium(n):
    assert n % 2 == 0
    equil = np.zeros((n, 1))
    for i in range(int(n / 2)):
        equil[i, 0] = 2
    return equil


def round_poly_coefficients(poly, ndigits):
    dict_representation = poly.as_dict()
    rounded = {
        exponents: round(coeff, ndigits)
        for exponents, coeff in dict_representation.items()
    }
    return sp.Poly.from_dict(rounded, poly.gens)


def compute_gradient(V, variables):
    return np.array([V.diff(var) for var in variables])