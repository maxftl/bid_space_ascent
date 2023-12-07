import sympy as sp
import numpy as np
import linearization as lin

n = 4

B = [sp.Matrix(b) for b in lin.create_B(n)]
C = [sp.Matrix(c) for c in lin.create_C(n)]

f = sp.symarray('f',n)
g = sp.symarray('g',n)

def get_alpha(n,f,g,B,C):
    alpha = []
    for i in range(n):
        b = B[i]
        c = C[i]
        expr = 0
        for j in range(n):
            for k in range(n):
                expr += g[j]*b[j,k]*f[k]
        for j in range(n):
            expr -= g[j]*c[j]
        alpha.append(expr)
    return alpha

def get_beta(n,f,alpha):
    beta = []
    for i in range(n):
        expr = f[i]*alpha[i]
        beta.append(expr)
    return beta

def get_gamma(n,f,beta):
    beta_sum = 0
    for i in range(n):
        beta_sum += beta[i]
    gamma = []
    for i in range(n):
        expr = f[i]*(1./n)*beta_sum
        gamma.append(expr)
    return gamma

def get_Psi(n, beta, gamma):
    psi = []
    for i in range(n):
        expr = n*(beta[i]-gamma[i])
        psi.append(expr)
    return psi

alphaf = get_alpha(n,f,g,B,C)
betaf = get_beta(n,f,alphaf)
gammaf = get_gamma(n,f,betaf)
Psif = get_Psi(n,betaf,gammaf)

alphag = get_alpha(n,g,f,B,C)
betag = get_beta(n,g,alphag)
gammag = get_gamma(n,g,betag)
Psig = get_Psi(n,betag,gammag)

# Reduce dimension
exprf = 1
for i in range(n-1):
    exprf -= f[i]
exprg = 1
for i in range(n-1):
    exprg -= f[i]
Psif_red = [pf.subs(f[n-1], exprf).subs(g[n-1], exprg) for pf in Psif]
Psig_red = [pg.subs(g[n-1], exprg).subs(f[n-1], exprf) for pg in Psig]

print(alphaf[3])
print()
print(betaf[3])
print()
print(gammaf[3])
print()
print(Psif[3])