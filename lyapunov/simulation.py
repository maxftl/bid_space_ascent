import numpy as np
from random import randint
from matplotlib import pyplot as plt
from tqdm import tqdm

# number of steps of the function
n = 10

# subdivide the interval in
interval_length = 1./n

# init f and g to play truthfully
f = np.ones(shape=(n,1))
g = np.ones(shape=(n,1))

# Vector of the matrices B^i
B = []

for i in range(n):
    Bi = np.zeros(shape=(n,n),dtype=float)
    # entries k < j < i
    for j in range(i):
        for k in range(j):
            Bi[j][k] = interval_length**3
    # entries j=j < i
    for j in range(i):
        Bi[j][j] = (interval_length**3)/2
    #entries k < i
    for k in range(i):
        Bi[i][k] = (interval_length**3)/2
    # entry ii
    Bi[i][i] = (interval_length**3)/6
    B.append(Bi)

def test_B_uniform():
    f = np.ones(shape=(n,1))
    g = np.ones(shape=(n,1))
    results = [np.dot(g.transpose(),np.dot(B[i],f))[0][0] for i in range(n)]
    goal = [(((i+1)*interval_length)**3-(i*interval_length)**3)/6 for i in range(n)]
    plt.plot(results)
    plt.plot(goal,'.')
    plt.show()

#test_B_uniform()
#exit(0)

# Vector of row vectors C^i
C = []
for i in range(n):
    Ci = np.zeros(shape=(1,n),dtype=float)
    for j in range(i):
        Ci[0][j] = (interval_length**3)/2
    bi = i*interval_length
    bip1 = (i+1)*interval_length
    for j in range(i):
        Ci[0][j] = (bip1**2-bi**2)/2 * interval_length
    Ci[0][i] = (bip1**3-bi**3)/3 - bi*(bip1**2-bi**2)/2
    C.append(Ci)


def test_C_uniform():
    f = np.ones(shape=(n,1))
    g = np.ones(shape=(n,1))
    results = [np.dot(C[i],g)[0][0] for i in range(n)]
    goal = [(((i+1)*interval_length)**3-(i*interval_length)**3)/3 for i in range(n)]
    plt.plot(results)
    plt.plot(goal,'.')
    plt.show()

#test_C_uniform()
#exit(0)



def get_Psi(f,g):
    alpha = np.array([(np.dot(np.dot(g.transpose(),B[i]),f)-np.dot(C[i],g))[0][0] for i in range(n)])
    beta = np.dot(np.diag(f.flatten()), alpha)
    gamma = np.dot(np.dot(np.diag(f.flatten()/n), np.ones((n,n))),beta)
    Dfalpha = np.zeros((n,n))
    Dgalpha = np.zeros((n,n))
    for i in range(n):
        Dfalpha[i,:] = np.dot(g.transpose(), B[i])
        Dgalpha[i,:] = np.dot(f.transpose(), B[i].transpose()) - C[i]
    beta = np.dot(np.diag(f.flatten()), alpha)
    Dfbeta = np.diag(alpha) + np.dot(np.diag(f.flatten()), Dfalpha)
    Dgbeta = np.dot(np.diag(f.flatten()),Dgalpha)
    gamma = np.dot(np.dot(np.diag(f.flatten()/n), np.ones((n,n))),beta)
    Dfgamma = np.sum(beta)*np.diag([1/n for i in range(n)]) + np.dot(np.dot(np.diag(f.flatten()/n), np.ones((n,n))), Dfbeta)
    Dggamma = np.dot(np.dot(np.diag(f.flatten()/n), np.ones((n,n))), Dgbeta)
    Psi = n * (beta - gamma)
    DfPsi = n * (Dfbeta - Dfgamma)
    DgPsi = n * (Dgbeta - Dggamma)
    DPsi = np.concatenate((DfPsi,DgPsi), axis = 1)
    return Psi,DPsi


#h = .01
#np.random.rand(4)
#v = np.random.rand(n,1)
#v /= np.linalg.norm(v)
#v -= v.mean()
#f_h = f + h*v
#Psi, DPsi = get_Psi(f,g)
#Psi_h, _ = get_Psi(f_h,g)
#plt.plot((Psi_h-Psi)/h)
#plt.plot(np.dot(DPsi,np.concatenate((v,np.zeros_like(v)))),'.')
#plt.show()
#exit(0)

f = 0.1 * np.ones((n,1))
f[0:1,0] += np.ones((1,))
f = f * n/f.sum()
g = 0.1 * np.ones((n,1))
g[0:1,0] = np.ones((1,))
g = g * n/g.sum()
n_rounds = 1000
h = 0.1
gradient_norms = []
for k in tqdm(range(n_rounds)):
    Psif, _ = get_Psi(f,g)
    Psig, _ = get_Psi(g,f)
    gradient_norms.append(np.linalg.norm(Psif))
    f[:,0] += h*Psif
    g[:,0] += h*Psig

print(np.sum(f)/n)
#plt.plot(gradient_norms)
plt.plot(f)
plt.show()