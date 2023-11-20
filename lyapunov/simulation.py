import numpy as np
from random import randint
from matplotlib import pyplot as plt
from tqdm import tqdm
from linearization import create_B, create_C, get_Psi

# number of steps of the function
n = 2

edges = np.linspace(start = 0.,stop = 1., num=n+1)

# subdivide the interval in
interval_length = 1./n

# init f and g to play truthfully
f = np.ones(shape=(n,1))
g = np.ones(shape=(n,1))


B = create_B(n)

C = create_C(n)






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
n_rounds = 10000
h = 0.1
gradient_norms = []
for k in tqdm(range(n_rounds)):
    Psif, _ = get_Psi(n,B,C,f,g)
    Psig, _ = get_Psi(n,B,C,g,f)
    gradient_norms.append(np.linalg.norm(Psif))
    f[:,0] += h*Psif
    g[:,0] += h*Psig

print(np.sum(f)/n)
#plt.plot(gradient_norms)
plt.stairs(values=f.flatten(), edges=edges)
plt.show()


