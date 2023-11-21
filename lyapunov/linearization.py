import numpy as np
import matplotlib.pyplot as plt

def create_B(n):
    B = []
    interval_length = 1./n
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
    return B

def test_B_uniform(n):
    interval_length = 1./n
    f = np.ones(shape=(n,1))
    g = np.ones(shape=(n,1))
    results = [np.dot(g.transpose(),np.dot(B[i],f))[0][0] for i in range(n)]
    goal = [(((i+1)*interval_length)**3-(i*interval_length)**3)/6 for i in range(n)]
    plt.plot(results)
    plt.plot(goal,'.')
    plt.show()

def create_C(n):
    C = []
    interval_length = 1./n
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
    return C


def test_C_uniform(n):
    interval_length = 1./n
    f = np.ones(shape=(n,1))
    g = np.ones(shape=(n,1))
    results = [np.dot(C[i],g)[0][0] for i in range(n)]
    goal = [(((i+1)*interval_length)**3-(i*interval_length)**3)/3 for i in range(n)]
    plt.plot(results)
    plt.plot(goal,'.')
    plt.show()


def get_Psi(n, B, C, f,g):
    alpha = np.array([(np.dot(np.dot(g.transpose(),B[i]),f)-np.dot(C[i],g))[0][0] for i in range(n)])
    beta = np.dot(np.diag(f.flatten()), alpha)
    gamma = np.dot(np.dot(np.diag(f.flatten()/n), np.ones((n,n))),beta)
    Dfalpha = np.zeros((n,n))
    Dgalpha = np.zeros((n,n))
    for i in range(n):
        Dfalpha[i,:] = np.dot(g.transpose(), B[i])
        Dgalpha[i,:] = np.dot(f.transpose(), B[i].transpose()) - C[i]
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

def get_beta(n, B, C, f, g):
    alpha = np.array([(np.dot(np.dot(g.transpose(),B[i]),f)-np.dot(C[i],g))[0][0] for i in range(n)])
    beta = np.dot(np.diag(f.flatten()), alpha)
    return beta


class PsiCalculator:

    def __init__(self, n) -> None:
        self.n = n
        self.B = create_B(n)
        self.C = create_C(n)
        # Dimension reduction
        X = np.eye(n,n-1) # maps n-1 vectors to n vectors (linear part)
        for i in range(n-1):
            X[n-1,i] = -1.
        self.DP = np.stack((X,X))

    def computeAt(self, f, g):
        Psif, DPsif = get_Psi(self.n,self.B,self.C,f,g)
        Psig, DPsig = get_Psi(self.n,self.B,self.C,g,f)
        DPsig = DPsig[:, list(range(self.n,2*self.n))+list(range(self.n))]
        return {
            'Psif': Psif,
            'DPsif': DPsif,
            'Psig': Psig,
            'DPsig': DPsig
        }
    
    def computeReducedAt(self, tilde_f, tilde_g):
        f = np.concatenate((tilde_f, [[self.n - np.sum(tilde_f)]]))
        g = np.concatenate((tilde_g, [[self.n - np.sum(tilde_g)]]))
        Psi = self.computeAt(f,g)
        return {
            'Psif': Psi['Psif'][0:self.n-1],
            'DPsif': None,
            'Psig': Psi['Psig'][0:self.n-1],
            'DPsig': None,
        }