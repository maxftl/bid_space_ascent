import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

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

def L2norm(f):
    n = np.size(f)
    return np.linalg.norm(f)/np.sqrt(n)

def compute_best_response(g, psi_calculator, start_f, n_iterations = 5000, h = 0.1):
    n = np.size(g)
    f = start_f.copy()
    f = f.reshape((n,1))
    gg = g.copy()
    gg = gg.reshape((n,1))
    for _ in range(n_iterations):
        Psi = psi_calculator.computeAt(f,gg)
        f[:,0] += h*Psi['Psif']
    return f[:]


def compute_best_response_sg(g, psi_calculator, start_f, n_iterations = 5000, h = 0.05, max_dist_to_start = np.inf):
    n = np.size(g)
    f = start_f.copy()
    f = f.reshape((n,1))
    gg = g.copy()
    gg = gg.reshape((n,1))
    for _ in range(n_iterations):
        Psi = psi_calculator.computeAt(f,gg)
        f[:,0] = project_L2_simplex(f[:,0] + h*Psi['Psif'])
        if L2norm(f.flatten()-start_f.flatten()) > max_dist_to_start:
            break
    return f

def compute_penalized_best_response(g, psi_calculator, start_f, penalty, h = 0.05, accuracy = 0.01, max_iter=1000):
    n = np.size(g)
    f = start_f.copy()
    f = f.reshape((n,1))
    gg = g.copy()
    gg = gg.reshape((n,1))
    for _ in range(max_iter):
        Psi = psi_calculator.computeAt(f,gg)
        f_new = project_L2_simplex(f.flatten() + h*Psi['Psif'] + h*penalty*(start_f.flatten()-f.flatten()))
        if L2norm(f_new-f[:,0])/h < accuracy:
            f[:,0] = f_new
            break
        else:
            f[:,0] = f_new
    return f


def random_density(n):
    r = np.random.random_sample((n+1,)).tolist()
    r.sort()
    d = np.array([r[i+1]-r[i] for i in range(n)])
    d = d * n/d.sum()
    return d


def compute_utility(n,f,g):
    l = 1./n
    flatf = f.flatten()
    flatg = g.flatten()
    cs_f = np.concatenate(([0],np.cumsum(flatf)))
    cs_g = np.concatenate(([0],np.cumsum(flatg)))
    result = 0.
    bi = 0.
    bip1 = l
    for i in range(n):
        result +=   f[i]*np.power(l,3.) * (
            cs_f[i]*cs_g[i] +
            f[i]*cs_g[i]/2. +
            g[i]*cs_f[i]/2. +
            f[i]*g[i]/3. )
        result -= f[i] * (
            (l*cs_g[i] - g[i]*bi)*(bip1**2 - bi**2)/2 +
            g[i]*(bip1**3-bi**3)/3
        )
        bi += l
        bip1 += l
    return result


def project_L2_simplex(f):
    shape = np.shape(f)
    n = np.size(f)
    pf = projection_simplex_pivot(f.flatten()/n)
    return n*pf.reshape(shape)

# from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


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

class BestResponseQP:

    def __init__(self, g):
        self.n = np.size(g)
        self.interval_length = 1./self.n
        self.g = g
        self.G = self.interval_length * np.concatenate(([0],np.cumsum(self.g.flatten())))
        self.init_qp_matrix()
        self.init_linear_part()

    def init_qp_matrix(self):
        self.Q = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i):
                self.Q[i,j] = self.G[i]*self.interval_length**2 +\
                                (1./2.) * self.g[i] * self.interval_length**3
            self.Q[i,i] = (1./3.)*self.g[i]*self.interval_length**3  + 0.5*self.G[i]*self.interval_length**2

    def init_linear_part(self):
        self.b = np.zeros((self.n))
        for i in range(self.n):
            bi = i*self.interval_length
            bip1 = bi + self.interval_length
            self.b[i] = 0.5*(self.G[i]-bi*self.g[i])*(bip1**2 - bi**2) +\
                        (1./3.)*self.g[i]*(bip1**3 - bi**3)
            

def best_response_in_neighbourhood(f, g, epsilon):
    '''Finds best response strategy f* to g in the neighbourhood
    |f-f*|_{L2} <= epsilon'''
    n = np.size(g)
    assert(n == np.size(f))
    qpmatrices = BestResponseQP(g)
    A = qpmatrices.Q
    c = qpmatrices.b
    #Basic correctness test
    #print(f.T @ (0.5*(A.T + A)) @ f - np.dot(c.T,f))

    Pr = np.eye(n,n-1)
    for i in range(n-1):
        Pr[-1,i] = -1
    APr = 0.5 * Pr.T @ (A+A.T) @ Pr
    cPr = Pr.T @ c
    en = np.zeros((n,))
    en[-1] = 1

    fr = f[:-1]
    gr = g[:-1]
    #Test of reduced problem
    #print("test quadratic part")
    #print(f.T @ A @ f)
    #print(fr.T @ APr @ fr + 2*n*en.T@(0.5*(A.T+A))@Pr@fr + n*n*en.T@A@en)
    #print("test linear part")
    #print(c.T @ f)
    #print(cPr.T @ fr + n*c[-1])

    y = cp.Variable(n-1)
    obj = cp.quad_form(y, APr) +  2*n*en.T@(0.5*(A.T+A))@Pr@y + n*n*en.T @ A @ en - cPr.T @ y - n*c[n-1]
    dist_to_f = cp.quad_form(y-fr, (1./n) * Pr.T@Pr)
    dist_to_f_small = dist_to_f <= epsilon**2
    is_nonneg = y >= 0
    is_density = np.ones((n-1,)).T @ y <= n

    #prob = cp.Problem(cp.Maximize(obj), [y == fr])
    #prob.solve()
    #print(prob.value)

    prob = cp.Problem(cp.Maximize(obj), [dist_to_f_small, is_nonneg, is_density])
    prob.solve()

    best_response = np.array(list(y.value) + [n - np.sum(y.value)])

    return {'utility': prob.value, 'strategy':best_response}
