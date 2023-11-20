import ot
import ot.plot
import ot.bregman
import ot.lp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from IPython.display import display, clear_output

n_values = 500
n_bids = 500

values = np.linspace(0,1,n_values)
bids = np.linspace(0,1,n_bids)
my_density = np.ones((n_bids,))/n_bids
opponent_density = np.ones((n_bids,))/n_bids
value_density = np.ones((n_values,))/n_values

def cumulative_bid_distribution(bid_density):
    return np.cumsum(bid_density) - 0.5 * bid_density

def utility_matrix(opponent_density, values, bids):
    my_winning_probability = cumulative_bid_distribution(opponent_density)
    M = (values[:,None] - bids) * my_winning_probability
    return M

def compute_gradient(value_density, my_density, utility_matrix):
    M = -utility_matrix
    plan, log = ot.emd(value_density, my_density, M, log = True)
    return -log['v'], plan

def vec2simplexV1(vecX, l=1.):
    m = vecX.size
    vecS = np.sort(vecX)[::-1]
    vecC = np.cumsum(vecS) - l
    vecH = vecS - vecC / (np.arange(m) + 1)
    r = np.max(np.where(vecH>0)[0])
    t = vecC[r] / (r + 1)
    return np.maximum(0, vecX - t)


density_1 = vec2simplexV1(np.random.uniform(0.,1.,(n_bids,)))
density_2 = vec2simplexV1(np.random.uniform(0.,1.,(n_bids,)))
step_size = .01
n_iterations = 3000

figure, ax = plt.subplots(1,2)
ax[0].plot(density_1)
ax[1].plot(density_2)
plt.show(block=False)

for i in tqdm(range(n_iterations)):
    utility_1 = utility_matrix(density_2, values, bids)
    utility_2 = utility_matrix(density_1, values, bids)
    gradient_1, _ = compute_gradient(value_density, density_1, utility_1)
    gradient_2, _ = compute_gradient(value_density, density_2, utility_2)
    density_1 = vec2simplexV1(density_1 + 10.*(1./(i+1.))*gradient_1)
    density_2 = vec2simplexV1(density_2 + 10*(1./(i+1.))*gradient_2)

    if i % 100 == 0:
        ax[0].clear()
        ax[0].plot(density_1)
        ax[1].clear()
        ax[1].plot(density_2)
        plt.pause(0.1)


plt.show()
