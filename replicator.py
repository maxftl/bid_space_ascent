import ot
import ot.plot
import ot.bregman
import ot.lp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from IPython.display import display, clear_output


n_values = 50
n_bids = 50
values = np.linspace(0,1,n_values)
bids = np.linspace(0,1,n_bids)

#n_values = 2
#n_bids = 2
#values = np.array([0.25, 0.75])
#bids = np.array([0.25, 0.75])



def compute_bid_distribution(X):
    bid_density = np.sum(X, axis = 0)
    return np.cumsum(bid_density) - 0.5 * bid_density

def utility_matrix(bid_distribution):
    A = values[:,None]*bid_distribution
    #print(f'A: {A}')
    B = np.transpose(np.tile((bid_distribution*bids)[:,None], (1,n_bids)))
    #print(f'B: {B}')
    return A-B

def weighted_utility(X, utility_matrix):
    return np.sum(X * utility_matrix, axis = 1)

def update_factor(utility_matrix, weighted_utility):
    return (1./n_values)*utility_matrix - weighted_utility[:,None]

def update_x(X, update_factor, epsilon):
    Y = X + epsilon*update_factor
    Y = np.maximum(Y, np.zeros_like(Y))
    rowsums = np.sum(Y, axis=1)
    scaling = 1./(n_values*rowsums)
    return scaling[:,None] * Y


## Constant init
#X_1 = np.zeros((n_values,n_bids))
#X_2 = np.zeros((n_values,n_bids))

#for i in range(n_values):
 #   X_1[i,0] = 1./n_values
  #  X_2[i,n_bids-1] = 1./n_values


## Uniform init
X_1 = np.ones((n_values,n_bids))/(n_values*n_bids)
X_2 = np.ones((n_values,n_bids))/(n_values*n_bids)

# Random init
#X_1 = np.random.rand(n_values,n_bids)
#rowsums = np.sum(X_1, axis=1)
#scaling = 1./(n_values*rowsums)
#X_1 = scaling[:,None] * X_1

#X_2 = np.random.rand(n_values,n_bids)
#rowsums = np.sum(X_2, axis=1)
#scaling = 1./(n_values*rowsums)
#X_2 = scaling[:,None] * X_2

## bad strategy init
#X_1 = np.zeros((n_values,n_bids))
#X_2 = np.zeros((n_values,n_bids))
#for i in range(int(n_values/2)):
#    X_1[i,n_bids-1] = 1/n_values
#    X_2[i,0] = 1/n_values
#for i in range(int(n_values/2),n_values):
#    X_2[i,n_bids-1] = 1/n_values
#    X_1[i,0] = 1/n_values


## Main loop
epsilon = 0.001
n_iterations = 10000

figure, ax = plt.subplots(1,2)
ax[0].matshow(X_1)
ax[1].matshow(X_2)
#plt.show()


for i in range(n_iterations):
    bid_distribution_1 = compute_bid_distribution(X_1)
    bid_distribution_2 = compute_bid_distribution(X_2)
    utility_matrix_1 = utility_matrix(bid_distribution_2)
    #print(f'utility_matrix: {utility_matrix_1}')
    utility_matrix_2 = utility_matrix(bid_distribution_1)
    weighted_utility_1 = weighted_utility(X_1, utility_matrix_1)
    #print(f'weighted_utility: {weighted_utility_1}')
    weighted_utility_2 = weighted_utility(X_2, utility_matrix_2)
    update_factor_1 = update_factor(utility_matrix_1, weighted_utility_1)
    #print(f'update factor: {update_factor_1}')
    update_factor_2 = update_factor(utility_matrix_2, weighted_utility_2)
    X_1 = update_x(X_1, update_factor_1, epsilon)
    X_2 = update_x(X_2, update_factor_2, epsilon)
    if i % 50 == 0:
        ax[0].matshow(X_1)
        ax[1].matshow(X_2)
        plt.pause(0.1)

plt.show()


