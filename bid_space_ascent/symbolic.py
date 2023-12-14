import numpy as np
import sympy as sp


def __get_symbolic_density(h, b):
    n = np.size(h)
    cases = [(h[i], b < sp.Rational(i+1,n)) for i in range(n)] + [(0,True)]
    return sp.Piecewise(*cases)

def __get_symbolic_distribution(density, b):
    return sp.integrate(density, (b,0,b))

def __get_symbolic_utility(own_density, own_distribution, other_distribution, b):
    return sp.integrate( (own_distribution-b)*other_distribution*own_density, (b,0,1) )

def generate_symbolic_utilities(density_variables_1, density_variables_2):
    b = sp.Symbol('b')
    density_1 = __get_symbolic_density(density_variables_1, b)
    density_2 = __get_symbolic_density(density_variables_2, b)
    distribution_1 = __get_symbolic_distribution(density_1, b)
    distribution_2 = __get_symbolic_distribution(density_2, b)
    return (
        __get_symbolic_utility(density_1, distribution_1, distribution_2, b),
        __get_symbolic_utility(density_2, distribution_2, distribution_1, b)
    )

def reduce_utility_dimension(utility, own_density_variables, other_density_variables):
    n = np.size(own_density_variables)
    substitution = {
        own_density_variables[-1]: n - np.sum(own_density_variables[:-1]),
        other_density_variables[-1]: n - np.sum(other_density_variables[:-1])
        }
    return utility.subs(substitution)

def centralize_reduced_utility(reduced_utility, own_density_variables, other_density_variables):
    substitution = {
        dv: dv + 2 for dv in own_density_variables[:-1]
    } | {
        dv: dv + 2 for dv in other_density_variables[:-1]
    }
    return reduced_utility.subs(substitution)

def generate_centralized_reduced_utilities(density_variables_1, density_variables_2):
    utility_1, utility_2 = generate_symbolic_utilities(density_variables_1, density_variables_2)
    reduced_utility_1 = reduce_utility_dimension(utility_1, density_variables_1, density_variables_2)
    reduced_utility_2 = reduce_utility_dimension(utility_2, density_variables_2, density_variables_1)
    centered_reduced_utility_1 = centralize_reduced_utility(
        reduced_utility_1, density_variables_1, density_variables_2,
    )
    centered_reduced_utility_2 = centralize_reduced_utility(
        reduced_utility_2, density_variables_2, density_variables_1,
    )
    return (
        centered_reduced_utility_1,
        centered_reduced_utility_2
    )

