import sympy as sp
import numpy as np
from bid_space_ascent.dynamics import UtilityCalculator
import SumOfSquares as sos
from bid_space_ascent.lyapunov_utils import (
    add_lyapunov_constraints,
    get_parameter_values,
    set_minimize_max_degree_objective,
)
from bid_space_ascent.utils import (
    compute_gradient,
    SympyNumpyEncoder,
    round_poly_coefficients,
)
import logging
import json

logger = logging.getLogger(__name__)


def get_reduced_equilibrium(n):
    equilibrium = np.zeros((n - 1,))
    for i in range(int(n / 2)):
        equilibrium[i] = 2
    return equilibrium


def to_uncentered_unreduced(strategy):
    n = np.size(strategy) + 1
    uncentered_strategy = strategy + get_reduced_equilibrium(n)
    unreduced_strategy = np.concatenate(
        [uncentered_strategy, [n - np.sum(uncentered_strategy)]]
    )
    return unreduced_strategy


def lower_bound_constraints(strategy_centered, other_strategy_centered, reverse=False):
    n = np.size(strategy_centered) + 1
    original_equilibrium = get_reduced_equilibrium(n)
    g = strategy_centered + original_equilibrium
    if not reverse:
        H = [
            [
                strategy_centered[i].subs(
                    [(strategy_centered[j], -original_equilibrium[j])]
                )
                for i in range(n - 1)
            ]
            + other_strategy_centered.tolist()
            for j in range(n - 1)
        ]
    else:
        H = [
            other_strategy_centered.tolist()
            + [
                strategy_centered[i].subs(
                    [(strategy_centered[j], -original_equilibrium[j])]
                )
                for i in range(n - 1)
            ]
            for j in range(n - 1)
        ]
    return (g.tolist(), H)


def upper_bound_constraint(strategy_centered, other_strategy_centered, reverse=False):
    g = [-np.sum(strategy_centered)]
    if not reverse:
        H = [
            strategy_centered[:-1].tolist()
            + [0.0 - np.sum(strategy_centered[:-1])]
            + other_strategy_centered.tolist()
        ]
    else:
        H = [
            other_strategy_centered.tolist()
            + strategy_centered[:-1].tolist()
            + [0.0 - np.sum(strategy_centered[:-1])]
        ]
    return (g, H)


def compute_lyapunov(n, filename, max_degree=2, objective_function_setter = None):
    assert n >= 2
    assert n % 2 == 0

    # These are the transformed variables f and g such that
    # the equilibrium is at (0,0)
    f_centered = sp.symarray("f", shape=(n - 1,))
    g_centered = sp.symarray("g", shape=(n - 1,))
    original_equilibrium = np.zeros_like(f_centered)
    for i in range(int(n / 2)):
        original_equilibrium[i] = 2

    lb_f, lb_H_f = lower_bound_constraints(f_centered, g_centered)
    lb_g, lb_H_g = lower_bound_constraints(g_centered, f_centered, reverse=True)
    ub_f, ub_H_f = upper_bound_constraint(f_centered, g_centered)
    ub_g, ub_H_g = upper_bound_constraint(g_centered, f_centered, reverse=True)
    constraints = lb_f + lb_g + ub_f + ub_g
    H = lb_H_f + lb_H_g + ub_H_f + ub_H_g

    original_f = to_uncentered_unreduced(f_centered).reshape((n, 1))
    original_g = to_uncentered_unreduced(g_centered).reshape((n, 1))

    utility_calculator = UtilityCalculator(n)
    utility_f = utility_calculator.getUtility(original_f, original_g)[0]
    utility_g = utility_calculator.getUtility(original_g, original_f)[0]

    grad_utility_f = compute_gradient(utility_f, f_centered)
    # grad_utility_f_test = utility_calculator.getReducedUtilityGradient(
    #     f_centered + original_equilibrium,
    #     g_centered + original_equilibrium
    #     )
    # print(grad_utility_f - grad_utility_f_test)
    grad_utility_g = compute_gradient(utility_g, g_centered)
    # grad_utility_g_test = utility_calculator.getReducedUtilityGradient(
    #     g_centered + original_equilibrium,
    #     f_centered + original_equilibrium
    #     )
    # print(grad_utility_g - grad_utility_g_test)

    vectorfield = np.concatenate([grad_utility_f, grad_utility_g])

    problem = sos.SOSProblem()
    lyapunov_info = add_lyapunov_constraints(
        problem=problem,
        max_degree=max_degree,
        variables=np.concatenate([f_centered, g_centered]).tolist(),
        f=vectorfield.tolist(),
        inequality_constraints=constraints,
        H=H,
    )
    logger.info("Start solving")
    if objective_function_setter:
        objective_function_setter(
            problem, lyapunov_info, np.concatenate([f_centered, g_centered]).tolist()
        )
    problem.solve()
    logger.info("Found solution")
    lyapunov_coefficients = lyapunov_info["lyapunov_parametrization"]["parameters"]
    lyapunov_coeff_values = get_parameter_values(lyapunov_coefficients, problem)
    lyapunov_info["lyapunov_parametrization"]["parameter_values"] = lyapunov_coeff_values
    neg_inner_product_coefficients = lyapunov_info["interior_constraint"]["parameters"]
    neg_inner_product_coeff_values = get_parameter_values(neg_inner_product_coefficients, problem)
    lyapunov_info["interior_constraint"]["parameter_values"] = neg_inner_product_coeff_values
    lyapunov_coeff_substitution = list(
        zip(
            lyapunov_coefficients.flatten().tolist(),
            lyapunov_coeff_values.flatten().tolist(),
        )
    )
    lyapunov_function = (
        lyapunov_info["lyapunov_function"].subs(lyapunov_coeff_substitution).simplify()
    )
    lyapunov_info["solution"] = lyapunov_function
    lyapunov_info["n"] = n

    with open(filename, "w") as outfile:
        json.dump(lyapunov_info, outfile, cls=SympyNumpyEncoder)


def load_lyapunov(filename, ndigits=None):
    with open(filename) as file:
        data = json.load(file)
        n = data["n"]
        f_symbols = [sp.Symbol(f"f_{i}") for i in range(n - 1)]
        g_symbols = [sp.Symbol(f"g_{i}") for i in range(n - 1)]
        locals = {str(fsymbol): fsymbol for fsymbol in f_symbols} | {
            str(gsymbol): gsymbol for gsymbol in g_symbols
        }
        lyapunov_function = sp.sympify(data["solution"], locals=locals)
        if ndigits is not None:
            lyapunov_function = round_poly_coefficients(
                sp.Poly(lyapunov_function), ndigits
            ).as_expr()
        return (lyapunov_function, f_symbols, g_symbols)


if __name__ == "__main__":
    compute_lyapunov(2, "results/test_2.json")
