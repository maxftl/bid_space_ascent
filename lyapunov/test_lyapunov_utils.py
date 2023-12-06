import sympy as sp
import numpy as np
from lyapunov_utils import *
import SumOfSquares as sos
from sympy.polys.monomials import itermonomials
from linearization import *
import json


def test_example_from_paper(): 
    x1, x2 = sp.symbols('x1 x2')
    max_degree = 2

    problem = sos.SOSProblem()

    variables = [x1, x2]
    n = len(variables)

    f = [-x1**2, 0]

    g1 = x1 - x2**2
    H1 = [x2**2, x2]

    g2 = 1 - x1
    H2 = [1, x2]

    H = [H1, H2]

    inequality_constraints = [g1, g2]

    monomials = list(itermonomials(variables, max_degrees=max_degree))

    V_parametrization, V_parameters = generate_parametrized_polynomials(n+1, monomials, 'sigma')
    V = np.dot(np.concatenate([[1],inequality_constraints]),V_parametrization)
    gradV = [V.diff(x) for x in variables]

    add_interior_constraint(
        problem,
        gradV,
        f,
        inequality_constraints,
        variables,
        monomials,
        'chi'
    )

    for constraint_index in range(2):
        add_tight_inequality_constraint(
            problem,
            gradV,
            inequality_constraints,
            constraint_index,
            H[constraint_index],
            variables,
            monomials,
            'phi'
        )

    problem.solve()
    V_parameter_values = [(sigma,problem.sym_to_var(sigma).value) for sigma in V_parameters.flatten().tolist()]
    V_parameter_values = [(sigma, val) for sigma, val in V_parameter_values if val is not None]
    Vresult = V.subs(V_parameter_values).simplify()
    print(Vresult)


def test_1D_example():
    problem = sos.SOSProblem()

    x = sp.symbols('x')
    max_degree = 4
    f = [-x]

    variables = [x]
    n = len(variables)
    monomials = list(itermonomials(variables, max_degrees=max_degree))
    inequality_constraints = []
    
    V_parametrization, V_parameters = generate_parametrized_polynomials(1, monomials, 'sigma')
    V = V_parametrization[0]
    gradV = [V.diff(x) for x in variables]

    add_interior_constraint(
        problem,
        gradV,
        f,
        inequality_constraints,
        variables,
        monomials,
        'chi'
    )

    problem.set_objective('max', problem.sym_to_var(V_parameters[0,2]))
    problem.solve()
    V_parameter_values = [(sigma,problem.sym_to_var(sigma).value) for sigma in V_parameters.flatten().tolist()]
    print(V_parameter_values)
    print(V)

def test_1D_direct():
    problem = sos.SOSProblem()
    

    x = sp.symbols('x')
    f = -x**2
    g = [x]
    extended_g = [1]+g
    H = [0]

    Vpolys, Vparameters = generate_parametrized_polynomials(2, [1,x,x**2,x**3,x**4],'sigma')
    for poly in Vpolys:
        problem.add_sos_constraint(poly, [x])
    V = np.dot(Vpolys, extended_g)
    
    dV = V.diff(x)
    add_interior_constraint(
        problem=problem,
        gradV=dV,
        f=f,
        inequality_constraints=g,
        variables=[x],
        monomials=[1,x,x**2],
        parameter_prefix='chi'
    )
    add_tight_inequality_constraint(
        problem=problem,
        gradV=dV,
        inequality_constraints=g,
        constraint_index=0,
        H=H,
        variables=[x],
        monomials=[1,x,x**2],
        parameter_prefix = 'phi'
    )
    problem.solve()
    values = [(param, problem.sym_to_var(param).value) for param in Vparameters.flatten()]
    print(V.subs(values))


def test_add_lyapunov_constraints():
    x1, x2 = sp.symbols('x1 x2')
    max_degree = 4

    problem = sos.SOSProblem()

    variables = [x1, x2]
    n = len(variables)

    f = [-x1**2, 0]

    g1 = x1 - x2**2
    H1 = [x2**2, x2]

    g2 = 1 - x1
    H2 = [1, x2]

    constraints = [g1, g2]

    add_lyapunov_constraints(
        problem=problem,
        max_degree=max_degree,
        variables=variables,
        f=f,
        inequality_constraints=constraints,
        H = [H1,H2]
    )

    problem.solve()



def to_uncentered_unreduced(strategy):
        uncentered_strategy = strategy + original_equilibrium
        unreduced_strategy  = np.concatenate([uncentered_strategy, [n-np.sum(uncentered_strategy)]])
        return unreduced_strategy

def lower_bound_constraints(strategy_centered, other_strategy_centered, reverse = False):
        g = strategy_centered + original_equilibrium
        if not reverse:
            H = [
                [strategy_centered[i].subs([(strategy_centered[j], -original_equilibrium[j])]) for i in range(n-1)] + other_strategy_centered.tolist()
                for j in range(n-1)
            ]
        else:
             H = [
                other_strategy_centered.tolist() + [strategy_centered[i].subs([(strategy_centered[j], -original_equilibrium[j])]) for i in range(n-1)]
                for j in range(n-1)
            ]
        return (g.tolist(),H)

def upper_bound_constraint(strategy_centered, other_strategy_centered, reverse = False):
        g = [-np.sum(strategy_centered)]
        if not reverse:
            H = [
                strategy_centered[:-1].tolist() + [0.-np.sum(strategy_centered[:-1])] + other_strategy_centered.tolist()
            ]
        else:
             H = [
                other_strategy_centered.tolist() + strategy_centered[:-1].tolist() + [0.-np.sum(strategy_centered[:-1])]
            ]
        return (g,H)

if __name__ == '__main__':
    #test_1D_direct()
    #test_example_from_paper()
    #test_add_lyapunov_constraints()
    n = 2
    max_degree = 2
    assert(n >= 2)
    assert(n%2 == 0)


    # These are the transformed variables f and g such that
    # the equilibrium is at (0,0)
    f_centered = sp.symarray('f', shape=(n-1,))
    g_centered = sp.symarray('g', shape=(n-1,))
    original_equilibrium = np.zeros_like(f_centered)
    for i in range(int(n/2)):
        original_equilibrium[i] = 2

    lb_f, lb_H_f = lower_bound_constraints(f_centered, g_centered)
    lb_g, lb_H_g = lower_bound_constraints(g_centered, f_centered, reverse=True)
    ub_f, ub_H_f = upper_bound_constraint(f_centered, g_centered)
    ub_g, ub_H_g = upper_bound_constraint(g_centered, f_centered, reverse = True)
    constraints = lb_f + lb_g + ub_f + ub_g
    H = lb_H_f + lb_H_g + ub_H_f + ub_H_g


    original_f = to_uncentered_unreduced(f_centered).reshape((n,1))
    original_g = to_uncentered_unreduced(g_centered).reshape((n,1))

    utility_f = compute_utility(n, original_f, original_g)[0]
    utility_g = compute_utility(n, original_g, original_f)[0]

    grad_utility_f = compute_gradient(utility_f, f_centered)
    grad_utility_g = compute_gradient(utility_g, g_centered)

    vectorfield = np.concatenate([grad_utility_f, grad_utility_g])

    problem = sos.SOSProblem()
    lyapunov = add_lyapunov_constraints(
        problem=problem,
        max_degree=max_degree,
        variables=np.concatenate([f_centered, g_centered]).tolist(),
        f=vectorfield.tolist(),
        inequality_constraints=constraints,
        H=H
    )
    print('Start solving...')
    problem.solve()
    print('Solved, writing Lyapunov function to file...')
    values = [(param, problem.sym_to_var(param).value) for param in lyapunov['lyapunov_params'].flatten()]
    print(lyapunov['lyapunov_function'].subs(values).simplify())
    file = open(f'lyapunov_n={n}_deg={max_degree}.txt', 'w')
    file.write(str(lyapunov['lyapunov_function'].subs(values).simplify()))
    file.close()
    np.savetxt(f'lyapunov_parameters_n={n}_deg={max_degree}.txt', get_parameter_values(lyapunov['lyapunov_params'], problem))
    computed_parameters = {
         'monomials': [str(m) for m in lyapunov['monomials']],
         'lyapunov_parameters': get_parameter_values(lyapunov['lyapunov_params'], problem).tolist(),
         'interior_parameters': get_parameter_values(lyapunov['interior_params'], problem).tolist(),
         'inequality_parameters': [
              get_parameter_values(params, problem).tolist()
              for params in lyapunov['tight_params']
         ]
    }
    with open(f"lyapunov_params_n={n}_deg={max_degree}.json", "w") as outfile:
        json.dump(computed_parameters, outfile)






