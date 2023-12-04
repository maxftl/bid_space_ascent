import sympy as sp
import numpy as np
from lyapunov_utils import *
import SumOfSquares as sos
from sympy.polys.monomials import itermonomials


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




if __name__ == '__main__':
    #test_1D_direct()
    test_example_from_paper()

