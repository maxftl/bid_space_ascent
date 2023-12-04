import sympy as sp
import SumOfSquares as sos
import numpy as np
from functools import reduce
from sympy.polys.monomials import itermonomials


def add_poly_equality_constraint(
        problem: sos.SOSProblem,
        poly: sp.Poly
):
    for coeff in poly.coeffs():
        picos_expr = problem.sp_to_picos(coeff)
        problem.add_constraint(picos_expr == 0)


def generate_parametrized_polynomials(
        n_polynomials,
        monomials,
        parameter_prefix
):
    parameters = sp.symarray(
        parameter_prefix,
        shape=(n_polynomials, len(monomials))
    )
    return parameters @ monomials, parameters
    


def add_tight_inequality_constraint(
        problem,
        gradV,
        inequality_constraints,
        constraint_index,
        H,
        variables,
        monomials,
        parameter_prefix
):
    g = inequality_constraints[constraint_index]
    gradG = [g.diff(v) for v in variables]
    substitution = list(zip(variables, H))
    parametrized_polynomials, _ = generate_parametrized_polynomials(
        len(inequality_constraints)+1,
        monomials,
        parameter_prefix
    )
    poly_expression = np.concatenate([[1], inequality_constraints])\
        @ parametrized_polynomials
    gVG = np.dot(gradV,gradG)
    add_poly_equality_constraint(
        problem,
        sp.Poly( (poly_expression + gVG).subs(substitution), *variables)
    )
    for i in range(0,len(inequality_constraints)+1):
        if i-1 == constraint_index:
            continue
        else:
            problem.add_sos_constraint(
                parametrized_polynomials[i],
                variables
            )


def add_interior_constraint(
        problem,
        gradV,
        f,
        inequality_constraints,
        variables,
        monomials,
        parameter_prefix
):
    parametrized_polynomials, _ = generate_parametrized_polynomials(
        len(inequality_constraints)+1,
        monomials,
        parameter_prefix
    )
    poly_expression = np.dot(
        np.concatenate([[1],inequality_constraints]),
        parametrized_polynomials
    )
    add_poly_equality_constraint(
        problem,
        sp.Poly(
            poly_expression + np.dot(gradV, f),
            *variables
        )
    )
    for polynomial in parametrized_polynomials:
        problem.add_sos_constraint(polynomial, variables)

def compute_gradient(V, variables):
    return np.array([V.diff(var) for var in variables])

def add_lyapunov_constraints(
        problem,
        max_degree,
        variables,
        f,
        constraints,
        H
):
    monomials = list(itermonomials(variables, max_degrees=max_degree))
    num_constraints = len(constraints)

    V_polys, V_params = generate_parametrized_polynomials(
        n_polynomials=num_constraints+1,
        monomials=monomials,
        parameter_prefix='sigma'
    )
    V = np.dot(V_polys,monomials)
    for poly in V_polys:
        problem.add_sos_constraint(poly, variables)

    gradV = compute_gradient(V)
    add_interior_constraint(
        problem=problem,
        gradV=gradV,
        f=f,
        inequality_constraints=inequality_constraints,
        variables=variables,
        monomials=monomials,
        parameter_prefix='chi'
    )
    for i in range(num_constraints):
        add_tight_inequality_constraint(
            problem=problem,
            gradV=gradV,
            inequality_constraints=inequality_constraints,
            constraint_index=i,
            H=H,
            variables=variables,
            monomials=monomials,
            parameter_prefix=f'phi{i}*'
        )
    

    