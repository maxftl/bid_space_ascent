import sympy as sp
import SumOfSquares as sos
import numpy as np
from sympy.polys.monomials import itermonomials
import logging
from bid_space_ascent.utils import compute_gradient

logger = logging.getLogger(__name__)


def add_poly_equality_constraint(
        problem: sos.SOSProblem,
        poly: sp.Poly
):
    logger.info("Adding polynomial equality constraints")
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
    parametrized_polynomials, parameters = generate_parametrized_polynomials(
        len(inequality_constraints)+1,
        monomials,
        parameter_prefix
    )
    base_polynomials = np.concatenate([[1], inequality_constraints])
    poly_expression = base_polynomials\
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
    return {
        'polynomials': parametrized_polynomials,
        'parameters': parameters,
        'base': base_polynomials
        }


def add_interior_constraint(
        problem,
        gradV,
        f,
        inequality_constraints,
        variables,
        monomials,
        parameter_prefix
):
    parametrized_polynomials, parameters = generate_parametrized_polynomials(
        len(inequality_constraints)+1,
        monomials,
        parameter_prefix
    )
    base_polynomials = np.concatenate([[1],inequality_constraints])
    poly_expression = np.dot(
        base_polynomials,
        parametrized_polynomials
    )
    logger.info("Computing difference polynomial")
    difference = sp.Poly(
            poly_expression + np.dot(gradV, f),
            *variables
        )
    add_poly_equality_constraint(
        problem,
        difference
    )
    for polynomial in parametrized_polynomials:
        problem.add_sos_constraint(polynomial, variables)
    return {
        'polynomials': parametrized_polynomials,
        'parameters': parameters,
        'base': base_polynomials
        }



def get_parameter_values(parameters, problem):
    fun = np.vectorize(lambda param: problem.sym_to_var(param).value)
    return fun(parameters)

def add_lyapunov_constraints(
        problem,
        max_degree,
        variables,
        f,
        inequality_constraints,
        H
):
    result = dict()
    monomials = list(itermonomials(variables, max_degrees=max_degree))
    num_constraints = len(inequality_constraints)
    base_polynomials = np.concatenate([[1],inequality_constraints])
    V_polys, V_params = generate_parametrized_polynomials(
        n_polynomials=num_constraints+1,
        monomials=monomials,
        parameter_prefix='sigma'
    )
    result['lyapunov_parametrization'] = {
        'polynomials': V_polys,
        'parameters': V_params,
        'base': base_polynomials
    }
    V = np.dot(V_polys,base_polynomials)
    result['lyapunov_function'] = V
    for poly in V_polys:
        problem.add_sos_constraint(poly, variables)

    gradV = compute_gradient(V, variables)
    logger.info("Adding interior constraint")
    result['interior_constraint'] = add_interior_constraint(
        problem=problem,
        gradV=gradV,
        f=f,
        inequality_constraints=inequality_constraints,
        variables=variables,
        monomials=monomials,
        parameter_prefix='chi'
    )
    result['inequality_constraints'] = []
    for i in range(num_constraints):
        logger.info(f"Adding inequality constraint {i}")
        result['inequality_constraints'].append(add_tight_inequality_constraint(
            problem=problem,
            gradV=gradV,
            inequality_constraints=inequality_constraints,
            constraint_index=i,
            H=H[i],
            variables=variables,
            monomials=monomials,
            parameter_prefix=f'phi{i}*'
        ))
    return result
    

    