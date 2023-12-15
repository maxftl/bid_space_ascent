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
    result['monomials'] = monomials
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
    

def set_minimize_max_degree_objective(problem, lyapunov_info, variables):
    lyapunov_poly = sp.Poly(lyapunov_info['lyapunov_function'], *variables)
    max_degree = lyapunov_poly.total_degree()
    max_deg_variables = []
    for exponents, coeff in lyapunov_poly.terms():
        if np.sum(exponents) < max_degree:
            continue
        coeff_abs = sp.Symbol(f'coeff_max_{len(max_deg_variables)}')
        max_deg_variables.append(coeff_abs)
        coeff_abs_minus_coeff = problem.sp_to_picos(coeff_abs - coeff)
        coeff_abs_plus_coeff = problem.sp_to_picos(coeff_abs + coeff)
        problem.add_constraint(coeff_abs_minus_coeff >= 0)
        problem.add_constraint(coeff_abs_plus_coeff >= 0)

    problem.set_objective('min', problem.sp_to_picos(np.sum(max_deg_variables)))

def set_minimize_coefficients_l1_norm(problem, lyapunov_info, variables):
    lyapunov_poly = sp.Poly(lyapunov_info['lyapunov_function'], *variables)
    l1_variables = []
    factors = []
    for exponents, coeff in lyapunov_poly.terms():
        if np.sum(exponents) == 2:
            factors.append(-1)
        else:
            factors.append(1)
        coeff_abs = sp.Symbol(f'coeff_abs_{len(l1_variables)}')
        l1_variables.append(coeff_abs)
        coeff_abs_minus_coeff = problem.sp_to_picos(coeff_abs - coeff)
        coeff_abs_plus_coeff = problem.sp_to_picos(coeff_abs + coeff)
        problem.add_constraint(coeff_abs_minus_coeff >= 0)
        problem.add_constraint(coeff_abs_plus_coeff >= 0)
        if np.sum(exponents) == 2:
            problem.add_constraint(problem.sp_to_picos(coeff_abs) <= 100.)

    problem.set_objective('min', problem.sp_to_picos(np.dot(l1_variables,factors)))




def add_symmetric_lyapunov_constraints(
        problem,
        variables,
        f,
        inequality_constraints,
        H
):
    result = dict()
    max_degree = 2
    monomials = list(itermonomials(variables, max_degrees=max_degree))
    quadratic_monomials = list(itermonomials(variables, max_degrees=max_degree, min_degrees=max_degree))
    num_constraints = len(inequality_constraints)
    base_polynomials = np.concatenate([[1],inequality_constraints])
    V_polys, V_params = generate_parametrized_polynomials(
        n_polynomials=1,
        monomials=quadratic_monomials,
        parameter_prefix='sigma'
    )
    result['monomials'] = monomials
    result['quadratic_monomials'] = quadratic_monomials
    result['lyapunov_parametrization'] = {
        'polynomials': V_polys,
        'parameters': V_params,
        'base': base_polynomials
    }
    V = np.dot(V_polys,[1])
    result['lyapunov_function'] = V
    for poly in V_polys:
        problem.add_sos_constraint(poly, variables)
    # Force one quadratic term to be non-zero, this is not very clean:
    quad_coeff = sp.Poly(V, *variables).coeff_monomial(variables[0]**2)
    coeff_expr = problem.sp_to_picos(quad_coeff)
    problem.add_constraint(coeff_expr == 1)
    # End of forcing

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
    """for i in range(num_constraints):
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
        ))"""
    return result


