import sympy as sp
import numpy as np
from SumOfSquares import SOSProblem, poly_opt_prob

x, y = sp.symbols('x y')
p = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1
prob = SOSProblem()
prob.add_sos_constraint(p, [x, y])
prob.solve() # Raises SolutionFailure error due to infeasibility