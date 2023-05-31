from jax import grad, jit, vmap
from jax.tree_util import tree_flatten
import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import ScipyRootFinding


class LagrangeMultiplier:
    """maximize f(x) subject to g(x) = 0
    """

    def __init__(self, objective, constraint):
        self.objective = objective
        self.constraint = constraint

    def build_lagrange_function(self):
        def f(v: jnp.ndarray) -> float:
            """lagrange function
            v: (x_1, ..., x_n, l)
            x_1, ..., x_n: decision variable
            l: lagrange multiplier
            """
            x, l = v[:-1], v[-1]
            return self.objective(x) - l * self.constraint(x)
        return f

    def solve(self, p0):
        lagrange_function = self.build_lagrange_function()
        grad_vec = grad(lagrange_function)
        tol = 1e-6
        x_solver = ScipyRootFinding(
            optimality_fun=grad_vec, method='hybr', tol=tol, use_jacrev=True)
        return x_solver.run(jnp.array(p0))


def obj(x): return (x[0] - 1)**2 + (x[1])**2


def con(x): return x[0] + x[1] - 1


solver = LagrangeMultiplier(obj, con)
res = solver.solve(jnp.array([0., 1., 1.]))

print(res.params)
