import torch
import logging
import warnings
import time
import os.path
import numpy as np
from time import perf_counter

logger = logging.getLogger(__name__)

"""This file contains various utility functions for the integrations methods."""

def setup_integration_domain(dim, integration_domain):
    """Sets up the integration domain if unspecified by the user.

    Args:
        dim (int): Dimensionality of the integration domain.
        integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

    Returns:
        torch.tensor: Integration domain.
    """

    # Store integration_domain
    # If not specified, create [-1,1]^d bounds
    logger.debug("Setting up integration domain.")
    if integration_domain is not None:
        if len(integration_domain) != dim:
            raise ValueError(
                "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]."
            )
        if type(integration_domain) == torch.Tensor:
            return integration_domain
        else:
            return torch.tensor(integration_domain)
    else:
        return torch.tensor([[-1, 1]] * dim)

class IntegrationGrid:
    """This class is used to store the integration grid for methods like Trapezoid or Simpsons, which require a grid."""

    points = None  # integration points
    h = None  # mesh width
    _N = None  # number of mesh points
    _dim = None  # dimensionality of the grid
    _runtime = None  # runtime for the creation of the integration grid

    def __init__(self, N, integration_domain):
        """Creates an integration grid of N points in the passed domain. Dimension will be len(integration_domain)

        Args:
            N (int): Total desired number of points in the grid (will take next lower root depending on dim)
            integration_domain (list): Domain to choose points in, e.g. [[-1,1],[0,1]].
        """
        start = perf_counter()
        self._check_inputs(N, integration_domain)
        self._dim = len(integration_domain)

        # TODO Add that N can be different for each dimension
        # A rounding error occurs for certain numbers with certain powers,
        # e.g. (4**3)**(1/3) = 3.99999... Because int() floors the number,
        # i.e. int(3.99999...) -> 3, a little error term is useful
        self._N = int(N ** (1.0 / self._dim) + 1e-8)  # convert to points per dim

        self.h = torch.zeros([self._dim])

        logger.debug(
            "Creating "
            + str(self._dim)
            + "-dimensional integration grid with "
            + str(N)
            + " points over"
            + str(integration_domain),
        )
        grid_1d = []
        # Determine for each dimension grid points and mesh width
        for dim in range(self._dim):
            grid_1d.append(
                torch.linspace(
                    integration_domain[dim][0], integration_domain[dim][1], self._N
                )
            )
            self.h[dim] = grid_1d[dim][1] - grid_1d[dim][0]

        logger.debug("Grid mesh width is " + str(self.h))

        # Get grid points
        points = torch.meshgrid(*grid_1d)

        # Flatten to 1D
        points = [p.flatten() for p in points]

        self.points = torch.stack((tuple(points))).transpose(0, 1)

        logger.info("Integration grid created.")

        self._runtime = perf_counter() - start

    def _check_inputs(self, N, integration_domain):
        """Used to check input validity"""

        logger.debug("Checking inputs to IntegrationGrid.")
        dim = len(integration_domain)

        if dim < 1:
            raise ValueError("len(integration_domain) needs to be 1 or larger.")

        if N < 2:
            raise ValueError("N has to be > 1.")

        if N ** (1.0 / dim) < 2:
            raise ValueError(
                "Cannot create a ",
                dim,
                "-dimensional grid with ",
                N,
                " points. Too few points per dimension.",
            )

        for bounds in integration_domain:
            if len(bounds) != 2:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )
            if bounds[0] > bounds[1]:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )

"""#### BaseIntegrator"""

class BaseIntegrator:
    """The (abstract) integrator that all other integrators inherit from. Provides no explicit definitions for methods."""

    # Function to evaluate
    _fn = None

    # Dimensionality of function to evaluate
    _dim = None

    # Integration domain
    _integration_domain = None

    # Number of function evaluations
    _nr_of_fevals = None

    def __init__(self):
        self._nr_of_fevals = 0

    def integrate(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )

    def _eval(self, points):
        """Evaluates the function at the passed points and updates nr_of_evals

        Args:
            points (torch.tensor): Integration points
        """
        self._nr_of_fevals += len(points)
        result = self._fn(points)
        if type(result) != torch.Tensor:
            warnings.warn(
                "The passed function did not return a torch.tensor. Will try to convert. Note that this may be slow as it results in memory transfers between CPU and GPU, if torchquad uses the GPU."
            )
            result = torch.tensor(result)

        if len(result) != len(points):
            raise ValueError(
                f"The passed function was given {len(points)} points but only returned {len(result)} value(s)."
                f"Please ensure that your function is vectorized, i.e. can be called with multiple evaluation points at once. It should return a tensor "
                f"where first dimension matches length of passed elements. "
            )

        return result

    def _check_inputs(self, dim=None, N=None, integration_domain=None):
        """Used to check input validity

        Args:
            dim (int, optional): Dimensionality of function to integrate. Defaults to None.
            N (int, optional): Total number of integration points. Defaults to None.
            integration_domain (list, optional): Integration domain, e.g. [[0,1],[1,2]]. Defaults to None.

        Raises:
            ValueError: if inputs are not compatible with each other.
        """
        logger.debug("Checking inputs to Integrator.")
        if dim is not None:
            if dim < 1:
                raise ValueError("Dimension needs to be 1 or larger.")

            if integration_domain is not None:
                if dim != len(integration_domain):
                    raise ValueError(
                        "Dimension of integration_domain needs to match the passed function dimensionality dim."
                    )

        if N is not None:
            if N < 1 or type(N) is not int:
                raise ValueError("N has to be a positive integer.")

        if integration_domain is not None:
            for bounds in integration_domain:
                if len(bounds) != 2:
                    raise ValueError(
                        bounds,
                        " in ",
                        integration_domain,
                        " does not specify a valid integration bound.",
                    )
                if bounds[0] > bounds[1]:
                    raise ValueError(
                        bounds,
                        " in ",
                        integration_domain,
                        " does not specify a valid integration bound.",
                    )

"""#### MonteCarlo"""

class MonteCarlo(BaseIntegrator):
    """Monte Carlo integration in torch."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=1000, integration_domain=None, seed=None):
        """Integrates the passed function on the passed domain using vanilla Monte Carlo Integration.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
            seed (int, optional): Random number generation seed to the sampling point creation, only set if provided. Defaults to None.

        Raises:
            ValueError: If len(integration_domain) != dim

        Returns:
            float: integral value
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.debug(
            "Monte Carlo integrating a "
            + str(dim)
            + "-dimensional fn with "
            + str(N)
            + " points over "
            + str(integration_domain),
        )

        self._dim = dim
        self._nr_of_fevals = 0
        self.fn = fn
        self._integration_domain = setup_integration_domain(dim, integration_domain)
        if seed is not None:
            torch.random.manual_seed(seed)

        logger.debug("Picking random sampling points")
        sample_points = torch.zeros([N, dim])
        for d in range(dim):
            scale = self._integration_domain[d, 1] - self._integration_domain[d, 0]
            offset = self._integration_domain[d, 0]
            sample_points[:, d] = torch.rand(N) * scale + offset

        logger.debug("Evaluating integrand")
        function_values = fn(sample_points)

        logger.debug("Computing integration domain volume")
        scales = self._integration_domain[:, 1] - self._integration_domain[:, 0]
        volume = torch.prod(scales)

        # Integral = V / N * sum(func values)
        integral = volume * torch.sum(function_values) / N
        logger.info("Computed integral was " + str(integral))
        return integral

"""#### Trapezoid"""

class Trapezoid(BaseIntegrator):
    """Trapezoidal rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=1000, integration_domain=None):
        """Integrates the passed function on the passed domain using the trapezoid rule.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Total number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value
        """
        self._integration_domain = setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)

        logger.debug(
            "Using Trapezoid for integrating a fn with "
            + str(N)
            + " points over "
            + str(self._integration_domain)
            + "."
        )

        self._dim = dim
        self._fn = fn

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, self._integration_domain)

        logger.debug("Evaluating integrand on the grid.")
        function_values = self._eval(self._grid.points)

        # Reshape the output to be [N,N,...] points
        # instead of [dim*N] points
        function_values = function_values.reshape([self._grid._N] * dim)

        logger.debug("Computing trapezoid areas.")

        # This will contain the trapezoid areas per dimension
        cur_dim_areas = function_values

        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                self._grid.h[cur_dim]
                / 2.0
                * (cur_dim_areas[..., 0:-1] + cur_dim_areas[..., 1:])
            )
            cur_dim_areas = torch.sum(cur_dim_areas, dim=dim - cur_dim - 1)

        logger.info("Computed integral was " + str(cur_dim_areas) + ".")

        return cur_dim_areas


dimensions = [1,2,3,4,5,10,20]
num_reruns = [10,10,10,10,10,5,5]
sample_points = [1,4,7,10]
minimum = [100,1000,10000,100000]

# The function we want to integrate, in this example f(x0,x1) = sin(x0) + e^x1 for x0=[0,1] and x1=[-1,1]
# Note that the function needs to support multiple evaluations at once (first dimension of x here)
# Expected result here is ~3.2698
def some_function(x):
    ans = 1
    for i in range(len(x[0])):
        ans *= 2/(torch.exp(x[:,i]) + 1)
    return ans
    #return torch.sin(x[:,0]) + torch.exp(x[:,1]) * torch.cos(x[:,2]+1)

# Declare an integrator, here we use the simple, stochastic Monte Carlo integration method
mc = MonteCarlo()
trap = Trapezoid()

# Compute the function integral by sampling 10000 points over domain
integral_value = mc.integrate(some_function,dim=3,N=10000,integration_domain = [[0,1],[-1,1],[-1,1]])
print(integral_value)

run = 0
for i in range(1,100000):
    if os.path.isfile("mc_trap_runs/points_" + str(sample_points[0]) + "_run_" + str(i)):
        run += 1
    else:
        run = i
        break

print("USING RUN: ", run)
for asdf,sample_point in enumerate(sample_points):
    print("On sample_points ", sample_point)
    f = open("mc_trap_runs/points_" + str(sample_point) + "_run_" + str(run), "a")
    f.write("dim,sample_points,num_reruns,mc_ans,mc_1000_time,trap_ans,trap_1000_time\n")
    for j, dim in enumerate(dimensions):
        sample_point_dim = sample_point*(2**dim) if sample_point*(2**dim) > minimum[asdf] else minimum[asdf]
        print("Sample points: ", sample_point_dim)
        print("On dim ", dim)
        int_domain = np.tile([0,1], (dim, 1))

        mc_start_time = time.perf_counter()
        mc_ans = 0
        for num_rerun in range(num_reruns[j]):
            mc_ans += mc.integrate(some_function, dim=dim, N=sample_point_dim, integration_domain=int_domain)
        mc_ans = mc_ans/num_reruns[j]
        mc_end_time = time.perf_counter()
        mc_time = 1000*(mc_end_time - mc_start_time)/num_reruns[j]

        trap_start_time = time.perf_counter()
        trap_ans = 0
        for num_rerun in range(num_reruns[j]):
            trap_ans += trap.integrate(some_function, dim=dim, N=sample_point_dim, integration_domain=int_domain)
        trap_ans = trap_ans/num_reruns[j]
        trap_end_time = time.perf_counter()
        trap_time = 1000*(trap_end_time - trap_start_time)/num_reruns[j]
        f.write(str(dim)+","+str(sample_points)+","+str(num_reruns[j])+","+str("%.4f"%mc_ans.item())+","+str("%.4f"%mc_time)+","+str("%.4f"%trap_ans.item())+","+str("%.4f"%trap_time)+"\n")

