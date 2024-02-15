import math

import numpy as np

from matplotlib import pyplot as plt

from scipy.optimize import bisect

from tqdm import tqdm


class LinearUnit:
    def __init__(self, u_i, u_i_plus_1, y_intercept, slope):
        if u_i_plus_1 <= u_i:
            raise ValueError("u_{i+1} must be greater than u_i")

        self.u_i = u_i
        self.u_i_plus_1 = u_i_plus_1
        self.y_intercept = y_intercept
        self.slope = slope

    def __repr__(self):
        return f"LinearUnit(u_i={self.u_i}, u_i_plus_1={self.u_i_plus_1}, y_intercept={self.y_intercept}, slope={self.slope})"

    def evaluate(self, x):
        if x < self.u_i or x > self.u_i_plus_1:
            raise ValueError(
                f"x is out of bounds {self.u_i} to {self.u_i_plus_1} of linear unit"
            )
        return self.y_intercept + self.slope * x


class PLU:
    def __init__(self):
        self.x_min = math.inf
        self.x_max = -math.inf
        self.linear_units: list[LinearUnit] = []

    def __repr__(self):
        linear_units = "\n".join(repr(lu) for lu in self.linear_units)
        return f"PLU(\nx_min={self.x_min}, x_max={self.x_max},\n{linear_units})"

    def add_linear_unit(self, linear_unit):
        self.x_min = min(self.x_min, linear_unit.u_i)
        self.x_max = max(self.x_max, linear_unit.u_i_plus_1)
        self.linear_units.append(linear_unit)

    def get_linear_unit(self, x):
        if x < self.x_min or x > self.x_max:
            raise ValueError(f"x is out of bounds {self.x_min} to {self.x_max}")

        for lu in self.linear_units:
            if lu.u_i <= x <= lu.u_i_plus_1:
                return lu
        raise ValueError(
            "No linear unit found. You likely constructed the PLU incorrectly."
        )

    def evaluate(self, x):
        if x < self.x_min:
            return 0
        if x > self.x_max:
            return math.inf

        lu = self.get_linear_unit(x)
        return lu.evaluate(x)


# def exp_arc_length_integrand(x):
#     return math.sqrt(1 + math.exp(2*x))


def exp_arc_length_integral(x):
    return (
        math.sqrt(1 + math.exp(2 * x))
        + math.log(math.sqrt(1 + math.exp(2 * x)) - 1)
        - x
    )


# def exp_arc_length(a, b):
#     return quad(exp_arc_length_integrand, a, b)[0]


def exp_arc_length(a, b):
    return exp_arc_length_integral(b) - exp_arc_length_integral(a)


def exp_to_plu(k: int, u_l: float = -10, u_h: float = 10) -> PLU:
    """
    Approximate an exponential function as a piecewise linear unit (PLU).
    Each linear unit is a secant line between two points on the exponential curve.
    The arc length between each pair of points is the same.

    k: number of segments. Must be at least 3.
    u_l: low threshold x value. Values below this will be approximated as 0.
    u_h: high threshold x value. Values above this will be approximated as infinity.
    """
    if k < 3:
        raise ValueError("k must be at least 3")

    # Arc length between u_l and u_h
    total_arc_length = exp_arc_length(u_l, u_h)
    segment_arc_length = total_arc_length / (k - 2)

    plu = PLU()

    u_i = u_l
    for i in tqdm(range(k - 2)):
        if i < k - 3:
            u_i_plus_1 = bisect(
                lambda x: exp_arc_length(u_i, x) - segment_arc_length, u_i, u_h
            )
        else:
            u_i_plus_1 = u_h

        y_i = math.exp(u_i)
        y_i_plus_1 = math.exp(u_i_plus_1)
        slope = (y_i_plus_1 - y_i) / (u_i_plus_1 - u_i)
        y_intercept = y_i - slope * u_i
        linear_unit = LinearUnit(u_i, u_i_plus_1, y_intercept, slope)
        plu.add_linear_unit(linear_unit)
        u_i = u_i_plus_1

    return plu


# == Inversion


def invert_linear_unit(linear_unit):
    """
    Invert a linear unit.
    """
    return LinearUnit(
        linear_unit.y_intercept + linear_unit.slope * linear_unit.u_i,
        (linear_unit.y_intercept + linear_unit.slope * linear_unit.u_i_plus_1),
        -linear_unit.y_intercept / linear_unit.slope,
        1 / linear_unit.slope,
    )


def invert_plu(plu):
    """
    Invert a PLU.
    """
    inverted_plu = PLU()
    for lu in plu.linear_units:
        inverted_plu.add_linear_unit(invert_linear_unit(lu))
    return inverted_plu


# == Drawing


def draw_plu(plu: PLU, a, b, ax):
    x = np.linspace(a, b, 100)
    y = [plu.evaluate(xi) for xi in x]
    ax.set_title(f"PLU approximation of $e^x$ ({a} < x < {b})")
    ax.set_xlabel("x")
    ax.set_ylabel("PLU(x)")
    ax.plot(x, y)


# == PLU approximation of SoftMax


def plu_max(plu: PLU, T: np.ndarray, a=-1000, b=1000) -> np.ndarray:
    """
    PLU approximation of SoftMax.

    T is the input vector.
    """
    out = np.zeros(T.shape)
    T_rows = T.reshape(-1, T.shape[-1])

    for T_row in T_rows:
        _lambda = bisect(lambda l: sum(plu.evaluate(t - l) for t in T_row) - 1, a, b)
        for i, t in enumerate(T_row):
            out[..., i] = plu.evaluate(t - _lambda)

    return out


if __name__ == "__main__":
    k = int(input("Enter the number of pieces (min 3, recommended 10000+): "))
    plu = exp_to_plu(k)

    print("Exponential PLU Created.")
    print()

    fig, ax = plt.subplots(1, 3)
    draw_plu(plu, -11, 11, ax[0])
    draw_plu(plu, -3, 3, ax[1])
    draw_plu(plu, -1, 1, ax[2])
    plt.show()

    while True:
        T = input("Enter T array (as space-separated numbers, e.g. '1 2 3'): ")
        if T == "exit":
            break
        T = np.array([float(x) for x in T.split()])

        print(plu_max(plu, T))
