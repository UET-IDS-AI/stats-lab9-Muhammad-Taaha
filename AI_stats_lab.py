import numpy as np


# -------------------------------------------------
# Sparse 4 by 4 Joint PMF
# -------------------------------------------------

def joint_pmf(x, y):
    table = [
        [0.10, 0.05, 0.00, 0.00],
        [0.15, 0.20, 0.05, 0.00],
        [0.00, 0.10, 0.15, 0.05],
        [0.00, 0.00, 0.05, 0.10]
    ]

    # ✅ Handle out-of-range safely
    if 0 <= x < 4 and 0 <= y < 4:
        return table[x][y]
    return 0.0


def marginal_px(x):
    return sum(joint_pmf(x, y) for y in range(4))


def marginal_py(y):
    return sum(joint_pmf(x, y) for x in range(4))


def conditional_pmf_x_given_y(x, y):
    py = marginal_py(y)
    if py == 0:
        return 0.0
    return joint_pmf(x, y) / py


def conditional_distribution_x_given_y(y):
    return {x: conditional_pmf_x_given_y(x, y) for x in range(4)}


def probability_sum_greater_than_3():
    total = 0.0
    for x in range(4):
        for y in range(4):
            if x + y > 3:
                total += joint_pmf(x, y)
    return total


def independence_check():
    for x in range(4):
        for y in range(4):
            if not np.isclose(joint_pmf(x, y), marginal_px(x) * marginal_py(y)):
                return False
    return True


# -------------------------------------------------
# Expectation, Covariance, and Correlation
# -------------------------------------------------

def expected_x():
    return sum(x * marginal_px(x) for x in range(4))


def expected_y():
    return sum(y * marginal_py(y) for y in range(4))


def expected_xy():
    total = 0.0
    for x in range(4):
        for y in range(4):
            total += x * y * joint_pmf(x, y)
    return total


def variance_x():
    ex = expected_x()
    ex2 = sum((x ** 2) * marginal_px(x) for x in range(4))
    return ex2 - ex ** 2


def variance_y():
    ey = expected_y()
    ey2 = sum((y ** 2) * marginal_py(y) for y in range(4))
    return ey2 - ey ** 2


def covariance_xy():
    return expected_xy() - expected_x() * expected_y()


def correlation_xy():
    cov = covariance_xy()
    varx = variance_x()
    vary = variance_y()
    if varx == 0 or vary == 0:
        return 0.0
    return cov / np.sqrt(varx * vary)


def variance_sum():
    total = 0.0
    exy = expected_x() + expected_y()
    for x in range(4):
        for y in range(4):
            total += ((x + y) ** 2) * joint_pmf(x, y)
    return total - exy ** 2


def variance_identity_check():
    lhs = variance_sum()
    rhs = variance_x() + variance_y() + 2 * covariance_xy()
    return bool(np.isclose(lhs, rhs))  # ✅ Convert to Python bool