import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def calculate_optimal_spread_kernel(
    gamma: float, sigma: float, time_factor: float, kappa: float, min_spread: float
) -> float:
    """
    JIT-compiled kernel for Avellaneda-Stoikov spread calculation.
    s = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + (gamma / kappa))
    """
    # Prevent division by zero or log error
    if gamma < 1e-9:
        gamma = 1e-9

    spread = (gamma * (sigma * sigma) * time_factor) + (
        (2.0 * np.log(1.0 + (gamma / kappa))) / gamma
    )

    if spread < min_spread:
        return min_spread
    return spread


@jit(nopython=True, cache=True)
def calculate_reservation_price_kernel(
    mid_price: float,
    position_size: float,
    gamma: float,
    sigma: float,
    time_horizon: float,
) -> float:
    """
    JIT-compiled kernel for reservation price.
    r = s - q * gamma * sigma^2 * (T - t)
    """
    inventory_skew = position_size * gamma * (sigma * sigma) * time_horizon
    return mid_price - inventory_skew
