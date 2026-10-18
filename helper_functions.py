"""Helper functions for the Bass diffusion model.

The formulas follow the DS 223 lecture slides (Bass Model for Innovation
Diffusion, K. Hovhannisyan) one-to-one; the slide R code is ported to Python.
"""
import numpy as np
from scipy.optimize import curve_fit


def bass_f(t, p, q):
    """f(t): fraction of the market adopting at time t (slide 17, bass.f)."""
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return ((p + q) ** 2 / p) * e / (1 + (q / p) * e) ** 2


def bass_F(t, p, q):
    """F(t): cumulative fraction adopted up to time t (slide 17, bass.F)."""
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / p) * e)


def bass_a(t, m, p, q):
    """a(t) = m * f(t): number of adopters at time t (slide 34)."""
    return m * bass_f(t, p, q)


def bass_A(t, m, p, q):
    """A(t) = m * F(t): cumulative adopters up to time t (slide 11)."""
    return m * bass_F(t, p, q)


def peak_time(p, q):
    """t* = ln(q/p) / (p+q): period of peak adoptions (slide 39)."""
    return np.log(q / p) / (p + q)


def fit_nls(sales, m0=None, p0=0.02, q0=0.4):
    """Method 1 (slide 34): non-linear least squares of sales ~ m*f(t), t = 1..n.

    Same starting values as the R nls() call in the slides:
    m = sum(sales), p = 0.02, q = 0.4.
    Returns dict with estimates, standard errors, fitted values and R^2.
    """
    sales = np.asarray(sales, dtype=float)
    t = np.arange(1, len(sales) + 1)
    if m0 is None:
        m0 = sales.sum()
    est, cov = curve_fit(bass_a, t, sales, p0=[m0, p0, q0], maxfev=20000,
                         bounds=([0, 1e-6, 1e-6], [np.inf, 1, 3]))
    se = np.sqrt(np.diag(cov))
    fitted = bass_a(t, *est)
    resid = sales - fitted
    dof = len(sales) - 3
    return {
        "m": est[0], "p": est[1], "q": est[2],
        "se_m": se[0], "se_p": se[1], "se_q": se[2],
        "fitted": fitted,
        "sigma": float(np.sqrt((resid ** 2).sum() / dof)),
        "r2": r_squared(sales, fitted),
    }


def fit_bass_ols(sales):
    """Method 2 (slides 13-14, Bass 1969): OLS of
    a(t) = b0 + b1*A(t-1) + b2*A(t-1)^2,
    with b0 = p*m, b1 = q - p, b2 = -q/m.

    A(t-1) is cumulative sales *before* period t. m solves
    b2*m^2 + b1*m + b0 = 0 (positive root), then p = b0/m, q = -b2*m.
    """
    sales = np.asarray(sales, dtype=float)
    A_prev = np.concatenate([[0.0], np.cumsum(sales)[:-1]])
    X = np.column_stack([np.ones_like(A_prev), A_prev, A_prev ** 2])
    (b0, b1, b2), *_ = np.linalg.lstsq(X, sales, rcond=None)
    roots = np.roots([b2, b1, b0])
    m = float(max(r.real for r in roots if abs(r.imag) < 1e-9))
    p, q = b0 / m, -b2 * m
    fitted = X @ np.array([b0, b1, b2])
    return {"b0": b0, "b1": b1, "b2": b2, "m": m, "p": p, "q": q,
            "fitted": fitted, "r2": r_squared(sales, fitted)}


def r_squared(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cumulative_at(dates, m, p, q, t0):
    """A(t) at calendar dates (decimal years), where t = date - t0."""
    t = np.maximum(np.asarray(dates, dtype=float) - t0, 0.0)
    return np.where(t > 0, bass_A(np.maximum(t, 1e-12), m, p, q), 0.0)


def calibrate_origin(dates, cumulative, m, p, q, guess=2020.0):
    """Find the diffusion origin t0 (decimal year) so that A(date - t0)
    best matches observed cumulative adopters, holding m, p, q fixed."""
    from scipy.optimize import least_squares
    obs = np.asarray(cumulative, dtype=float)
    res = least_squares(lambda th: cumulative_at(dates, m, p, q, th[0]) - obs,
                        [guess])
    return float(res.x[0])


def forecast_calendar(m, p, q, t0, years):
    """Adopters by calendar year: cumulative A at each year end and new
    adopters in the year, a_Y = A(end of Y) - A(end of Y-1)."""
    years = np.asarray(years)
    cum_end = cumulative_at(years + 1.0, m, p, q, t0)
    cum_start = cumulative_at(years.astype(float), m, p, q, t0)
    return {"year": years, "new_adopters": cum_end - cum_start,
            "cumulative_adopters": cum_end}
