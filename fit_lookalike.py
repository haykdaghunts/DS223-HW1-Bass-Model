"""Step 3-4: estimate Bass parameters (p, q, M) on the look-alike innovation,
Fitbit fitness trackers (devices sold, 2011-2019).

Writes data/bass_params.json and img/fitbit_sales.png, img/fitbit_fit.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import fit_bass_ols, fit_nls, peak_time

df = pd.read_csv("data/fitbit_devices_sold.csv")
years = df["year"].values
sales = df["devices_sold_thousands"].values / 1000  # million units

nls = fit_nls(sales)
ols = fit_bass_ols(sales)
t_star = peak_time(nls["p"], nls["q"])

print("Method 1 - NLS:  m = %.2f M (se %.2f), p = %.5f (se %.5f), q = %.4f (se %.4f), R2 = %.3f"
      % (nls["m"], nls["se_m"], nls["p"], nls["se_p"], nls["q"], nls["se_q"], nls["r2"]))
print("Method 2 - OLS:  m = %.2f M, p = %.5f, q = %.4f, R2 = %.3f"
      % (ols["m"], ols["p"], ols["q"], ols["r2"]))
print("Peak t* = %.2f -> year %.1f (actual peak %d)"
      % (t_star, years[0] - 1 + t_star, years[np.argmax(sales)]))

params = {
    "lookalike": "Fitbit fitness trackers (devices sold, million units)",
    "period": f"{years[0]}-{years[-1]}",
    "nls": {k: float(nls[k]) for k in ["m", "p", "q", "se_m", "se_p", "se_q", "sigma", "r2"]},
    "ols": {k: float(ols[k]) for k in ["b0", "b1", "b2", "m", "p", "q", "r2"]},
    "peak_t_star": float(t_star),
    "peak_year_predicted": float(years[0] - 1 + t_star),
    "peak_year_actual": int(years[np.argmax(sales)]),
}
with open("data/bass_params.json", "w") as f:
    json.dump(params, f, indent=2)

# Historical sales
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(years, sales, color="#4C72B0")
ax.set_title("Fitbit devices sold per year (million units)")
ax.set_xlabel("Year")
ax.set_ylabel("Million units")
fig.tight_layout()
fig.savefig("img/fitbit_sales.png", dpi=150)

# Actual vs fitted
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(years, sales, color="#BBBBBB", label="Actual")
ax.plot(years, nls["fitted"], "o-", color="#C44E52",
        label=f"Bass NLS (p={nls['p']:.4f}, q={nls['q']:.3f}, M={nls['m']:.0f}M)")
ax.plot(years, ols["fitted"], "s--", color="#4C72B0",
        label=f"Bass OLS (p={ols['p']:.4f}, q={ols['q']:.3f}, M={ols['m']:.0f}M)")
ax.set_title("Fitbit: actual vs Bass-fitted sales")
ax.set_xlabel("Year")
ax.set_ylabel("Million units")
ax.legend()
fig.tight_layout()
fig.savefig("img/fitbit_fit.png", dpi=150)
