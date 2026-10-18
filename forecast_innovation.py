"""Step 5-7: forecast the diffusion of the Oura Ring (TIME Best Inventions 2025,
Oura Ring 4) worldwide, using p, q from the Fitbit look-alike and a Fermi
estimate of the market potential M.

Writes data/market_potential.json, data/oura_forecast.csv,
img/oura_forecast.png and img/oura_cumulative.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import calibrate_origin, cumulative_at, forecast_calendar, peak_time

with open("data/bass_params.json") as f:
    params = json.load(f)
p, q = params["nls"]["p"], params["nls"]["q"]

# --- Market potential M (Fermi estimate, every factor cited in the JSON) ---
with open("data/market_potential_inputs.json") as f:
    inp = json.load(f)
adults = inp["high_income_population_2024"]["value"] * (1 - inp["share_aged_0_14_high_income_2024"]["value"])
wearable_owners = adults * inp["wearable_ownership_rate_2024"]["value"]
oura_share = inp["oura_smart_ring_market_share"]["value"]
ring_share = inp["smart_ring_share_of_wearable_owners"]
M = {s: wearable_owners * ring_share[s] * oura_share / 1e6 for s in ["low", "base", "high"]}
print("Adults in high-income countries: %.0f M" % (adults / 1e6))
print("Wearable owners: %.0f M" % (wearable_owners / 1e6))
print("M (million rings): low %.1f, base %.1f, high %.1f" % (M["low"], M["base"], M["high"]))

# --- Align the diffusion clock with Oura's reported cumulative sales ---
ms = pd.read_csv("data/oura_milestones.csv")
t0 = calibrate_origin(ms["year_fraction"], ms["cumulative_rings_millions"], M["base"], p, q)
ms["predicted_millions"] = cumulative_at(ms["year_fraction"], M["base"], p, q, t0)
print("Calibrated diffusion origin t0 = %.2f" % t0)
print(ms[["date", "cumulative_rings_millions", "predicted_millions"]].round(2).to_string(index=False))
t_star = peak_time(p, q)
print("Peak adoption year: %.1f" % (t0 + t_star))

with open("data/market_potential.json", "w") as f:
    json.dump({"adults_high_income": adults, "wearable_owners": wearable_owners,
               "M_million": M, "t0": t0, "peak_year": t0 + t_star,
               "p": p, "q": q}, f, indent=2)

# --- Adopters by period (calendar years) ---
years = np.arange(2020, 2036)
rows = {"year": years}
for s in ["low", "base", "high"]:
    fc = forecast_calendar(M[s], p, q, t0, years)
    rows[f"new_{s}_m"] = fc["new_adopters"]
    rows[f"cumulative_{s}_m"] = fc["cumulative_adopters"]
out = pd.DataFrame(rows)
out.round(3).to_csv("data/oura_forecast.csv", index=False)
print(out[["year", "new_base_m", "cumulative_base_m"]].round(2).to_string(index=False))

colors = {"low": "#8DA0CB", "base": "#C44E52", "high": "#66C2A5"}

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(years, out["new_base_m"], color="#BBBBBB", label=f"Base (M={M['base']:.1f}M)")
for s in ["low", "high"]:
    ax.plot(years, out[f"new_{s}_m"], "o-", color=colors[s], label=f"{s.title()} (M={M[s]:.1f}M)")
ax.axvline(t0 + t_star, ls=":", color="k", lw=1)
ax.text(t0 + t_star + 0.1, ax.get_ylim()[1] * 0.92, "peak", fontsize=9)
ax.set_title("Oura Ring: forecast new adopters per year (million rings, worldwide)")
ax.set_xlabel("Year")
ax.set_ylabel("Million rings")
ax.legend()
fig.tight_layout()
fig.savefig("img/oura_forecast.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4.5))
grid = np.linspace(2020, 2036, 400)
for s in ["low", "base", "high"]:
    ax.plot(grid, cumulative_at(grid, M[s], p, q, t0), color=colors[s], label=f"{s.title()} (M={M[s]:.1f}M)")
ax.scatter(ms["year_fraction"], ms["cumulative_rings_millions"], color="k", zorder=3,
           label="Reported cumulative sales (Oura)")
ax.set_title("Oura Ring: cumulative adopters, Bass forecast vs reported")
ax.set_xlabel("Year")
ax.set_ylabel("Million rings")
ax.legend()
fig.tight_layout()
fig.savefig("img/oura_cumulative.png", dpi=150)
