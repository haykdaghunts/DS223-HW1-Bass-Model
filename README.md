# DS 223 HW1 — Bass Model: Oura Ring 4

Innovation diffusion analysis of a product from **TIME's Best Inventions 2025** list, the
[Oura Ring 4](https://time.com/collections/best-inventions-2025/7318554/oura-ring-4/) smart ring.
Its look-alike is the **Fitbit** fitness tracker (2011–2019).

**Main results**

| | Value |
|---|---|
| Look-alike Bass parameters (Fitbit, NLS) | p = 0.0087, q = 0.670, M = 122M devices |
| Predicted Fitbit peak | t* = 6.4 → 2016 (actual: 2016) |
| Oura market potential M (Fermi, worldwide) | 18.2M rings (low 9.1M / high 36.4M) |
| Oura check against reported sales | model 5.4M vs reported 5.5M (Sep 2025) |
| Oura peak adoption | ≈ 2026–2027, ~3M rings/year (base) |

The full report is [`report/report.pdf`](report/report.pdf). The notebook source is [`bass_analysis.ipynb`](bass_analysis.ipynb).

## Structure

- `bass_analysis.ipynb` — main notebook covering steps 1–7 of the assignment, with narrative and citations.
- `helper_functions.py` — Bass model functions `bass_f`, `bass_F` (the slide formulas), NLS estimation (slides' Method 1), Bass OLS estimation (slides 13–14), peak time `t*`, origin calibration and calendar-year forecast.
- `fit_lookalike.py` — fits the Bass model to Fitbit data and writes `data/bass_params.json` and the Fitbit plots.
- `forecast_innovation.py` — Fermi estimate of M, forecast of Oura adopters by year, writes `data/oura_forecast.csv` and the Oura plots.
- `data/`
  - `fitbit_devices_sold.csv` — Fitbit annual devices sold (thousands), with a source per row (SEC 10-K / S-1).
  - `oura_milestones.csv` — Oura's reported cumulative ring sales, with sources.
  - `market_potential_inputs.json` — cited inputs for the Fermi estimate of M.
  - `bass_params.json`, `market_potential.json`, `oura_forecast.csv` — generated outputs.
- `img/`
  - `fitbit_sales.png` — Fitbit devices sold per year.
  - `fitbit_fit.png` — Fitbit actual vs Bass-fitted sales (NLS and OLS).
  - `oura_cumulative.png` — Oura cumulative adopters forecast vs reported sales.
  - `oura_forecast.png` — Oura new adopters per year, low/base/high M.
- `report/`
  - `report.pdf` — final report (PDF export of the notebook).
  - `report_source.md` — Markdown source of the report.
- `references.md` — full reference list.

## Setup and usage

```bash
pip install -r requirements.txt        # or: conda activate <env> && pip install -r requirements.txt
python fit_lookalike.py                # step 3-4: estimate p, q, M on Fitbit
python forecast_innovation.py          # step 5-7: Oura forecast
jupyter nbconvert --to notebook --execute --inplace bass_analysis.ipynb
```

## Notes on the method

The model is the lecture's Bass model: $f(t)/(1-F(t)) = p + qF(t)$, $a(t)=Mf(t)$, $A(t)=MF(t)$,
with $t = 1, \dots, n$. On the slides' smartphone dataset the Python NLS port gives
M = 17,457, p = 0.00533, q = 0.366, matching the slides' R `nls()` output (17,450 / 0.00533 / 0.366).
