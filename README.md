# Deal Desk

A form-driven dashboard for the property/wrap/investor/tax models, wired to a FastAPI backend.

## Files
- `property_strategy_model.py` — rent-strategy + cash-flow engine
- `capital_stack_model.py` — wrap economics, investor returns (IRR), depreciation shield
- `main.py` — FastAPI service exposing all four models as endpoints
- `dashboard.html` — the UI (submits values to the API; falls back to in-browser preview)
- `requirements.txt` — backend dependencies

## Two ways to plug in values

### 1. The dashboard (easiest)
Type into the form fields and click **Run**. Enter rates as whole percents (e.g. `8.5`, not `0.085`).
Each tab maps to one model. The result panel shows a color-coded verdict (Make / Watch / Loss)
plus the full line-item readout.

### 2. Directly in Python (quick one-offs)
Import the functions and pass your own numbers:

```python
from capital_stack_model import wrap_economics, investor_returns

w = wrap_economics(
    payoff=358_026, underlying_rate=0.065, underlying_orig_amount=379_000,
    underlying_orig_term_yrs=30, underlying_months_elapsed=26,
    wrap_sale_price=395_000, down_payment=35_000,   # <- your numbers here
    wrap_rate=0.085, wrap_amort_yrs=30, balloon_yrs=5,
)
print(w.monthly_spread, w.balloon_shortfall_or_surplus, w.verdict)
```

Or edit the `FITZPATRICK` / `BELLFLOWER` defaults and `SCENARIOS` at the bottom of
`property_strategy_model.py` and run `python property_strategy_model.py`.

## Run the backend locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload         # http://127.0.0.1:8000  (docs at /docs)
```
Then open `dashboard.html`, confirm the API URL, and click **Connect**. The status pill turns
green ("Connected") and every Run hits the Python engine. If the API is down, the pill stays
amber ("Offline demo") and the dashboard computes in-browser so you can still use it.

## Deploy (your stack)
- **API → Railway:** deploy the repo, start command
  `uvicorn main:app --host 0.0.0.0 --port $PORT`. Copy the Railway URL.
- **UI → Cloudflare Pages:** publish `dashboard.html`. Set the API URL field to your Railway URL.
- **Lock down CORS:** in `main.py`, replace `allow_origins=["*"]` with your Pages domain.

## Note
Modeling aid only — not tax or legal advice. Building-component bonus depreciation, installment-sale
treatment, and any passive-investor raise are professional (CPA / securities counsel) determinations.
