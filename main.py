"""
main.py — Deal Desk API
-----------------------
FastAPI service that exposes the two model modules as JSON endpoints so the
dashboard (or any client) can submit values "by request" and get results back.

Keeps the Python models as the single source of truth — the browser never
re-implements the math when it's talking to this API.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload            # -> http://127.0.0.1:8000
    # interactive docs at http://127.0.0.1:8000/docs

Deploy (matches your stack):
    Railway: add this repo, set start command:
        uvicorn main:app --host 0.0.0.0 --port $PORT
    Then paste the Railway URL into the dashboard's "API URL" field.

Files expected alongside this one:
    property_strategy_model.py
    capital_stack_model.py
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import property_strategy_model as psm
import capital_stack_model as csm

app = FastAPI(title="Deal Desk API", version="1.0")

# Dev-open CORS so the dashboard can call from anywhere. In production,
# replace ["*"] with your Cloudflare Pages domain, e.g. ["https://deals.yoursite.com"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request bodies (mirror the model signatures)
# --------------------------------------------------------------------------- #

class RentStrategyReq(BaseModel):
    # property
    name: str = "Property"
    est_value: float
    loan_balance: float
    monthly_payment_piti: float
    monthly_hoa: float = 0.0
    bedrooms: int = 0
    monthly_principal_paydown: float = 0.0
    # strategy
    strategy_label: str = "Scenario"
    kind: Literal["whole_house", "by_room", "mid_term_furnished"] = "whole_house"
    gross_monthly_rent: float = 0.0
    vacancy_rate: float = 0.06
    maint_capex_rate: float = 0.06
    landlord_paid_utilities: float = 0.0
    extra_mgmt_cost: float = 0.0


class WrapReq(BaseModel):
    payoff: float
    underlying_rate: float
    underlying_orig_amount: float
    underlying_orig_term_yrs: int = 30
    underlying_months_elapsed: int = 0
    wrap_sale_price: float
    down_payment: float
    wrap_rate: float
    wrap_amort_yrs: int = 30
    balloon_yrs: int = 5


class InvestorReq(BaseModel):
    capital_in: float
    structure: Literal["debt", "pref_equity"] = "debt"
    hold_years: int = 3
    debt_rate: float = 0.10
    debt_paid_monthly: bool = True
    pref_rate: float = 0.08
    sponsor_split: float = 0.30
    projected_exit_equity: float = 0.0
    annual_free_cash_flow: float = 0.0


class DepreciationReq(BaseModel):
    furnishings_cost: float
    building_basis: float = 0.0
    cost_seg_pct: float = 0.0
    bonus_rate: float = 1.00
    marginal_tax_rate: float = 0.32


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/defaults")
def defaults():
    """Prefill values for the dashboard, straight from the model module."""
    return {
        "fitzpatrick": asdict(psm.FITZPATRICK),
        "bellflower": asdict(psm.BELLFLOWER),
    }


@app.post("/rent-strategy")
def rent_strategy(req: RentStrategyReq):
    prop = psm.Property(
        name=req.name, est_value=req.est_value, loan_balance=req.loan_balance,
        monthly_payment_piti=req.monthly_payment_piti, monthly_hoa=req.monthly_hoa,
        bedrooms=req.bedrooms, monthly_principal_paydown=req.monthly_principal_paydown,
    )
    strat = psm.RentStrategy(
        label=req.strategy_label, kind=req.kind,
        gross_monthly_rent=req.gross_monthly_rent, vacancy_rate=req.vacancy_rate,
        maint_capex_rate=req.maint_capex_rate,
        landlord_paid_utilities=req.landlord_paid_utilities,
        extra_mgmt_cost=req.extra_mgmt_cost,
    )
    return asdict(psm.evaluate_scenario(prop, strat))


@app.post("/wrap")
def wrap(req: WrapReq):
    return asdict(csm.wrap_economics(**req.model_dump()))


class SolveWrapReq(BaseModel):
    payoff: float
    underlying_rate: float
    underlying_orig_amount: float
    underlying_orig_term_yrs: int = 30
    underlying_months_elapsed: int = 0
    current_value: float
    hold_monthly_cashflow: float = 0.0
    balloon_yrs: int = 5
    wrap_amort_yrs: int = 30
    typical_down_pct: float = 0.10
    typical_rate: float = 0.09
    max_price_premium: float = 0.06


@app.post("/solve-wrap")
def solve_wrap(req: SolveWrapReq):
    return csm.solve_wrap(**req.model_dump())


@app.post("/investor")
def investor(req: InvestorReq):
    return asdict(csm.investor_returns(**req.model_dump()))


@app.post("/depreciation")
def depreciation(req: DepreciationReq):
    return csm.depreciation_shield(**req.model_dump())
