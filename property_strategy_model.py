"""
property_strategy_model.py
---------------------------
Cash-flow + equity comparator for hold/sell/rent-strategy decisions.

Built for the 739 Fitzpatrick vs 5333 Bellflower decision, but fully
parameterized so you can re-run it as real numbers firm up (actual HOA
dues, true CMA value, real room rates, real furnished mid-term rates).

Design:
  - `evaluate_scenario()` is a PURE function (no I/O) -> easy to unit test,
    easy to call from anything.
  - `Property` + `RentStrategy` dataclasses model the inputs.
  - A tiny FastAPI app at the bottom exposes it as a microservice so it
    can live next to your other deal tools (run: `uvicorn property_strategy_model:app`).

Run directly for a printed comparison of both properties:
    python property_strategy_model.py
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

@dataclass
class Property:
    name: str
    est_value: float
    loan_balance: float
    monthly_payment_piti: float   # principal+interest+taxes+insurance (NOT HOA)
    monthly_hoa: float = 0.0
    bedrooms: int = 0
    # Rough monthly principal portion of the PITI payment (equity you build).
    # Approximate; pull the real number from your amortization schedule.
    monthly_principal_paydown: float = 0.0


@dataclass
class RentStrategy:
    label: str
    kind: Literal["whole_house", "by_room", "mid_term_furnished"]
    gross_monthly_rent: float          # total collected if fully occupied
    vacancy_rate: float                # 0.06 = 6%
    maint_capex_rate: float            # 0.05 = 5% of gross, reserves
    landlord_paid_utilities: float = 0.0   # $/mo (0 if tenant pays)
    extra_mgmt_cost: float = 0.0           # $/mo cleaning / co-host / common areas


# --------------------------------------------------------------------------- #
# Core math (pure)
# --------------------------------------------------------------------------- #

SELLING_COST_RATE = 0.08  # agent + closing + concessions planning figure


@dataclass
class ScenarioResult:
    property_name: str
    strategy: str
    equity: float
    gross_rent: float
    vacancy_loss: float
    maint_capex: float
    utilities: float
    mgmt: float
    hoa: float
    debt_service: float
    monthly_cash_flow: float
    annual_cash_flow: float
    # "economic" cash flow credits the principal you pay down each month
    monthly_economic: float
    net_if_sold_today: float
    verdict: str


def evaluate_scenario(p: Property, s: RentStrategy) -> ScenarioResult:
    equity = p.est_value - p.loan_balance

    vacancy_loss = s.gross_monthly_rent * s.vacancy_rate
    maint_capex = s.gross_monthly_rent * s.maint_capex_rate
    debt_service = p.monthly_payment_piti

    monthly_cf = (
        s.gross_monthly_rent
        - vacancy_loss
        - maint_capex
        - s.landlord_paid_utilities
        - s.extra_mgmt_cost
        - p.monthly_hoa
        - debt_service
    )
    monthly_economic = monthly_cf + p.monthly_principal_paydown
    net_if_sold = p.est_value * (1 - SELLING_COST_RATE) - p.loan_balance

    if monthly_cf >= 50:
        verdict = "Cash-flow positive."
    elif monthly_cf >= -150:
        verdict = "Roughly breakeven (cash); equity build may make it economically neutral."
    elif monthly_cf >= -600:
        verdict = "Negative carry — sustainable only if you can subsidize it."
    else:
        verdict = "Heavily negative — bleeds cash; not a viable hold as-is."

    return ScenarioResult(
        property_name=p.name,
        strategy=s.label,
        equity=round(equity, 2),
        gross_rent=round(s.gross_monthly_rent, 2),
        vacancy_loss=round(-vacancy_loss, 2),
        maint_capex=round(-maint_capex, 2),
        utilities=round(-s.landlord_paid_utilities, 2),
        mgmt=round(-s.extra_mgmt_cost, 2),
        hoa=round(-p.monthly_hoa, 2),
        debt_service=round(-debt_service, 2),
        monthly_cash_flow=round(monthly_cf, 2),
        annual_cash_flow=round(monthly_cf * 12, 2),
        monthly_economic=round(monthly_economic, 2),
        net_if_sold_today=round(net_if_sold, 2),
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# The two real properties + the scenarios discussed
# --------------------------------------------------------------------------- #

FITZPATRICK = Property(
    name="739 Fitzpatrick",
    est_value=240_000,
    loan_balance=216_570.08,
    monthly_payment_piti=1_883.00,
    monthly_hoa=0.0,
    bedrooms=2,
    monthly_principal_paydown=300.0,   # estimate; replace with amort figure
)

BELLFLOWER = Property(
    name="5333 Bellflower",
    est_value=344_000,                 # conservative; comps hint ~$375k -> re-run both
    loan_balance=358_026.45,
    monthly_payment_piti=2_668.74,
    monthly_hoa=200.0,                  # ESTIMATE — confirm Tulip Hills dues
    bedrooms=3,
    monthly_principal_paydown=420.0,   # estimate; replace with amort figure
)

SCENARIOS = {
    "739 Fitzpatrick": [
        RentStrategy("Whole-house lease", "whole_house",
                     gross_monthly_rent=1_850, vacancy_rate=0.06,
                     maint_capex_rate=0.08, landlord_paid_utilities=0),
        RentStrategy("Rent-by-room (2 x $750)", "by_room",
                     gross_monthly_rent=1_500, vacancy_rate=0.12,
                     maint_capex_rate=0.08, landlord_paid_utilities=250,
                     extra_mgmt_cost=0),
    ],
    "5333 Bellflower": [
        RentStrategy("Whole-house lease", "whole_house",
                     gross_monthly_rent=2_250, vacancy_rate=0.06,
                     maint_capex_rate=0.05, landlord_paid_utilities=0),
        RentStrategy("Rent-by-room (3 x $900)", "by_room",
                     gross_monthly_rent=2_700, vacancy_rate=0.12,
                     maint_capex_rate=0.08, landlord_paid_utilities=300,
                     extra_mgmt_cost=150),
        RentStrategy("Mid-term furnished (travel nurse, $3,200)", "mid_term_furnished",
                     gross_monthly_rent=3_200, vacancy_rate=0.15,
                     maint_capex_rate=0.10, landlord_paid_utilities=350,
                     extra_mgmt_cost=0),
        RentStrategy("Mid-term furnished (strong rate, $4,100)", "mid_term_furnished",
                     gross_monthly_rent=4_100, vacancy_rate=0.15,
                     maint_capex_rate=0.10, landlord_paid_utilities=350,
                     extra_mgmt_cost=0),
    ],
}


def _print_block(p: Property):
    print(f"\n{'='*70}\n{p.name}  |  equity: ${p.est_value - p.loan_balance:,.0f}  "
          f"|  net if sold today: ${p.est_value*(1-SELLING_COST_RATE)-p.loan_balance:,.0f}")
    print('='*70)
    for s in SCENARIOS[p.name]:
        r = evaluate_scenario(p, s)
        print(f"\n  {r.strategy}")
        print(f"    monthly cash flow : ${r.monthly_cash_flow:>10,.2f}   "
              f"(annual ${r.annual_cash_flow:,.0f})")
        print(f"    + principal build : ${r.monthly_economic:>10,.2f}  (economic)")
        print(f"    verdict           : {r.verdict}")


if __name__ == "__main__":
    for prop in (FITZPATRICK, BELLFLOWER):
        _print_block(prop)
    print("\nNote: all rents, HOA, and utility figures are estimates. "
          "Replace with your real numbers and re-run.\n")


# --------------------------------------------------------------------------- #
# Optional: expose as a microservice (pip install fastapi uvicorn)
#   uvicorn property_strategy_model:app --reload
#   POST /evaluate  with {"property": {...}, "strategy": {...}}
# --------------------------------------------------------------------------- #
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Property Strategy Model")

    class _Prop(BaseModel):
        name: str
        est_value: float
        loan_balance: float
        monthly_payment_piti: float
        monthly_hoa: float = 0.0
        bedrooms: int = 0
        monthly_principal_paydown: float = 0.0

    class _Strat(BaseModel):
        label: str
        kind: Literal["whole_house", "by_room", "mid_term_furnished"]
        gross_monthly_rent: float
        vacancy_rate: float
        maint_capex_rate: float
        landlord_paid_utilities: float = 0.0
        extra_mgmt_cost: float = 0.0

    class _Req(BaseModel):
        property: _Prop
        strategy: _Strat

    @app.post("/evaluate")
    def evaluate(req: _Req):
        p = Property(**req.property.model_dump())
        s = RentStrategy(**req.strategy.model_dump())
        return asdict(evaluate_scenario(p, s))

except ImportError:
    # FastAPI not installed; the pure model + CLI still work fine.
    pass
