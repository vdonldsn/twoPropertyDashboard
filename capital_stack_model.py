"""
capital_stack_model.py
----------------------
Models the two "bring outside money in" paths for an underwater, near-breakeven
asset (built around 5333 Bellflower, but fully parameterized):

  1. wrap_economics()      -> wrap-around / subject-to seller-finance exit
  2. investor_returns()    -> investor capital stack (debt OR preferred equity),
                              with IRR / cash-on-cash / equity multiple
  3. depreciation_shield() -> rough first-year paper-loss estimate (furnishings
                              at 100% bonus + optional cost-seg on the building)

Pure functions, no I/O -> unit-testable and callable from a FastAPI service
(same pattern as property_strategy_model.py). Run directly for a demo:
    python capital_stack_model.py

NOTE: every figure below is an ESTIMATE. Replace with your real payoff, CMA,
furnished comps, and your CPA's depreciation numbers before relying on output.
This is a modeling aid, not tax or legal advice.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


# --------------------------------------------------------------------------- #
# Small finance helpers
# --------------------------------------------------------------------------- #

def monthly_pi(principal: float, annual_rate: float, years: int) -> float:
    """Standard amortizing monthly principal+interest payment."""
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def balance_after(principal: float, annual_rate: float, years: int, months_elapsed: int) -> float:
    """Remaining balance on an amortizing loan after `months_elapsed` payments."""
    r = annual_rate / 12
    pmt = monthly_pi(principal, annual_rate, years)
    bal = principal
    for _ in range(months_elapsed):
        interest = bal * r
        bal -= (pmt - interest)
    return max(bal, 0.0)


def npv(rate: float, cashflows: list[float]) -> float:
    """NPV with cashflows[0] at t=0 (per-period rate)."""
    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))


def irr(cashflows: list[float], lo: float = -0.9, hi: float = 5.0) -> float | None:
    """Bisection IRR (per-period). Returns None if no sign change / no root."""
    f_lo, f_hi = npv(lo, cashflows), npv(hi, cashflows)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < 1e-6:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------- #
# 1. Wrap-around / subject-to economics
# --------------------------------------------------------------------------- #

@dataclass
class WrapResult:
    down_payment: float
    wrap_note_balance: float
    buyer_monthly_pi: float
    seller_underlying_pi: float
    monthly_spread: float
    balloon_month: int
    wrap_balance_at_balloon: float
    underlying_balance_at_balloon: float
    balloon_shortfall_or_surplus: float   # positive = surplus to seller
    verdict: str


def wrap_economics(
    payoff: float,
    underlying_rate: float,
    underlying_orig_amount: float,
    underlying_orig_term_yrs: int,
    underlying_months_elapsed: int,
    wrap_sale_price: float,
    down_payment: float,
    wrap_rate: float,
    wrap_amort_yrs: int,
    balloon_yrs: int,
) -> WrapResult:
    wrap_note = wrap_sale_price - down_payment
    buyer_pi = monthly_pi(wrap_note, wrap_rate, wrap_amort_yrs)
    seller_pi = monthly_pi(underlying_orig_amount, underlying_rate, underlying_orig_term_yrs)
    spread = buyer_pi - seller_pi

    balloon_m = balloon_yrs * 12
    wrap_bal_balloon = balance_after(wrap_note, wrap_rate, wrap_amort_yrs, balloon_m)
    und_bal_balloon = balance_after(
        underlying_orig_amount, underlying_rate, underlying_orig_term_yrs,
        underlying_months_elapsed + balloon_m,
    )
    balloon_delta = wrap_bal_balloon - und_bal_balloon

    if wrap_note < payoff:
        verdict = ("NEGATIVE-EQUITY WRAP: wrap note is below your payoff. You'd "
                   "carry a shortfall — raise the price or the down payment.")
    elif balloon_delta >= 0:
        verdict = "Workable: monthly spread positive and balloon covers your payoff."
    else:
        verdict = (f"Monthly spread positive, but you're ~${-balloon_delta:,.0f} short "
                   "at balloon — plan to feed that gap or extend the term.")

    return WrapResult(
        down_payment=round(down_payment, 2),
        wrap_note_balance=round(wrap_note, 2),
        buyer_monthly_pi=round(buyer_pi, 2),
        seller_underlying_pi=round(seller_pi, 2),
        monthly_spread=round(spread, 2),
        balloon_month=balloon_m,
        wrap_balance_at_balloon=round(wrap_bal_balloon, 2),
        underlying_balance_at_balloon=round(und_bal_balloon, 2),
        balloon_shortfall_or_surplus=round(balloon_delta, 2),
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# 2. Investor capital stack
# --------------------------------------------------------------------------- #

@dataclass
class InvestorResult:
    structure: str
    capital_in: float
    hold_years: int
    total_distributions: float
    profit: float
    equity_multiple: float
    cash_on_cash_annual: float
    irr_annual: float | None
    verdict: str


def investor_returns(
    capital_in: float,
    structure: Literal["debt", "pref_equity"],
    hold_years: int,
    # debt inputs:
    debt_rate: float = 0.10,
    debt_paid_monthly: bool = True,
    # pref-equity inputs:
    pref_rate: float = 0.08,
    sponsor_split: float = 0.30,            # sponsor share of upside after pref+return of capital
    projected_exit_equity: float = 0.0,     # net sale/refi proceeds attributable to the deal
    annual_free_cash_flow: float = 0.0,     # distributable cash flow per year (often ~0 here)
) -> InvestorResult:
    """
    Builds an annual cashflow series from the INVESTOR's perspective
    (t=0 outflow = -capital_in) and computes IRR / equity multiple / CoC.
    """
    flows = [-capital_in]

    if structure == "debt":
        annual_interest = capital_in * debt_rate
        for yr in range(1, hold_years + 1):
            cf = annual_interest if debt_paid_monthly else 0.0
            if yr == hold_years:
                accrued = 0.0 if debt_paid_monthly else annual_interest * hold_years
                cf += capital_in + accrued  # principal back + any accrued interest
            flows.append(cf)
        total_dist = sum(flows[1:])
        coc = debt_rate if debt_paid_monthly else 0.0

    else:  # pref_equity
        pref_accrued = 0.0
        for yr in range(1, hold_years + 1):
            pref_accrued += capital_in * pref_rate
            cf = min(annual_free_cash_flow, capital_in * pref_rate)  # current pref if cash allows
            pref_accrued -= cf
            if yr == hold_years:
                # exit waterfall: return of capital -> unpaid pref -> split of remainder
                remaining = projected_exit_equity
                ret_cap = min(remaining, capital_in);            remaining -= ret_cap
                pay_pref = min(remaining, pref_accrued);         remaining -= pay_pref
                upside = remaining * (1 - sponsor_split) if remaining > 0 else 0.0
                cf += ret_cap + pay_pref + upside
            flows.append(cf)
        total_dist = sum(flows[1:])
        coc = (annual_free_cash_flow / capital_in) if capital_in else 0.0

    profit = total_dist - capital_in
    em = (total_dist / capital_in) if capital_in else 0.0
    deal_irr = irr(flows)

    if profit < 0:
        verdict = "LOSES investor money as modeled — exit equity too thin to cover capital."
    elif deal_irr is None:
        verdict = "No clean IRR — check the cashflow assumptions."
    elif deal_irr < 0.06:
        verdict = f"Weak: {deal_irr*100:,.1f}% IRR. Hard to justify vs. a plain note."
    else:
        verdict = f"Returns capital with {deal_irr*100:,.1f}% IRR, {em:,.2f}x equity multiple."

    return InvestorResult(
        structure=structure,
        capital_in=round(capital_in, 2),
        hold_years=hold_years,
        total_distributions=round(total_dist, 2),
        profit=round(profit, 2),
        equity_multiple=round(em, 3),
        cash_on_cash_annual=round(coc, 4),
        irr_annual=round(deal_irr, 4) if deal_irr is not None else None,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# 3. Depreciation shield (rough, first-year)
# --------------------------------------------------------------------------- #

def depreciation_shield(
    furnishings_cost: float,
    building_basis: float = 0.0,
    cost_seg_pct: float = 0.0,      # % of building basis reclassified to <=15yr property
    bonus_rate: float = 1.00,       # 100% under OBBBA for eligible, post-1/19/2025 property
    marginal_tax_rate: float = 0.32,
) -> dict:
    """
    Very rough Year-1 paper loss + tax deferral estimate. Furnishings (5-yr, new
    to taxpayer, placed in service 2026) generally get 100% bonus. Building
    cost-seg eligibility for BONUS depends on acquisition/placed-in-service
    timing -> CPA call. This is a planning sketch only.
    """
    furnishings_deduction = furnishings_cost * bonus_rate
    building_bonus_deduction = building_basis * cost_seg_pct * bonus_rate
    year1_paper_loss = furnishings_deduction + building_bonus_deduction
    est_tax_deferred = year1_paper_loss * marginal_tax_rate
    return {
        "year1_paper_loss": round(year1_paper_loss, 2),
        "est_tax_deferred": round(est_tax_deferred, 2),
        "note": ("Timing benefit, not permanent — recaptured at sale (up to 25% "
                 "on 1250, ordinary on 1245, +possible 3.8% NIIT). Passive-loss "
                 "rules may limit current use. Confirm with CPA."),
    }


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=" * 72)
    print("WRAP / SUBJECT-TO EXIT  (illustrative Bellflower numbers)")
    print("=" * 72)
    w = wrap_economics(
        payoff=358_026,
        underlying_rate=0.065,
        underlying_orig_amount=379_000,
        underlying_orig_term_yrs=30,
        underlying_months_elapsed=26,       # ~ Apr 2024 -> mid 2026
        wrap_sale_price=380_000,
        down_payment=25_000,
        wrap_rate=0.085,
        wrap_amort_yrs=30,
        balloon_yrs=5,
    )
    for k, v in w.__dict__.items():
        print(f"  {k:32}: {v}")

    print("\n" + "=" * 72)
    print("INVESTOR STACK  (raise $40k to furnish + fund carry reserve)")
    print("=" * 72)
    print("\n-- Option A: private note (debt), 10%, interest paid monthly, 3yr --")
    a = investor_returns(40_000, "debt", hold_years=3, debt_rate=0.10, debt_paid_monthly=True)
    for k, v in a.__dict__.items():
        print(f"  {k:24}: {v}")

    print("\n-- Option B: pref equity, 8% pref, 30% sponsor split, 4yr hold --")
    print("   (exit equity assumed thin because asset is underwater today)")
    b = investor_returns(
        40_000, "pref_equity", hold_years=4,
        pref_rate=0.08, sponsor_split=0.30,
        projected_exit_equity=35_000,   # net proceeds attributable to deal at exit
        annual_free_cash_flow=0,        # ~breakeven, little to distribute early
    )
    for k, v in b.__dict__.items():
        print(f"  {k:24}: {v}")

    print("\n" + "=" * 72)
    print("DEPRECIATION SHIELD  (rough Year-1)")
    print("=" * 72)
    d = depreciation_shield(furnishings_cost=12_000, building_basis=0, cost_seg_pct=0.0,
                            marginal_tax_rate=0.32)
    for k, v in d.items():
        print(f"  {k}: {v}")

    print("\nReplace all inputs with real payoff, CMA, furnished comps, and CPA "
          "depreciation figures before relying on any of this.\n")
