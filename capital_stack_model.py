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
# 1b. Wrap SOLVER — you give constraints, it returns the terms
# --------------------------------------------------------------------------- #
# The calculator above answers "are THESE terms good?". The solver answers
# "what SHOULD the terms be?" — it sweeps price / rate / down payment, keeps
# only clean-exit structures (wrap note >= payoff AND balloon covers payoff),
# then maximizes monthly spread. It also says whether wrapping beats holding.

def _clean_exit(price, down_pct, rate, payoff, u_rate, u_orig, u_term, u_elapsed,
                wrap_amort_yrs, balloon_yrs):
    """Run the wrap and return (is_clean_exit, WrapResult) for one combo."""
    r = wrap_economics(
        payoff=payoff, underlying_rate=u_rate, underlying_orig_amount=u_orig,
        underlying_orig_term_yrs=u_term, underlying_months_elapsed=u_elapsed,
        wrap_sale_price=price, down_payment=price * down_pct, wrap_rate=rate,
        wrap_amort_yrs=wrap_amort_yrs, balloon_yrs=balloon_yrs,
    )
    clean = (r.wrap_note_balance >= payoff) and (r.balloon_shortfall_or_surplus >= 0)
    return clean, r


def _min_clean_price(down_pct, rate, ceiling, payoff, u_rate, u_orig, u_term,
                     u_elapsed, wrap_amort_yrs, balloon_yrs):
    """Lowest sale price that yields a clean exit at (down_pct, rate), or None
    if even the realistic ceiling can't clear it. Both constraints rise
    monotonically with price, so we binary-search."""
    lo = payoff * 0.3
    feasible_ceiling, _ = _clean_exit(ceiling, down_pct, rate, payoff, u_rate,
                                      u_orig, u_term, u_elapsed, wrap_amort_yrs, balloon_yrs)
    if not feasible_ceiling:
        return None
    hi = ceiling
    for _ in range(44):
        mid = (lo + hi) / 2
        ok, _ = _clean_exit(mid, down_pct, rate, payoff, u_rate, u_orig, u_term,
                            u_elapsed, wrap_amort_yrs, balloon_yrs)
        if ok:
            hi = mid
        else:
            lo = mid
    return hi


def solve_wrap(
    payoff: float,
    underlying_rate: float,
    underlying_orig_amount: float,
    underlying_orig_term_yrs: int,
    underlying_months_elapsed: int,
    current_value: float,
    hold_monthly_cashflow: float,       # your monthly bleed today if you DON'T wrap (usually negative)
    balloon_yrs: int = 5,
    wrap_amort_yrs: int = 30,
    # realistic terms-buyer profile (the recommended pick):
    typical_down_pct: float = 0.10,
    typical_rate: float = 0.09,
    # how far above appraised value a terms buyer might realistically pay:
    max_price_premium: float = 0.06,
    # ranges for the "show me everything" frontier:
    down_pct_grid=(0.05, 0.08, 0.10, 0.15),
    rate_grid=(0.080, 0.090, 0.100, 0.110),
) -> dict:
    ceiling_price = current_value * (1 + max_price_premium)
    common = dict(payoff=payoff, u_rate=underlying_rate, u_orig=underlying_orig_amount,
                  u_term=underlying_orig_term_yrs, u_elapsed=underlying_months_elapsed,
                  wrap_amort_yrs=wrap_amort_yrs, balloon_yrs=balloon_yrs)
    term_m = balloon_yrs * 12

    # ---- Recommendation: the feasible terms CLOSEST to a realistic buyer ----
    # Prefer the typical profile; if being underwater breaks it, search the grid
    # for the clean-exit combo that deviates least from typical (buyer-friendliest
    # fix first). On an underwater wrap the usual fix is a SMALLER down payment,
    # which keeps more balance in the note so it clears your payoff.
    cand_downs = sorted(set(list(down_pct_grid) + [typical_down_pct]))
    cand_rates = sorted(set(list(rate_grid) + [typical_rate]))
    best_rec = None
    for dp in cand_downs:
        for rt in cand_rates:
            mp = _min_clean_price(dp, rt, ceiling_price, **common)
            if mp is None:
                continue
            deviation = abs(dp - typical_down_pct) * 4 + abs(rt - typical_rate) * 10
            if best_rec is None or deviation < best_rec[0]:
                best_rec = (deviation, dp, rt, mp)

    feasible = best_rec is not None
    recommendation = None
    per_10k_spread = None
    ceiling_spread = None
    adjusted = False

    if feasible:
        _, rec_dp, rec_rate, floor_price = best_rec
        adjusted = (abs(rec_dp - typical_down_pct) > 1e-9) or (abs(rec_rate - typical_rate) > 1e-9)
        rec_price = min(-(-floor_price // 1000) * 1000, ceiling_price)  # round UP to $1k, cap at ceiling
        _, r = _clean_exit(rec_price, rec_dp, rec_rate, **common)
        _, r10 = _clean_exit(min(rec_price + 10_000, ceiling_price), rec_dp, rec_rate, **common)
        per_10k_spread = round(r10.monthly_spread - r.monthly_spread, 2)
        _, rc = _clean_exit(ceiling_price, rec_dp, rec_rate, **common)
        ceiling_spread = round(rc.monthly_spread, 2)
        recommendation = {
            "sale_price": round(rec_price, 2),
            "down_pct": rec_dp,
            "down_payment": round(rec_price * rec_dp, 2),
            "wrap_rate": rec_rate,
            "monthly_spread": round(r.monthly_spread, 2),
            "balloon_surplus": round(r.balloon_shortfall_or_surplus, 2),
            "cash_at_close": round(rec_price * rec_dp, 2),
            "adjusted_from_typical": adjusted,
        }

    # ---- The frontier: min clean-exit price across the whole grid ----
    frontier = []
    best = None
    for dp in down_pct_grid:
        for rt in rate_grid:
            mp = _min_clean_price(dp, rt, ceiling_price, **common)
            if mp is None:
                frontier.append({"down_pct": dp, "wrap_rate": rt, "min_price": None,
                                 "monthly_spread": None, "balloon_surplus": None,
                                 "clean_exit": False})
                continue
            mp1k = min(-(-mp // 1000) * 1000, ceiling_price)
            _, rr = _clean_exit(mp1k, dp, rt, **common)
            row = {"down_pct": dp, "wrap_rate": rt, "min_price": round(mp1k, 2),
                   "monthly_spread": round(rr.monthly_spread, 2),
                   "balloon_surplus": round(rr.balloon_shortfall_or_surplus, 2),
                   "clean_exit": True}
            frontier.append(row)
            if best is None or rr.monthly_spread > best["monthly_spread"]:
                best = row

    # ---- Worth-it: wrap vs. keep holding, over the balloon horizon ----
    hold_total = hold_monthly_cashflow * term_m           # your bleed if you do nothing
    if feasible:
        wrap_total = (recommendation["cash_at_close"]
                      + recommendation["monthly_spread"] * term_m
                      + recommendation["balloon_surplus"])
        swing = wrap_total - hold_total
        thin = recommendation["monthly_spread"] < 150
        verdict = (
            f"Wrap it. Ask ${recommendation['sale_price']:,.0f} at "
            f"{recommendation['wrap_rate']*100:.1f}% with {recommendation['down_pct']*100:.0f}% down "
            f"(${recommendation['down_payment']:,.0f} at close). Clean exit, "
            f"${recommendation['monthly_spread']:,.0f}/mo spread. "
            f"Versus holding (${hold_total:,.0f} over {balloon_yrs} yrs), "
            f"that's about a ${swing:,.0f} swing in your favor."
        )
        if adjusted:
            verdict += (
                f" Note: a standard {typical_down_pct*100:.0f}% down at "
                f"{typical_rate*100:.1f}% wouldn't clear your payoff because you're underwater — "
                f"the {recommendation['down_pct']*100:.0f}% down keeps enough balance in the note "
                "to wrap what you still owe. Counterintuitive, but that's the underwater math."
            )
        if thin:
            verdict += " Spread is thin, so the real win is stopping the bleed and getting out, not the monthly income."
    else:
        wrap_total = None
        swing = None
        verdict = (
            f"Don't wrap — at least not on these numbers. No clean-exit structure "
            f"clears your ${payoff:,.0f} payoff even at a realistic price ceiling of "
            f"${ceiling_price:,.0f} (value +{max_price_premium*100:.0f}%). You're too far "
            f"underwater to wrap into a covered balloon. Hold and let equity rebuild, "
            f"pursue mid-term furnished, or wait for the value/rate picture to improve."
        )

    return {
        "feasible": feasible,
        "realistic_ceiling_price": round(ceiling_price, 2),
        "recommendation": recommendation,
        "spread_per_10k_higher_price": per_10k_spread,
        "spread_at_ceiling_price": ceiling_spread,
        "best_spread_combo": best,
        "frontier": frontier,
        "worth_it": {
            "hold_total_over_term": round(hold_total, 2),
            "wrap_total_over_term": round(wrap_total, 2) if wrap_total is not None else None,
            "swing": round(swing, 2) if swing is not None else None,
            "verdict": verdict,
        },
    }


# --------------------------------------------------------------------------- #
# 1c. Target-cash wrap solver — "I need $X at close, what terms get me there?"
# --------------------------------------------------------------------------- #
# solve_wrap() minimizes price (right when you're UNDERWATER and just need to
# clear the payoff). This one is for POSITIVE equity: you name the cash you need
# at closing, and it finds the clean-exit terms that maximize your total take
# over the balloon term — cash at close + net spread + balloon surplus — while
# landing the down payment inside your target band.

def solve_wrap_for_cash(
    payoff: float,
    underlying_rate: float,
    underlying_orig_amount: float,
    underlying_orig_term_yrs: int,
    underlying_months_elapsed: int,
    current_value: float,
    target_cash_min: float,
    target_cash_max: float,
    escrow_monthly: float = 0.0,        # T&I you still owe on the underlying
    buyer_pays_escrow: bool = True,     # does the buyer reimburse it?
    balloon_yrs: int = 5,
    wrap_amort_yrs: int = 30,
    max_price_premium: float = 0.08,    # highest ask above value a buyer accepts
    min_price_discount: float = 0.02,   # lowest ask below value you'd accept
    rate_grid=(0.080, 0.085, 0.090, 0.095, 0.100, 0.105),
    down_pct_grid=(0.05, 0.06, 0.07, 0.08, 0.10, 0.12),
    price_step: float = 5_000.0,
) -> dict:
    term_m = balloon_yrs * 12
    seller_pi = monthly_pi(underlying_orig_amount, underlying_rate, underlying_orig_term_yrs)
    und_bal_balloon = balance_after(underlying_orig_amount, underlying_rate,
                                    underlying_orig_term_yrs,
                                    underlying_months_elapsed + term_m)

    lo_price = current_value * (1 - min_price_discount)
    hi_price = current_value * (1 + max_price_premium)
    prices = []
    p = (lo_price // price_step) * price_step
    while p <= hi_price + 1:
        if p >= lo_price - 1:
            prices.append(p)
        p += price_step

    combos = []
    for price in prices:
        for dp in down_pct_grid:
            cash = price * dp
            if not (target_cash_min <= cash <= target_cash_max):
                continue
            note = price - cash
            if note < payoff:                      # must clear what you owe
                continue
            for rt in rate_grid:
                buyer_pi = monthly_pi(note, rt, wrap_amort_yrs)
                spread = buyer_pi - seller_pi
                net_spread = spread if buyer_pays_escrow else spread - escrow_monthly
                wrap_bal = balance_after(note, rt, wrap_amort_yrs, term_m)
                balloon = wrap_bal - und_bal_balloon
                if balloon < 0:                    # balloon must cover your payoff
                    continue
                total = cash + net_spread * term_m + balloon
                combos.append({
                    "sale_price": round(price, 2),
                    "down_pct": round(dp, 4),
                    "cash_at_close": round(cash, 2),
                    "wrap_rate": rt,
                    "wrap_note": round(note, 2),
                    "buyer_pi": round(buyer_pi, 2),
                    "buyer_all_in": round(buyer_pi + (escrow_monthly if buyer_pays_escrow else 0), 2),
                    "gross_spread": round(spread, 2),
                    "net_spread": round(net_spread, 2),
                    "spread_over_term": round(net_spread * term_m, 2),
                    "balloon_surplus": round(balloon, 2),
                    "total_5yr": round(total, 2),
                })

    combos.sort(key=lambda c: c["total_5yr"], reverse=True)
    best = combos[0] if combos else None
    sale_net = current_value * 0.92 - payoff       # sell outright, 8% costs

    if best is None:
        verdict = (f"No clean-exit structure delivers ${target_cash_min:,.0f}–${target_cash_max:,.0f} "
                   f"at close within a ${lo_price:,.0f}–${hi_price:,.0f} price range. Widen the price "
                   f"ceiling, lower the cash target, or the equity isn't there yet.")
        escrow_cost = None
    else:
        escrow_cost = round(escrow_monthly * term_m, 2)
        verdict = (
            f"Ask ${best['sale_price']:,.0f} at {best['wrap_rate']*100:.1f}% with "
            f"{best['down_pct']*100:.0f}% down — ${best['cash_at_close']:,.0f} in your pocket at close. "
            f"Buyer's all-in payment ${best['buyer_all_in']:,.0f}/mo, your net spread "
            f"${best['net_spread']:,.0f}/mo, balloon surplus ${best['balloon_surplus']:,.0f}. "
            f"Five-year total ${best['total_5yr']:,.0f} vs. ${sale_net:,.0f} selling outright today."
        )
        if not buyer_pays_escrow:
            verdict += (f" Note: you're absorbing ${escrow_monthly:,.0f}/mo of escrow — "
                        f"${escrow_cost:,.0f} over the term. Get the buyer to reimburse T&I and "
                        f"that goes straight to your bottom line.")

    return {
        "feasible": best is not None,
        "price_range_tested": [round(lo_price, 2), round(hi_price, 2)],
        "seller_underlying_pi": round(seller_pi, 2),
        "escrow_monthly": round(escrow_monthly, 2),
        "escrow_cost_over_term": escrow_cost,
        "buyer_pays_escrow": buyer_pays_escrow,
        "best": best,
        "alternatives": combos[1:6],
        "sell_outright_net": round(sale_net, 2),
        "verdict": verdict,
    }


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
# 3. Mid-term furnished SOLVER — works backward to the rent you'd need
# --------------------------------------------------------------------------- #
# You give the carrying cost and the furnished operating reality; it solves for
# the monthly rent required to hit each of three thresholds, shows furnishing
# both amortized and sunk, and judges each required rent against real market
# rates (a comp you enter, or a benchmark range).

def solve_midterm(
    monthly_payment_piti: float,
    monthly_hoa: float = 0.0,
    # furnished operating reality:
    utilities: float = 350.0,          # landlord-paid, all-inclusive
    internet: float = 75.0,
    cleaning_monthly: float = 60.0,    # turnover cleaning, spread monthly
    occupancy: float = 0.85,           # effective, after gap days between contracts
    mgmt_pct: float = 0.10,            # co-host share of collected rent (0 if self-managed)
    maint_pct: float = 0.05,           # maintenance + capex reserve, % of collected
    # furnishing:
    furnishing_cost: float = 12_000.0,
    furnishing_payback_months: int = 24,
    # thresholds:
    profit_target: float = 300.0,      # your desired monthly profit
    # realism yardsticks:
    market_comp: float = 0.0,          # a furnished rate YOU found (0 = none)
    benchmark_low: float = 3_200.0,    # area furnished range (suburban 3bd)
    benchmark_high: float = 4_200.0,
) -> dict:
    # Fixed monthly costs that don't scale with revenue:
    fixed = utilities + internet + monthly_hoa + cleaning_monthly + monthly_payment_piti
    # Revenue efficiency: fraction of asking rent that survives occupancy + %-of-rev costs
    eff = occupancy * (1 - mgmt_pct - maint_pct)

    def required_rent(target):
        return (target + fixed) / eff if eff > 0 else float("inf")

    furnishing_monthly = furnishing_cost / furnishing_payback_months if furnishing_payback_months else 0.0

    # Three thresholds, each shown both ways (furnishing sunk vs amortized):
    thresholds = {
        "stop_the_bleed":        {"sunk": required_rent(0.0),
                                  "amortized": required_rent(furnishing_monthly)},
        "cover_target_profit":   {"sunk": required_rent(profit_target),
                                  "amortized": required_rent(profit_target + furnishing_monthly)},
    }
    # (break-even + furnishing payback == stop_the_bleed "amortized")

    # Realism: pick the yardstick and label each required rent
    yardstick = market_comp if market_comp > 0 else benchmark_high

    def realism(req):
        if req <= benchmark_low:
            return "comfortably within market"
        if req <= benchmark_high:
            return "achievable at the top of the market"
        return "above realistic market"

    # What actually happens at an achievable rent (the yardstick):
    collected = yardstick * occupancy
    surplus_at_yardstick = collected * (1 - mgmt_pct - maint_pct) - fixed  # furnishing sunk
    months_to_recoup = (furnishing_cost / surplus_at_yardstick
                        if surplus_at_yardstick > 0 else None)

    # Plain-English verdict driven by the break-even number
    be = thresholds["stop_the_bleed"]["sunk"]
    src = f"your comp of ${market_comp:,.0f}" if market_comp > 0 else f"the ${benchmark_high:,.0f} top of market"
    if be <= benchmark_low:
        verdict = (f"Works. To stop the bleed you need ${be:,.0f}/mo furnished — inside the "
                   f"${benchmark_low:,.0f}–${benchmark_high:,.0f} market range, so there's room above it "
                   f"for profit. Mid-term is a real play here.")
    elif be <= yardstick:
        verdict = (f"Works, but tight. Break-even needs ${be:,.0f}/mo furnished, which you only clear "
                   f"at {src}. It pencils if you self-manage and stay near full occupancy — thin margin otherwise.")
    else:
        verdict = (f"Doesn't pencil as-is. Break-even needs ${be:,.0f}/mo furnished, above {src}. "
                   f"The carry (mortgage + HOA + landlord-paid utilities) is too heavy for the furnished "
                   f"rate this submarket supports. Drop the co-host, push occupancy, or it's not the move.")

    return {
        "fixed_monthly_costs": round(fixed, 2),
        "revenue_efficiency": round(eff, 4),
        "furnishing_monthly": round(furnishing_monthly, 2),
        "required_rent": {
            "stop_the_bleed_sunk": round(thresholds["stop_the_bleed"]["sunk"], 2),
            "stop_the_bleed_plus_furnishing": round(thresholds["stop_the_bleed"]["amortized"], 2),
            "target_profit_sunk": round(thresholds["cover_target_profit"]["sunk"], 2),
            "target_profit_plus_furnishing": round(thresholds["cover_target_profit"]["amortized"], 2),
        },
        "realism": {
            "stop_the_bleed": realism(thresholds["stop_the_bleed"]["sunk"]),
            "stop_the_bleed_plus_furnishing": realism(thresholds["stop_the_bleed"]["amortized"]),
            "target_profit": realism(thresholds["cover_target_profit"]["sunk"]),
            "benchmark_low": benchmark_low,
            "benchmark_high": benchmark_high,
            "yardstick": yardstick,
        },
        "at_yardstick_rent": {
            "rent": round(yardstick, 2),
            "monthly_surplus": round(surplus_at_yardstick, 2),
            "months_to_recoup_furnishing": (round(months_to_recoup, 1)
                                            if months_to_recoup is not None else None),
        },
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# 4. Depreciation shield (rough, first-year)
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
