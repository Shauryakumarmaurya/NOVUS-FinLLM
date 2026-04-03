import time
from core.agent_base_v3 import AuditTrail
from .agent_utils import _fget, _reverse_dcf

class ForensicQuantV3:
    agent_name = "forensic_quant"

    def execute(self, ticker: str, financial_tables: dict, **kwargs) -> AuditTrail:
        start = time.time()
        
        # Support both legacy internal keys and direct Screener.in keys
        pl = financial_tables.get("Profit & Loss", financial_tables.get("profit_loss", {}))
        bs = financial_tables.get("Balance Sheet", financial_tables.get("balance_sheet", {}))
        cf = financial_tables.get("Cash Flows", financial_tables.get("cash_flow", {}))
        
        findings = {}
        data_gaps = []
        flags = []

        years = sorted(pl.keys())
        if not years:
            data_gaps.append("No P&L data available")
            return self._build_trail(ticker, findings, data_gaps, flags, start)

        latest = years[-1]
        latest_pl = pl.get(latest, {})
        latest_bs = bs.get(latest, {})
        latest_cf = cf.get(latest, {})

        # ── Profitability ──
        revenue = _fget(latest_pl, "Revenue", "Sales+", "Net Sales", "Revenue from Operations")
        ebit = _fget(latest_pl, "EBIT", "Operating Profit")
        pat = _fget(latest_pl, "Net Profit", "PAT", "Profit after tax")
        total_assets = _fget(latest_bs, "Total Assets")
        total_equity = _fget(latest_bs, "Shareholders Funds", "Total Equity", "Equity")
        total_debt = _fget(latest_bs, "Borrowings", "Total Debt", "Long Term Borrowings", default=0)
        cash = _fget(latest_bs, "Cash Equivalents", "Cash and Bank", "Cash", default=0)

        if revenue and pat and total_assets and total_equity:
            margin = round(pat / revenue, 4)
            turnover = round(revenue / total_assets, 4)
            multiplier = round(total_assets / total_equity, 4)
            roe = round(margin * turnover * multiplier, 4)
            findings["dupont"] = {
                "roe": roe, "net_margin": margin,
                "asset_turnover": turnover, "equity_multiplier": multiplier,
                "primary_driver": "margin" if margin > 0.15 else ("leverage" if multiplier > 2.5 else "turnover"),
            }
        else:
            data_gaps.append("Insufficient data for DuPont decomposition")

        # ── ROIC ──
        if ebit and total_equity is not None and total_debt is not None:
            nopat = ebit * 0.75
            invested_capital = total_equity + total_debt - (cash or 0)
            if invested_capital > 0:
                roic = round(nopat / invested_capital, 4)
                findings["roic_latest"] = roic
                wacc = kwargs.get("wacc", 0.12)
                if roic < wacc:
                    flags.append(f"ROIC ({roic:.1%}) < WACC ({wacc:.1%}) — value destruction")
            else:
                data_gaps.append("Invested capital ≤ 0 — cannot compute ROIC")

        # ── Earnings Quality ──
        ocf = _fget(latest_cf, "Operating Cash Flow", "Cash from Operating", "CFO")
        depreciation = _fget(latest_pl, "Depreciation", "Depreciation and Amortisation", default=0)
        capex = _fget(latest_cf, "Capital Expenditure", "Purchase of Fixed Assets", "Capex")

        ebitda = (ebit or 0) + depreciation
        if ocf and ebitda and ebitda > 0:
            findings["ocf_ebitda_ratio"] = round(ocf / ebitda, 4)

        if ocf and capex and pat:
            fcf = ocf - abs(capex)
            if pat != 0:
                findings["fcf_pat_ratio"] = round(fcf / pat, 4)

        # ── Working Capital (CCC) ──
        inventory = _fget(latest_bs, "Inventories", "Inventory", default=0)
        receivables = _fget(latest_bs, "Trade Receivables", "Debtors", "Receivables", default=0)
        payables = _fget(latest_bs, "Trade Payables", "Sundry Creditors", "Creditors", default=0)
        cogs = _fget(latest_pl, "Cost of Materials", "Cost of Goods Sold", "COGS",
                      "Material Cost", "Raw Material Cost", default=0)

        if cogs > 0 and revenue and revenue > 0:
            dio = round((inventory / cogs) * 365, 1) if inventory else None
            dso = round((receivables / revenue) * 365, 1) if receivables is not None else None
            dpo = round((payables / cogs) * 365, 1) if payables else None
            ccc = None
            if dio is not None and dso is not None and dpo is not None:
                ccc = round(dio + dso - dpo, 1)
            findings["working_capital"] = {"dio": dio, "dso": dso, "dpo": dpo, "ccc_days": ccc}

        # ── Revenue CAGR ──
        if len(years) >= 4:
            rev_first = _fget(pl.get(years[0], {}), "Revenue", "Sales+", "Net Sales")
            rev_last = _fget(pl.get(years[-1], {}), "Revenue", "Sales+", "Net Sales")
            if rev_first and rev_last and rev_first > 0 and rev_last > 0:
                n = len(years) - 1
                cagr = ((rev_last / rev_first) ** (1 / n) - 1) * 100
                findings["revenue_cagr"] = {"pct": round(cagr, 2), "years": n}

        # ── Leverage ──
        interest = _fget(latest_pl, "Interest", "Finance Costs", "Interest Expense", default=0)
        if ebit and interest and interest > 0:
            ic = round(ebit / interest, 2)
            findings["interest_coverage"] = ic
            if ic < 3:
                flags.append(f"Interest coverage {ic}x — debt servicing risk")

        if ebitda and ebitda > 0:
            net_debt = total_debt - (cash or 0)
            findings["net_debt_ebitda"] = round(net_debt / ebitda, 2)

        # ── Reverse DCF ──
        market_cap = kwargs.get("market_cap")
        if market_cap and ocf and capex:
            fcf_base = ocf - abs(capex)
            if fcf_base > 0:
                wacc = kwargs.get("wacc", 0.12)
                tg = kwargs.get("terminal_growth", 0.05)
                implied_g = _reverse_dcf(market_cap, fcf_base, wacc, tg)
                if implied_g is not None:
                    findings["reverse_dcf_implied_growth"] = implied_g

        findings["flags"] = flags
        findings["data_gaps"] = data_gaps

        return self._build_trail(ticker, findings, data_gaps, flags, start)

    def _build_trail(self, ticker, findings, gaps, flags, start):
        elapsed = round(time.time() - start, 2)
        gap_count = len(gaps)
        confidence = max(0.5, 1.0 - (gap_count * 0.1))
        return AuditTrail(
            agent_name=self.agent_name,
            ticker=ticker,
            findings=findings,
            data_gaps=gaps,
            confidence=round(confidence, 2),
            execution_time_s=elapsed,
            steps=[{"action": "python_computation", "thought": "Pure deterministic calculation"}],
        )
