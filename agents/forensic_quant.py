import time
from core.agent_base_v3 import AuditTrail
from .agent_utils import _fget, _reverse_dcf
from rag_engine import query as rag_query

class ForensicQuantV3:
    agent_name = "forensic_quant"

    def execute(self, ticker: str, financial_tables: dict, **kwargs) -> AuditTrail:
        start = time.time()
        
        # Data is pre-normalized at the boundary (structured_data_fetcher.py)
        # Internal keys: profit_loss, balance_sheet, cash_flow
        pl = financial_tables.get("profit_loss", {})
        bs = financial_tables.get("balance_sheet", {})
        cf = financial_tables.get("cash_flow", {})
        qr = financial_tables.get("quarterly_results", {})
        
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

        # ── Profitability (DuPont) ──
        try:
            revenue = _fget(latest_pl, "Revenue", "Sales+", "Sales +", "Net Sales", "Revenue from Operations")
            ebit = _fget(latest_pl, "EBIT", "Operating Profit")
            pat = _fget(latest_pl, "Net Profit", "PAT", "Profit after tax")
            total_assets = _fget(latest_bs, "Total Assets")
            
            # Defensive Taxonomy: Screener splits Equity and Reserves
            equity_cap = _fget(latest_bs, "Equity Capital", default=0)
            reserves = _fget(latest_bs, "Reserves", default=0)
            if equity_cap > 0 or reserves > 0:
                total_equity = equity_cap + reserves
            else:
                total_equity = _fget(latest_bs, "Shareholders Funds", "Total Equity", "Equity", default=0)
                
            total_debt = _fget(latest_bs, "Borrowings", "Total Debt", "Long Term Borrowings", default=0)
            cash = _fget(latest_bs, "Cash Equivalents", "Cash and Bank", "Cash", default=0)

            if revenue and pat and total_assets and total_equity and total_equity > 0:
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
        except Exception as e:
            data_gaps.append(f"DuPont computation failed: {e}")

        # ── ROIC ──
        try:
            if ebit and total_equity is not None and total_debt is not None:
                nopat = ebit * 0.75
                invested_capital = total_equity + total_debt - (cash or 0)
                
                # Defensive Taxonomy for FMCG negative working capital
                if invested_capital > 0:
                    if invested_capital < (revenue * 0.05):
                        findings["roic_latest"] = "Unable to Verify (Potential massive goodwill or negative working capital skewing base)"
                    else:
                        roic = round(nopat / invested_capital, 4)
                        findings["roic_latest"] = roic
                        wacc = kwargs.get("wacc", 0.12)
                        if roic < wacc:
                            flags.append(f"ROIC ({roic:.1%}) < WACC ({wacc:.1%}) — value destruction")
                else:
                    findings["roic_latest"] = "Unable to Verify (Invested capital is negative/zero)"
        except Exception as e:
            data_gaps.append(f"ROIC computation failed: {e}")

        # ── Earnings Quality ──
        try:
            ocf = _fget(latest_cf, "Operating Cash Flow", "Cash from Operating", "CFO", "Cash from Operating Activity +", "Cash from Operating Activity")
            depreciation = _fget(latest_pl, "Depreciation", "Depreciation and Amortisation", default=0)
            capex = _fget(latest_cf, "Capital Expenditure", "Purchase of Fixed Assets", "Capex")

            ebitda = (ebit or 0) + (depreciation or 0)
            if ocf and ebitda and ebitda > 0:
                findings["ocf_ebitda_ratio"] = round(ocf / ebitda, 4)

            if ocf and capex and pat:
                fcf = ocf - abs(capex)
                if pat != 0:
                    findings["fcf_pat_ratio"] = round(fcf / pat, 4)
        except Exception as e:
            data_gaps.append(f"Earnings quality computation failed: {e}")

        # ── Working Capital (CCC) ──
        try:
            inventory = _fget(latest_bs, "Inventories", "Inventory", default=0)
            receivables = _fget(latest_bs, "Trade Receivables", "Debtors", "Receivables", default=0)
            payables = _fget(latest_bs, "Trade Payables", "Sundry Creditors", "Creditors", default=0)
            cogs = _fget(latest_pl, "Cost of Materials", "Cost of Goods Sold", "COGS",
                          "Material Cost", "Raw Material Cost", default=0)

            if cogs and cogs > 0 and revenue and revenue > 0:
                dio = round((inventory / cogs) * 365, 1) if inventory else None
                dso = round((receivables / revenue) * 365, 1) if receivables is not None else None
                dpo = round((payables / cogs) * 365, 1) if payables else None
                ccc = None
                if dio is not None and dso is not None and dpo is not None:
                    ccc = round(dio + dso - dpo, 1)
                findings["working_capital"] = {"dio": dio, "dso": dso, "dpo": dpo, "ccc_days": ccc}
        except Exception as e:
            data_gaps.append(f"Working capital computation failed: {e}")

        # ── Revenue CAGR ──
        try:
            if len(years) >= 4:
                rev_first = _fget(pl.get(years[0], {}), "Revenue", "Sales+", "Sales +", "Net Sales")
                rev_last = _fget(pl.get(years[-1], {}), "Revenue", "Sales+", "Sales +", "Net Sales")
                if rev_first and rev_last and rev_first > 0 and rev_last > 0:
                    n = len(years) - 1
                    cagr = ((rev_last / rev_first) ** (1 / n) - 1) * 100
                    findings["revenue_cagr"] = {"pct": round(cagr, 2), "years": n}
        except Exception as e:
            data_gaps.append(f"Revenue CAGR computation failed: {e}")

        # ── Leverage ──
        try:
            interest = _fget(latest_pl, "Interest", "Finance Costs", "Interest Expense", default=0)
            if ebit and interest and interest > 0:
                ic = round(ebit / interest, 2)
                findings["interest_coverage"] = ic
                if ic < 3:
                    flags.append(f"Interest coverage {ic}x — debt servicing risk")

            if ebitda and ebitda > 0:
                net_debt = (total_debt or 0) - (cash or 0)
                findings["net_debt_ebitda"] = round(net_debt / ebitda, 2)
        except Exception as e:
            data_gaps.append(f"Leverage computation failed: {e}")

        # ── Reverse DCF ──
        try:
            market_cap = kwargs.get("market_cap")
            if market_cap and ocf and capex:
                fcf_base = ocf - abs(capex)
                if fcf_base > 0:
                    wacc = kwargs.get("wacc", 0.12)
                    tg = kwargs.get("terminal_growth", 0.05)
                    implied_g = _reverse_dcf(market_cap, fcf_base, wacc, tg)
                    if implied_g is not None:
                        findings["reverse_dcf_implied_growth"] = implied_g
        except Exception as e:
            data_gaps.append(f"Reverse DCF computation failed: {e}")

        # ── Anomaly Bridge (RAG Integration) ──
        try:
            # 1. Check Quarterly Anomalies (Most critical for recent spikes like the 4048Cr Other Income)
            q_quarters = sorted(qr.keys())
            if len(q_quarters) >= 2:
                latest_q = q_quarters[-1]
                prev_q = q_quarters[-2]
                curr_pat_q = _fget(qr[latest_q], "Net Profit", "PAT", default=0)
                prev_pat_q = _fget(qr[prev_q], "Net Profit", "PAT", default=0)
                curr_other_q = _fget(qr[latest_q], "Other Income", default=0)
                
                if curr_pat_q and prev_pat_q and prev_pat_q > 0:
                    pat_growth_q = (curr_pat_q - prev_pat_q) / prev_pat_q
                    
                    # Also check if Other Income is massively distorting PBT
                    pbt_q = _fget(qr[latest_q], "Profit before tax", "PBT", default=curr_pat_q)
                    other_inc_ratio = (curr_other_q / pbt_q) if pbt_q > 0 else 0

                    if pat_growth_q > 0.30 or other_inc_ratio > 0.15:
                        res = rag_query(ticker, f"Why did net profit or other income jump heavily in the {latest_q} quarter? exceptional items", top_k=2)
                        ex = " | ".join(r['text'][:400] for r in res) if res else "No context found."
                        findings["anomaly_flag"] = f"Quarterly Spike: {latest_q} PAT changed {pat_growth_q:.1%} QoQ. RAG: {ex}"
            
            # 2. Check Annual Anomalies if Quarterly didn't trigger
            if "anomaly_flag" not in findings and len(years) >= 2:
                prev_pl = pl.get(years[-2], {})
                prev_pat = _fget(prev_pl, "Net Profit", "PAT", "Profit after tax", default=0)
                curr_pat = _fget(latest_pl, "Net Profit", "PAT", "Profit after tax", default=0)
                
                if curr_pat and prev_pat and prev_pat > 0:
                    pat_growth = (curr_pat - prev_pat) / prev_pat
                    if pat_growth > 0.20:
                        res = rag_query(ticker, f"Why did net profit or other income jump heavily in {latest}? exceptional items", top_k=2)
                        ex = " | ".join(r['text'][:400] for r in res) if res else "No context found."
                        findings["anomaly_flag"] = f"Annual Spike: {latest} PAT surged {pat_growth:.1%} YoY. RAG: {ex}"
        except Exception as e:
            flags.append(f"Bridge analysis failed: {e}")

        # Filter out valid zeros
        findings = {k: v for k, v in findings.items() if v is not None}
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
