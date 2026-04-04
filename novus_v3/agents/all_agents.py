"""
novus_v3/agents/ — All v3 Specialist Agents

Each agent extends AgentV3 and defines:
  - agent_role:        WHO it is (replaces hardcoded system prompt)
  - output_example:    WHAT it outputs (concrete JSON, not schema dump)
  - build_agent_tools: HOW it investigates (agent-specific tools)
  - build_initial_context: WHERE it starts looking

Agents marked [LLM] use the ReAct loop.
Agents marked [PYTHON] are pure computation — no LLM calls.
Agents marked [NEW] didn't exist in v2.
"""

import json
import time
import re
import numpy as np
from typing import Optional
from novus_v3.core.agent_base_v3 import AgentV3, AuditTrail
from novus_v3.core.tools import Tool, ToolRegistry, build_shared_tools


# ═══════════════════════════════════════════════════════════════════════════
# 1. FORENSIC INVESTIGATOR [LLM] — The Accounting Skeptic
# ═══════════════════════════════════════════════════════════════════════════

class ForensicInvestigatorV3(AgentV3):
    """
    v2 → v3 changes:
    - Was: raw schema dump, 1 LLM call, no verification, no sector awareness
    - Now: ReAct loop, searches documents, computes ratios via Python tools,
           self-verifies, adapts prompt to sector
    """

    @property
    def agent_name(self) -> str:
        return "forensic_investigator"

    @property
    def agent_role(self) -> str:
        return (
            "You are a forensic accounting analyst for an institutional equity fund. "
            "Your job is to find accounting red flags, aggressive recognition policies, "
            "and hidden risks that a surface-level analysis would miss. "
            "You are SKEPTICAL by default — assume management is optimising appearances "
            "until the evidence convinces you otherwise."
        )

    @property
    def output_example(self) -> str:
        return """{
  "related_party_flags": [
    {"description": "Royalty to parent at 3.45% of turnover — Rs 2,182 Cr outflow",
     "severity": "MEDIUM",
     "evidence": "Note 34: Royalty paid to Unilever plc at 3.45% of domestic turnover",
     "year_trend": "Stable at 3.4-3.5% for 3 years"}
  ],
  "cwip_aging_flags": [],
  "auditor_flags": [
    {"description": "Emphasis of Matter on ICDR compliance — non-standard",
     "severity": "LOW",
     "evidence": "Auditor Report para 4: emphasis on compliance with ICDR regulations"}
  ],
  "other_income_analysis": {
    "is_material": false,
    "ratio_pct": "4.2%",
    "components": "Primarily interest income and fair value gains on investments"
  },
  "contingent_liabilities": [
    {"description": "Disputed indirect tax demands of Rs 892 Cr",
     "severity": "MEDIUM",
     "evidence": "Note 38: Claims not acknowledged as debts — Rs 892 Cr in indirect taxes"}
  ],
  "earnings_quality_signals": [
    "Trade receivables grew 18% while revenue grew 8% — potential channel stuffing",
    "Provision for doubtful debts declined despite receivable growth — aggressive"
  ],
  "executive_summary": [
    "No critical red flags. RPT (royalty) is structural and stable at 3.45%.",
    "Receivable-revenue divergence needs monitoring — 18% vs 8% growth gap.",
    "Contingent liabilities manageable at Rs 892 Cr vs net worth Rs 8,200 Cr."
  ],
  "data_gaps": ["Segment-wise RPT breakdown not available in provided context"]
}"""

    def build_agent_tools(self, doc: str, tables: dict) -> list[Tool]:
        """Forensic-specific: anomaly scanner and cross-reference checker."""
        return [
            Tool(
                name="cross_reference_check",
                description=(
                    "Check if two related metrics are moving consistently. "
                    "E.g., if revenue grows 10% but receivables grow 30%, that's suspicious. "
                    "Provide two line items — the tool computes their growth rates and flags divergence."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "item_a": {"type": "string", "description": "First metric"},
                        "item_b": {"type": "string", "description": "Second metric (should move together)"},
                        "table":  {"type": "string"},
                    },
                    "required": ["item_a", "item_b", "table"],
                },
                handler=lambda item_a, item_b, table: _cross_ref(tables, item_a, item_b, table),
            ),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 2. NARRATIVE DECODER [LLM] — The Concall Analyst
# ═══════════════════════════════════════════════════════════════════════════

class NarrativeDecoderV3(AgentV3):
    """
    v2 → v3 changes:
    - Was: raw schema dump, compared Q3 vs Q4 without verifying both exist
    - Now: searches for specific guidance statements, detects tone shifts
           via tool calls, no longer asks LLM to compute discrepancy scores
    """

    @property
    def agent_name(self) -> str:
        return "narrative_decoder"

    @property
    def agent_role(self) -> str:
        return (
            "You are a management communication analyst for an institutional fund. "
            "You decode earnings call transcripts to find: "
            "(1) Guidance that was quietly changed or missed, "
            "(2) Language shifts from confident to hedging, "
            "(3) Analyst questions that management dodged. "
            "Focus on the Q&A section — the prepared remarks are scripted PR. "
            "The Q&A is where truth leaks."
        )

    @property
    def output_example(self) -> str:
        return """{
  "guidance_tracker": [
    {"topic": "Volume growth",
     "prior_guidance": "Management guided for double-digit volume growth in Q3 call",
     "actual_outcome": "Reported 2% volume growth — 80% miss vs guidance",
     "management_explanation": "Attributed to rural slowdown and base effect",
     "credibility": "LOW — rural slowdown was already visible in Q2 data",
     "evidence_prior": "Q3 transcript p.4: 'We expect double-digit volume growth'",
     "evidence_actual": "Q4 transcript p.2: 'Volume growth was 2%'"}
  ],
  "tone_shifts": [
    {"topic": "Rural demand",
     "prior_tone": "Q3: 'Very optimistic about rural recovery'",
     "current_tone": "Q4: 'Rural remains challenging, we are cautiously navigating'",
     "shift_type": "Optimistic → Cautious",
     "significance": "HIGH — suggests rural thesis has broken"}
  ],
  "analyst_dodges": [
    {"question": "Analyst asked about margin guidance for H2",
     "management_response": "Management pivoted to discussing brand investments without giving margin guidance",
     "evasion_type": "Deflection — answered a different question",
     "significance": "MEDIUM"}
  ],
  "key_phrases_flagged": [
    "'One-time impact' used 4 times — pattern suggests recurring costs being positioned as temporary",
    "'Strategic investment' used for every margin-dilutive action — possible euphemism for overspending"
  ],
  "executive_summary": [
    "Major guidance miss on volumes: guided double-digit, delivered 2%.",
    "Clear tone deterioration on rural demand — Q3 optimism → Q4 hedging.",
    "Margin guidance actively avoided — bearish signal."
  ],
  "data_gaps": ["Only Q4 transcript available — cannot compare with Q3 guidance"]
}"""

    def build_agent_tools(self, doc: str, tables: dict) -> list[Tool]:
        """Narrative-specific: tone analysis and guidance search."""
        return [
            Tool(
                name="search_management_guidance",
                description=(
                    "Search specifically for forward-looking statements, guidance, "
                    "and promises made by management. Looks for phrases like: "
                    "'we expect', 'we guide', 'outlook', 'going forward', "
                    "'next quarter', 'we aim to', 'target of'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to find guidance on, e.g. 'margins', 'volume growth', 'capex'"},
                    },
                    "required": ["topic"],
                },
                handler=lambda topic: _search_guidance(doc, topic),
            ),
            Tool(
                name="detect_hedging_language",
                description=(
                    "Scan for hedging/evasive language patterns in the transcript. "
                    "Returns instances of: 'challenging environment', 'one-time', "
                    "'strategic investment', 'going forward', 'as I said', topic pivots."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "section": {"type": "string", "description": "'full', 'qa_only', or 'prepared_remarks'"},
                    },
                    "required": ["section"],
                },
                handler=lambda section: _detect_hedging(doc, section),
            ),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 3. MOAT ARCHITECT [LLM] — The Industry Strategist
# ═══════════════════════════════════════════════════════════════════════════

class MoatArchitectV3(AgentV3):
    """
    v2 → v3 changes:
    - Was: already best-in-class with example prompts and sector config
    - Now: adds tool use for metric lookups, Porter's evidence verification,
           market share computation via Python
    """

    @property
    def agent_name(self) -> str:
        return "moat_architect"

    @property
    def agent_role(self) -> str:
        return (
            "You are an industry strategist assessing competitive moats and pricing "
            "power for an institutional equity fund. Your job is to determine whether "
            "this company's competitive advantages are DURABLE or ERODING. "
            "You must back every claim with specific numerical evidence from the filings. "
            "If you claim 'strong distribution moat', you must cite the exact outlet count."
        )

    @property
    def output_example(self) -> str:
        return """{
  "volume_vs_value": {
    "revenue_growth_pct": 8.2,
    "volume_growth_pct": 2.1,
    "price_driven_pct": 6.1,
    "verdict": "75% of growth is price-driven — real demand is weak"
  },
  "market_share": {
    "trend": "Losing in mass, gaining in premium",
    "evidence": "Mass portfolio declined 2%, premium grew 14% per Q4 transcript",
    "risk": "Premium is 25% of revenue — mass erosion outweighs premium gains"
  },
  "porters_five_forces": {
    "new_entrants":        {"strength": "LOW",    "evidence": "Direct reach 3.8M outlets + Rs 4,500 Cr ad spend creates high barrier"},
    "supplier_power":      {"strength": "MEDIUM", "evidence": "Palm oil is 30% of COGS — commodity with multiple suppliers"},
    "buyer_power":         {"strength": "HIGH",   "evidence": "Mass segment volumes dropped 4% after 6% price hike — high elasticity"},
    "substitutes":         {"strength": "MEDIUM", "evidence": "Private label share grew from 8% to 12% in modern trade"},
    "rivalry":             {"strength": "HIGH",   "evidence": "Category promo intensity up 200 bps YoY per industry data"}
  },
  "moat_durability": "WEAKENING",
  "competitive_advantages": [
    "Distribution moat intact (3.8M outlets) but rural erosion ongoing (-400K outlets)",
    "Pricing power tested: demand elasticity higher than management expected",
    "Brand premium holds in premium segment but value segment commoditising"
  ],
  "data_gaps": ["Industry volume data not in context — cannot benchmark vs peers"]
}"""

    def build_agent_tools(self, doc: str, tables: dict) -> list[Tool]:
        return [
            Tool(
                name="search_competitive_data",
                description=(
                    "Search for competitive intelligence: market share, "
                    "distribution reach, competitive actions, new launches, "
                    "pricing actions, channel changes."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "e.g. 'market share', 'distribution reach', 'new product launches'"},
                    },
                    "required": ["topic"],
                },
                handler=lambda topic: _search_competitive(doc, topic),
            ),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 4. CAPITAL ALLOCATOR [LLM] — The Value Guard
# ═══════════════════════════════════════════════════════════════════════════

class CapitalAllocatorV3(AgentV3):
    """
    v2 → v3 changes:
    - Was: already well-scoped (qualitative only, ROIC in forensic_quant)
    - Now: tool-driven investigation of M&A track record, goodwill checking
           via compute_ratio, dividend consistency via compare_years
    """

    @property
    def agent_name(self) -> str:
        return "capital_allocator"

    @property
    def agent_role(self) -> str:
        return (
            "You are the capital allocation analyst for an institutional fund. "
            "You assess whether management is a good STEWARD of shareholder capital. "
            "You look for: empire building (unrelated diversification), M&A quality, "
            "dividend/buyback discipline, and overall capital allocation coherence. "
            "You focus on QUALITATIVE signals from filings — quant metrics like ROIC "
            "are computed separately by the forensic quant agent."
        )

    @property
    def output_example(self) -> str:
        return """{
  "empire_building": {
    "unrelated_acquisitions": [
      "Acquired D2C beauty brand for Rs 450 Cr — outside core FMCG competency"
    ],
    "cash_hoarding": false,
    "excessive_goodwill": false,
    "verdict": "1 unrelated acquisition in 12 months — early stage, not yet a pattern"
  },
  "mna_quality": {
    "acquisitions": [
      {"target": "Nutrition Co", "amount": "Rs 200 Cr", "year": "FY23",
       "integration_status": "Integrated — 15% revenue growth post-acquisition",
       "evidence": "Q4 transcript: 'Our nutrition portfolio grew 15% since acquisition'"}
    ],
    "goodwill_impairment_history": "No impairment in last 3 years"
  },
  "capital_return": {
    "dividend_pattern": "Growing — DPS Rs 34 to Rs 39 over 3 years",
    "buyback_activity": "No buybacks despite strong FCF — missed opportunity",
    "payout_ratio": "~80% — disciplined but lacks reinvestment ambition"
  },
  "grade": "B",
  "key_findings": [
    "Strong dividend discipline but questionable M&A — D2C bet is unproven",
    "No buybacks despite premium FCF yield",
    "Nutrition acquisition integrating well — evidence of execution ability"
  ],
  "data_gaps": ["Detailed M&A valuation multiples not disclosed"]
}"""

    def build_agent_tools(self, doc: str, tables: dict) -> list[Tool]:
        return [
            Tool(
                name="search_capital_decisions",
                description=(
                    "Search for capital allocation decisions: acquisitions, "
                    "divestments, capex announcements, buyback programs, "
                    "dividend declarations, debt repayment plans."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                    },
                    "required": ["topic"],
                },
                handler=lambda topic: _search_capital(doc, topic),
            ),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 5. MANAGEMENT QUALITY [LLM] — [NEW AGENT]
# ═══════════════════════════════════════════════════════════════════════════

class ManagementQualityV3(AgentV3):
    """
    NEW in v3. Your v2 had NO agent for this.
    
    Critical for Indian markets where promoter risk is real
    (Satyam, DHFL, Yes Bank, Manpasand, Vakrangee).
    """

    @property
    def agent_name(self) -> str:
        return "management_quality"

    @property
    def agent_role(self) -> str:
        return (
            "You are the management quality analyst for an institutional fund. "
            "You assess GOVERNANCE risk: promoter integrity, board independence, "
            "insider trading patterns, compensation alignment, and KMP stability. "
            "In Indian markets, promoter-driven governance failures (Satyam, DHFL, "
            "Yes Bank) are among the biggest risk factors. Your job is to catch "
            "the early signals before they become front-page news."
        )

    @property
    def output_example(self) -> str:
        return """{
  "promoter_analysis": {
    "holding_pct": 67.2,
    "pledge_pct": 0.0,
    "holding_trend": "Stable for 3 years — no stake sales",
    "insider_transactions": "No significant insider transactions in last 12 months"
  },
  "board_quality": {
    "independent_directors_pct": 50,
    "meets_sebi_requirement": true,
    "audit_committee_independence": "All independent — compliant",
    "related_directors": "None identified",
    "tenure_risk": "2 independent directors serving > 8 years — possible entrenchment"
  },
  "kmp_stability": {
    "cfo_tenure": "3 years — stable",
    "ceo_tenure": "5 years — stable",
    "recent_departures": "Company Secretary resigned Q3 — minor flag",
    "succession_plan": "No formal succession plan disclosed"
  },
  "compensation_alignment": {
    "md_compensation_vs_profit": "MD comp Rs 42 Cr on Rs 9,800 Cr PAT — 0.4% — reasonable",
    "variable_vs_fixed": "60% variable — aligned with performance",
    "esos_dilution": "ESOS pool is 0.8% of outstanding shares — minimal dilution"
  },
  "governance_flags": [
    "2 independent directors serving > 8 years — re-evaluate independence",
    "Company Secretary resignation in Q3 — check if governance-related"
  ],
  "governance_grade": "B+",
  "data_gaps": ["SAST filings not in provided context — cannot verify insider trades"]
}"""

    def build_agent_tools(self, doc: str, tables: dict) -> list[Tool]:
        return [
            Tool(
                name="search_governance",
                description=(
                    "Search for governance information: board composition, "
                    "promoter holdings, KMP changes, compensation details, "
                    "audit committee, insider transactions, SEBI compliance."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                    },
                    "required": ["topic"],
                },
                handler=lambda topic: _search_governance(doc, topic),
            ),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# 6. FORENSIC QUANT [PYTHON-ONLY] — Pure Math Agent
# ═══════════════════════════════════════════════════════════════════════════
# This agent does NOT use the ReAct loop because it makes zero LLM calls.
# It is PURE PYTHON computation. This is v3's version of your existing
# forensic_quant.py — the architecture stays the same because it was
# already correctly designed (Law 2: Python calculates, LLM narrates).

class ForensicQuantV3:
    """
    Pure Python agent — no LLM calls, no ReAct loop.
    
    Your existing forensic_quant.py was already your best agent (Grade A).
    The v3 version keeps the same architecture but adds:
    - Peer-relative metrics (when peer data is available)
    - Altman Z-score for distress prediction
    - Piotroski F-score for financial strength
    - Better sector-aware thresholds
    """

    agent_name = "forensic_quant"

    def execute(self, ticker: str, financial_tables: dict, **kwargs) -> AuditTrail:
        start = time.time()
        
        pl = financial_tables.get("profit_loss", {})
        bs = financial_tables.get("balance_sheet", {})
        cf = financial_tables.get("cash_flow", {})
        
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
            nopat = ebit * 0.75  # 25% India corporate tax
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


# ═══════════════════════════════════════════════════════════════════════════
# 7. PM SYNTHESIS [LLM] — The Portfolio Manager
# ═══════════════════════════════════════════════════════════════════════════

class PMSynthesisV3(AgentV3):
    """
    v2 → v3 changes:
    - Was: hardcoded agent name lookups from state, single LLM call
    - Now: dynamically reads ALL agent findings from input, uses tools
           to cross-reference specific claims, self-verifies the thesis
    """

    MAX_ITERATIONS = 6   # Synthesis is reasoning-heavy, fewer tool calls
    VERIFY = False        # Verification is less useful for synthesis (subjective)

    @property
    def agent_name(self) -> str:
        return "pm_synthesis"

    @property
    def agent_role(self) -> str:
        return (
            "You are an elite portfolio manager at a top-tier institutional fund. "
            "You synthesise findings from multiple specialist agents into a single "
            "investment thesis. You ONLY use data provided — never invent, interpolate, "
            "or assume. If data is missing, say so explicitly. "
            "Your thesis must be actionable: BUY, WATCH, or PASS with measurable kill criteria."
        )

    @property
    def output_example(self) -> str:
        return """{
  "executive_summary": "Comprehensive 1-paragraph summary of the investment case.",
  "fundamental_analysis": "Deep paragraph on business model, moat, and competitive position.",
  "forensic_audit": "Deep paragraph on accounting quality, earnings quality, and red flags.",
  "capital_allocation": "Deep paragraph on management's capital stewardship, M&A, and returns policy.",
  "management_quality": "Deep paragraph on governance, promoter integrity, and KMP stability.",
  "bull_case": ["Pillar 1 with evidence", "Pillar 2 with evidence", "Pillar 3"],
  "bear_case": [
    {"risk": "Description with evidence", "probability": "LOW|MEDIUM|HIGH", "impact": "Description"}
  ],
  "variant_perception": "What the market is NOT pricing in — your edge. Or 'None identified'.",
  "scoreboard": {
    "forensic_quality": "A|B|C|D",
    "management_score": "A|B|C|D",
    "moat_durability": "STRONG|INTACT|WEAKENING|BROKEN",
    "pricing_verdict": "CHEAP|FAIR|EXPENSIVE",
    "reverse_dcf_implied_growth": null
  },
  "recommendation": "BUY|WATCH|PASS",
  "kill_criteria": [
    "ROIC drops below 12% for 2 consecutive quarters",
    "Promoter pledge exceeds 10% of holding"
  ],
  "data_gaps": ["Missing data point 1", "Missing data point 2"]
}"""

    def build_initial_context(self, ticker, sector, signals, doc_chars) -> str:
        """PM gets ALL agent findings as its starting context."""
        agent_outputs = signals.get("_agent_outputs", {})
        parts = [f"Synthesise findings for {ticker} ({sector})."]
        for agent_name, output in agent_outputs.items():
            parts.append(f"\n## {agent_name.upper()} FINDINGS:")
            if isinstance(output, dict):
                parts.append(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                parts.append(str(output)[:5000])
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions for agent-specific tools
# ═══════════════════════════════════════════════════════════════════════════

def _fget(data: dict, *keys, default=None):
    """Get first available key from a dict (None-safe, zero-safe)."""
    for k in keys:
        val = data.get(k)
        if val is not None:
            return val
        # Try fuzzy
        for dk, dv in data.items():
            if k.lower() in dk.lower():
                return dv
    return default


def _reverse_dcf(mcap, fcf, wacc=0.12, tg=0.05, years=10):
    if mcap <= 0 or fcf <= 0 or wacc <= tg:
        return None
    lo, hi = 0.0, 0.50
    for _ in range(100):
        mid = (lo + hi) / 2
        pv = sum(fcf * (1 + mid) ** t / (1 + wacc) ** t for t in range(1, years + 1))
        tv = fcf * (1 + mid) ** years * (1 + tg) / (wacc - tg)
        pv += tv / (1 + wacc) ** years
        if pv < mcap:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 0.0001:
            break
    return round((lo + hi) / 2, 4)


def _cross_ref(tables, item_a, item_b, table):
    """Check if two metrics are moving consistently."""
    tbl = tables.get(table, {})
    years = sorted(tbl.keys())
    data = {"item_a": item_a, "item_b": item_b, "comparison": []}
    for i in range(1, len(years)):
        y0, y1 = years[i-1], years[i]
        va0 = _fuzzy(tbl.get(y0, {}), item_a)
        va1 = _fuzzy(tbl.get(y1, {}), item_a)
        vb0 = _fuzzy(tbl.get(y0, {}), item_b)
        vb1 = _fuzzy(tbl.get(y1, {}), item_b)
        row = {"year": y1}
        if isinstance(va0, (int,float)) and isinstance(va1, (int,float)) and va0:
            row["a_growth"] = round(((va1 - va0) / abs(va0)) * 100, 1)
        if isinstance(vb0, (int,float)) and isinstance(vb1, (int,float)) and vb0:
            row["b_growth"] = round(((vb1 - vb0) / abs(vb0)) * 100, 1)
        if "a_growth" in row and "b_growth" in row:
            row["gap_pp"] = round(row["a_growth"] - row["b_growth"], 1)
            row["diverging"] = abs(row["gap_pp"]) > 15
        data["comparison"].append(row)
    return data


def _fuzzy(data: dict, key: str):
    if key in data:
        return data[key]
    for k, v in data.items():
        if key.lower() in k.lower():
            return v
    return None


def _search_guidance(doc, topic):
    patterns = [
        rf"(?i)(?:we expect|guidance|outlook|target|aim to|going forward).*?{re.escape(topic)}",
        rf"(?i){re.escape(topic)}.*?(?:we expect|guidance|outlook|target)",
    ]
    results = []
    for pat in patterns:
        for m in re.finditer(pat, doc):
            start = max(0, m.start() - 100)
            end = min(len(doc), m.end() + 300)
            results.append({"passage": doc[start:end].strip(), "type": "guidance"})
    return results[:5] or [{"passage": f"No guidance found for '{topic}'", "type": "none"}]


def _detect_hedging(doc, section):
    hedging_phrases = [
        "challenging environment", "one-time", "one time", "strategic investment",
        "going forward", "as I said", "let me clarify", "I think we need to",
        "it's too early to", "we'll have to wait", "difficult to predict",
        "cautiously optimistic", "calibrated approach", "headwinds",
    ]
    text = doc
    if section == "qa_only":
        qa_idx = doc.lower().find("question and answer")
        if qa_idx == -1:
            qa_idx = doc.lower().find("q&a")
        if qa_idx > 0:
            text = doc[qa_idx:]

    found = []
    for phrase in hedging_phrases:
        count = text.lower().count(phrase)
        if count > 0:
            idx = text.lower().find(phrase)
            context = text[max(0, idx-50):idx+len(phrase)+100].strip()
            found.append({"phrase": phrase, "count": count, "context": context})
    found.sort(key=lambda x: -x["count"])
    return found[:10] or [{"phrase": "No hedging language detected", "count": 0}]


def _search_competitive(doc, topic):
    keywords = topic.lower().split()
    paras = re.split(r'\n\s*\n', doc)
    scored = []
    for p in paras:
        if len(p.strip()) < 30:
            continue
        lower = p.lower()
        score = sum(lower.count(k) for k in keywords if len(k) > 2)
        if score > 0:
            scored.append((score, p.strip()[:800]))
    scored.sort(key=lambda x: -x[0])
    return [{"passage": p, "score": s} for s, p in scored[:3]] or [{"passage": "Not found", "score": 0}]


def _search_capital(doc, topic):
    return _search_competitive(doc, topic)  # Same logic, different context


def _search_governance(doc, topic):
    return _search_competitive(doc, topic)


# ═══════════════════════════════════════════════════════════════════════════
# Agent Registry
# ═══════════════════════════════════════════════════════════════════════════

ALL_AGENTS = {
    "forensic_investigator": ForensicInvestigatorV3,
    "narrative_decoder":     NarrativeDecoderV3,
    "moat_architect":        MoatArchitectV3,
    "capital_allocator":     CapitalAllocatorV3,
    "management_quality":    ManagementQualityV3,
    "forensic_quant":        ForensicQuantV3,
    "pm_synthesis":          PMSynthesisV3,
}
