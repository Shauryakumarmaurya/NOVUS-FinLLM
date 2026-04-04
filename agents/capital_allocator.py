from core.agent_base_v3 import AgentV3
from core.tools import Tool
from .agent_utils import _search_capital, _safe_handler

class CapitalAllocatorV3(AgentV3):
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

    def build_agent_tools(self, doc: str, tables: dict, ticker: str = "") -> list[Tool]:
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
                handler=_safe_handler(lambda topic: _search_capital(doc, topic, ticker)),
            ),
        ]
