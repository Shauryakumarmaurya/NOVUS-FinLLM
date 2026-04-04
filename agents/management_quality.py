from core.agent_base_v3 import AgentV3
from core.tools import Tool
from .agent_utils import _search_governance, _safe_handler

class ManagementQualityV3(AgentV3):
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

    def build_agent_tools(self, doc: str, tables: dict, ticker: str = "") -> list[Tool]:
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
                handler=_safe_handler(lambda topic: _search_governance(doc, topic, ticker)),
            ),
        ]
