from core.agent_base_v3 import AgentV3
from core.tools import Tool
from .agent_utils import _cross_ref, _safe_handler

class ForensicInvestigatorV3(AgentV3):
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

    def build_agent_tools(self, doc: str, tables: dict, ticker: str = "") -> list[Tool]:
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
                handler=_safe_handler(lambda item_a, item_b, table: _cross_ref(tables, item_a, item_b, table)),
            ),
        ]
