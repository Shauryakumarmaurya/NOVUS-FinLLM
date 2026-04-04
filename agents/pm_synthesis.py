import json
from core.agent_base_v3 import AgentV3

class PMSynthesisV3(AgentV3):
    MAX_ITERATIONS = 6   
    VERIFY = False        

    @property
    def agent_name(self) -> str:
        return "pm_synthesis"

    @property
    def agent_role(self) -> str:
        return (
            "You are an elite portfolio manager at a top-tier institutional fund. "
            "You synthesise findings from multiple specialist agents into a single "
            "investment thesis. You ONLY use data provided — never invent, interpolate, "
            "or assume. "
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
  "open_questions_for_management": [
    "Could you provide a bridge for the 40% jump in other income?",
    "Why have CWIP projects extended past the 24-month delay range?"
  ]
}"""

    def build_initial_context(self, ticker, sector, signals, doc_chars) -> str:
        agent_outputs = signals.get("_agent_outputs", {})
        parts = [f"Synthesise findings for {ticker} ({sector})."]
        for agent_name, output in agent_outputs.items():
            parts.append(f"\n## {agent_name.upper()} FINDINGS:")
            if isinstance(output, dict):
                parts.append(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                parts.append(str(output)[:5000])
        return "\n".join(parts)
