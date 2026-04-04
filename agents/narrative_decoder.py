from core.agent_base_v3 import AgentV3
from core.tools import Tool
from .agent_utils import _search_guidance, _detect_hedging, _safe_handler

class NarrativeDecoderV3(AgentV3):
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

    def build_agent_tools(self, doc: str, tables: dict, ticker: str = "") -> list[Tool]:
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
                handler=_safe_handler(lambda topic: _search_guidance(doc, topic, ticker)),
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
                handler=_safe_handler(lambda section: _detect_hedging(doc, section)),
            ),
        ]
