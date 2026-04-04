from core.agent_base_v3 import AgentV3
from core.tools import Tool
from .agent_utils import _search_competitive, _safe_handler

class MoatArchitectV3(AgentV3):
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

    def build_agent_tools(self, doc: str, tables: dict, ticker: str = "") -> list[Tool]:
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
                handler=_safe_handler(lambda topic: _search_competitive(doc, topic, ticker)),
            ),
        ]
