"""
novus_v3/core/agent_base_v3.py — v3 Agent Base Class

Wires together all 5 shifts:
  1. Tool use          → build_agent_tools()
  2. ReAct loop        → react_engine.react_loop()
  3. Self-verification → react_engine.run_verification()
  4. Dynamic prompts   → prompt_composer.compose_prompt()
  5. Audit trail       → AuditTrail dataclass

Subclasses implement:
  - agent_name, agent_role, output_example
  - build_agent_tools()    → register agent-specific tools
  - build_initial_context() → what to tell the model to start investigating
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from core.tools import ToolRegistry, build_shared_tools, Tool
from core.react_engine import react_loop, run_verification, ReActResult
from core.prompt_composer import compose_prompt
from core.llm_client import LLMClient, get_llm_client


@dataclass
class AuditTrail:
    """The v3 agent output — reasoning chain IS the product."""
    agent_name: str
    ticker: str
    sector: str = ""

    # Investigation record
    steps: list[dict] = field(default_factory=list)
    findings: Optional[dict] = None
    data_gaps: list[str] = field(default_factory=list)

    # Verification
    verification: Optional[dict] = None
    verified: bool = False

    # Metadata
    confidence: float = 0.0
    tools_called: int = 0
    llm_calls: int = 0
    execution_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "ticker": self.ticker,
            "sector": self.sector,
            "findings": self.findings,
            "data_gaps": self.data_gaps,
            "confidence": self.confidence,
            "verification": self.verification,
            "investigation_depth": {
                "tools_called": self.tools_called,
                "llm_calls": self.llm_calls,
                "steps": len(self.steps),
            },
            "execution_time_s": self.execution_time_s,
        }

    def to_analyst_note(self) -> str:
        """Render as a readable investigation report."""
        lines = [f"## {self.agent_name.replace('_',' ').title()}: {self.ticker}"]
        lines.append(f"**Confidence:** {self.confidence:.0%} | "
                      f"**Tools used:** {self.tools_called} | "
                      f"**Time:** {self.execution_time_s}s")
        lines.append("")

        if self.steps:
            lines.append("### Investigation path")
            for s in self.steps:
                action = s.get("action", "Reasoning")
                thought = s.get("thought", "")
                if not thought:
                    # Provide tool input parameters as context if thought trace is empty
                    inps = s.get("input", {})
                    thought = str(inps)[:150]
                lines.append(f"- **{action}**: {thought}")
            lines.append("")

        if self.findings:
            lines.append("### Findings")
            from utils.formatters import format_dict_as_markdown
            lines.extend(format_dict_as_markdown(self.findings, indent=0))
            lines.append("")

        if self.data_gaps:
            lines.append("### Data gaps")
            for g in self.data_gaps:
                lines.append(f"- {g}")
            lines.append("")

        if self.verification:
            rel = self.verification.get("overall_reliability", "N/A")
            lines.append(f"### Verification: {rel}")
            for err in self.verification.get("critical_errors", []):
                lines.append(f"- ⚠️ {err}")

        return "\n".join(lines)


class AgentV3(ABC):
    """
    v3 agent base.
    
    Comparison to your current AgentBase:
    
    Current AgentBase.execute():
        system_prompt = self.build_system_prompt(ticker)       # hardcoded
        raw_response = call_deepseek(system_prompt, context)   # 1 call
        parsed = self.output_model.model_validate_json(resp)   # parse
        citations = self._validate_citations(parsed, context)  # substring check
        return AgentFinding(...)                                # done
    
    AgentV3.execute():
        prompt = compose_prompt(sector, signals, ...)          # dynamic
        tools = shared_tools + agent_tools                     # tool registry
        result = react_loop(prompt, context, tools)            # multi-turn
        verified = run_verification(result, tools)             # critic pass
        return AuditTrail(...)                                 # full trail
    """

    MAX_ITERATIONS = 12
    VERIFY = True

    @property
    @abstractmethod
    def agent_name(self) -> str: ...

    @property
    @abstractmethod
    def agent_role(self) -> str:
        """One-paragraph description of this agent's expertise."""
        ...

    @property
    @abstractmethod
    def output_example(self) -> str:
        """Concrete JSON example of the expected output (NOT a schema dump)."""
        ...

    def build_agent_tools(self, doc: str, tables: dict, ticker: str = "") -> list[Tool]:
        """
        Override to add agent-specific tools beyond the shared set.
        Return a list of Tool objects. They'll be merged into the shared registry.
        """
        return []

    def build_initial_context(
        self, ticker: str, sector: str, signals: dict, doc_chars: int,
    ) -> str:
        """
        Override to customise what the model sees on its first turn.
        Default: brief overview + extraction signals.
        """
        ctx = (
            f"Analyze {ticker} ({sector} sector). "
            f"The document contains {doc_chars:,} characters. "
            f"Use your tools to investigate — do not try to read everything at once."
        )
        # Append extraction signals
        signal_messages = {
            "has_rpt_disclosures":     "⚠️ Significant RPT disclosures detected.",
            "has_contingent_liabilities": "⚠️ Contingent liabilities found.",
            "auditor_changed":         "⚠️ Auditor change detected.",
            "promoter_shares_pledged": "⚠️ Promoter shares pledged.",
            "high_other_income":       "⚠️ Other income appears elevated.",
        }
        for key, msg in signal_messages.items():
            if signals.get(key):
                ctx += f"\n{msg}"
        return ctx

    def execute(
        self,
        ticker: str,
        document_text: str,
        financial_tables: dict,
        sector: str,
        extraction_signals: dict,
        llm: LLMClient = None,
        dynamic_mandate: str = "",
    ) -> AuditTrail:
        """Full v3 execution: compose → investigate → verify → audit trail."""
        start = time.time()

        # ── 1. Compose prompt ──
        output_instruction = (
            "When you have completed your investigation, output your findings "
            "as a JSON object matching this example structure:\n\n"
            f"```json\n{self.output_example}\n```\n\n"
            "Only output the JSON when you are confident. Until then, keep investigating."
        )
        system_prompt = compose_prompt(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            agent_output_instruction=output_instruction,
            sector=sector,
            extraction_signals=extraction_signals,
            ticker=ticker,
        )
        # Inject the dynamic mandate from the Lead Analyst if provided
        if dynamic_mandate:
            system_prompt += f"\n\n## DYNAMIC MANDATE (from Lead Analyst)\n{dynamic_mandate}"

        # ── 2. Build tools ──
        tools = build_shared_tools(document_text, financial_tables, ticker=ticker)
        for extra_tool in self.build_agent_tools(document_text, financial_tables, ticker=ticker):
            tools.register(extra_tool)

        # ── 3. Initial context ──
        initial = self.build_initial_context(
            ticker, sector, extraction_signals, len(document_text),
        )

        # ── 4. ReAct loop ──
        react_result: ReActResult = react_loop(
            system_prompt=system_prompt,
            initial_context=initial,
            tools=tools,
            max_iterations=self.MAX_ITERATIONS,
            llm=llm,
        )

        # ── 5. Verification ──
        verification = None
        if self.VERIFY and react_result.final_output:
            verification = run_verification(
                findings=react_result.final_output,
                tools=tools,
                llm=llm,
            )

        # ── 6. Compute confidence ──
        confidence = self._compute_confidence(react_result, verification)

        # ── 7. Assemble audit trail ──
        elapsed = round(time.time() - start, 2)
        trail = AuditTrail(
            agent_name=self.agent_name,
            ticker=ticker,
            sector=sector,
            steps=[
                {
                    "action": s.action or "reasoning",
                    "thought": (s.thought or "")[:200],
                    "observation": (s.observation or "")[:200],
                }
                for s in react_result.reasoning_chain
            ],
            findings=react_result.final_output,
            data_gaps=(react_result.final_output or {}).get("data_gaps", []),
            verification=verification,
            verified=verification is not None,
            confidence=confidence,
            tools_called=react_result.tools_called,
            llm_calls=react_result.total_llm_calls,
            execution_time_s=elapsed,
        )
        return trail

    def _compute_confidence(self, react: ReActResult, verif: Optional[dict]) -> float:
        if react.final_output is None:
            return 0.1
        score = 0.4                                       # base: produced output
        score += min(react.tools_called * 0.07, 0.25)     # investigation depth
        if react.unique_tools_used >= 3:
            score += 0.1                                  # breadth bonus
        if verif:
            rel = verif.get("overall_reliability", 0.5)
            score += 0.25 * rel                           # verification score
            errors = verif.get("critical_errors", [])
            score -= len(errors) * 0.1                    # penalise errors
        return round(max(0.1, min(1.0, score)), 2)
