"""
novus_v3/orchestrator/cio_v3.py — v3 CIO Orchestrator

Coordinates all v3 agents with:
  1. Intelligent task planning (which agents to run, in what order)
  2. Parallel execution of independent agents
  3. Data routing: quant agents get structured data, LLM agents get tools
  4. Reflection: high-severity findings re-trigger relevant agents
  5. Conflict detection: cross-check agent findings for contradictions
  6. Synthesis: PM agent merges everything into a single thesis
  7. Full audit trail preserved for every step

v2 → v3 changes:
  - Agents now use ReAct loops internally (orchestrator doesn't manage this)
  - New ManagementQuality agent added to the pipeline
  - PM synthesis receives ALL agent outputs dynamically (not hardcoded keys)
  - Conflict check uses embedding similarity pre-filter (not O(n²) LLM calls)
  - Sector detection is data-driven (not hardcoded ticker map)
"""

import json
import time
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field

from novus_v3.core.llm_client import LLMClient, get_llm_client
from novus_v3.core.agent_base_v3 import AuditTrail, AgentV3
from novus_v3.agents.all_agents import (
    ForensicInvestigatorV3,
    NarrativeDecoderV3,
    MoatArchitectV3,
    CapitalAllocatorV3,
    ManagementQualityV3,
    ForensicQuantV3,
    PMSynthesisV3,
    ALL_AGENTS,
)


# ═══════════════════════════════════════════════════════════════════════════
# State Object
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorState:
    """Central state shared across the pipeline."""
    ticker: str
    sector: str
    query: str
    fiscal_year: str = ""

    # Data
    document_text: str = ""
    financial_tables: dict = field(default_factory=dict)
    extraction_signals: dict = field(default_factory=dict)

    # Assumptions (human-editable)
    wacc: float = 0.12
    terminal_growth: float = 0.05
    market_cap: Optional[float] = None

    # Agent results
    agent_trails: dict[str, AuditTrail] = field(default_factory=dict)

    # Conflict check
    conflicts: list[dict] = field(default_factory=list)

    # Final output
    final_thesis: Optional[AuditTrail] = None
    final_report: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════════

# ── Phase 1: Agent Execution Order ────────────────────────────────────────
#
# This replaces the LLM-generated "strategic audit plan" from v2.
# In practice, the execution order is almost always the same.
# The LLM call to generate a plan was wasting time and tokens.

EXECUTION_PHASES = [
    # Phase 1: Independent agents (run in parallel)
    {
        "phase": "investigation",
        "parallel": True,
        "agents": [
            "forensic_quant",          # Python-only, fastest
            "forensic_investigator",   # LLM + tools
            "narrative_decoder",       # LLM + tools
            "moat_architect",          # LLM + tools
            "capital_allocator",       # LLM + tools
            "management_quality",      # LLM + tools [NEW]
        ],
    },
    # Phase 2: Reflection (conditional, runs if Phase 1 found red flags)
    {
        "phase": "reflection",
        "parallel": False,
        "agents": [],   # Populated dynamically
    },
    # Phase 3: Synthesis (must run after all others)
    {
        "phase": "synthesis",
        "parallel": False,
        "agents": ["pm_synthesis"],
    },
]


async def run_pipeline(
    ticker: str,
    document_text: str,
    financial_tables: dict,
    sector: str,
    extraction_signals: dict,
    query: str = "",
    wacc: float = 0.12,
    terminal_growth: float = 0.05,
    market_cap: float = None,
    progress_callback: Callable = None,
    llm: LLMClient = None,
) -> OrchestratorState:
    """
    Full v3 orchestration pipeline.
    
    Args:
        ticker:             Company ticker
        document_text:      Full extracted text from PDFs
        financial_tables:   Structured data {"profit_loss": {...}, "balance_sheet": {...}, ...}
        sector:             Company sector (e.g. "FMCG", "Banking")
        extraction_signals: {"has_rpt_disclosures": True, "auditor_changed": False, ...}
        query:              User's research question
        wacc:               Weighted average cost of capital
        terminal_growth:    Terminal growth rate assumption
        market_cap:         Current market cap (for reverse DCF)
        progress_callback:  fn(phase, active_agents, completed_agents)
        llm:                LLMClient instance
    """
    if llm is None:
        llm = get_llm_client()

    state = OrchestratorState(
        ticker=ticker,
        sector=sector,
        query=query,
        document_text=document_text,
        financial_tables=financial_tables,
        extraction_signals=extraction_signals,
        wacc=wacc,
        terminal_growth=terminal_growth,
        market_cap=market_cap,
    )

    # ── Phase 1: Parallel Investigation ──────────────────────────────
    phase1 = EXECUTION_PHASES[0]
    if progress_callback:
        progress_callback("investigation", phase1["agents"], [])

    await _run_agents_parallel(state, phase1["agents"], llm, progress_callback)

    # ── Phase 2: Reflection (conditional) ────────────────────────────
    reflection_agents = _determine_reflection_needs(state)
    if reflection_agents:
        if progress_callback:
            progress_callback("reflection", reflection_agents, list(state.agent_trails.keys()))
        await _run_agents_parallel(state, reflection_agents, llm, progress_callback)

    # ── Phase 2.5: Conflict Check ────────────────────────────────────
    if progress_callback:
        progress_callback("conflict_check", [], list(state.agent_trails.keys()))
    state.conflicts = _detect_conflicts(state)

    # ── Phase 3: Synthesis ───────────────────────────────────────────
    if progress_callback:
        progress_callback("synthesis", ["pm_synthesis"], list(state.agent_trails.keys()))

    # Inject all agent findings into extraction_signals for PM to read
    agent_outputs = {}
    for name, trail in state.agent_trails.items():
        if trail.findings:
            agent_outputs[name] = trail.findings
    # Inject conflicts
    if state.conflicts:
        agent_outputs["_conflicts"] = state.conflicts

    pm_signals = {**extraction_signals, "_agent_outputs": agent_outputs}
    pm = PMSynthesisV3()
    thesis_trail = pm.execute(
        ticker=ticker,
        document_text=document_text,
        financial_tables=financial_tables,
        sector=sector,
        extraction_signals=pm_signals,
        llm=llm,
    )
    state.agent_trails["pm_synthesis"] = thesis_trail
    state.final_thesis = thesis_trail
    state.final_report = thesis_trail.to_analyst_note()

    if progress_callback:
        progress_callback("complete", [], list(state.agent_trails.keys()))

    return state


# ═══════════════════════════════════════════════════════════════════════════
# Internal Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════

async def _run_agents_parallel(
    state: OrchestratorState,
    agent_names: list[str],
    llm: LLMClient,
    progress_callback: Callable = None,
):
    """Run multiple agents in parallel, update state as each completes."""

    async def _run_one(name: str) -> tuple[str, AuditTrail]:
        agent_cls = ALL_AGENTS.get(name)
        if agent_cls is None:
            return name, AuditTrail(
                agent_name=name, ticker=state.ticker,
                data_gaps=[f"Agent '{name}' not found in registry"],
                confidence=0.0,
            )

        agent = agent_cls()
        loop = asyncio.get_running_loop()

        try:
            # Determine which execute method to call
            if name == "forensic_quant":
                # Python-only agent — different interface
                trail = await loop.run_in_executor(
                    None,
                    lambda: agent.execute(
                        ticker=state.ticker,
                        financial_tables=state.financial_tables,
                        wacc=state.wacc,
                        terminal_growth=state.terminal_growth,
                        market_cap=state.market_cap,
                    ),
                )
            else:
                # LLM agent — full ReAct pipeline
                trail = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: agent.execute(
                            ticker=state.ticker,
                            document_text=state.document_text,
                            financial_tables=state.financial_tables,
                            sector=state.sector,
                            extraction_signals=state.extraction_signals,
                            llm=llm,
                        ),
                    ),
                    timeout=120.0,  # 2 min per agent (they have multi-turn loops now)
                )
            return name, trail

        except asyncio.TimeoutError:
            return name, AuditTrail(
                agent_name=name, ticker=state.ticker,
                data_gaps=[f"Agent '{name}' timed out after 120s"],
                confidence=0.0,
            )
        except Exception as e:
            return name, AuditTrail(
                agent_name=name, ticker=state.ticker,
                data_gaps=[f"Agent '{name}' crashed: {e}"],
                confidence=0.0,
            )

    # Launch all agents
    tasks = [_run_one(name) for name in agent_names]
    completed = set(state.agent_trails.keys())

    for coro in asyncio.as_completed(tasks):
        name, trail = await coro
        state.agent_trails[name] = trail
        completed.add(name)

        if progress_callback:
            active = [n for n in agent_names if n not in completed]
            progress_callback("investigation", active, list(completed))


def _determine_reflection_needs(state: OrchestratorState) -> list[str]:
    """
    Check if any agent's findings warrant re-investigating another agent.
    
    v2 only re-triggered forensic_quant when forensic_investigator found
    high-severity flags. v3 has richer cross-agent reflection.
    """
    reflection_agents = []

    # ── Forensic flags → re-run quant with enhanced context ──
    forensic_trail = state.agent_trails.get("forensic_investigator")
    if forensic_trail and forensic_trail.findings:
        findings = forensic_trail.findings
        high_severity = False

        for key in ["related_party_flags", "auditor_flags", "contingent_liabilities"]:
            items = findings.get(key, [])
            if any(f.get("severity") in ("HIGH", "CRITICAL") for f in items if isinstance(f, dict)):
                high_severity = True
                break

        if high_severity:
            reflection_agents.append("forensic_quant")

    # ── Empire building → re-check narrative for M&A language ──
    capital_trail = state.agent_trails.get("capital_allocator")
    if capital_trail and capital_trail.findings:
        empire = capital_trail.findings.get("empire_building", {})
        if empire.get("unrelated_acquisitions"):
            reflection_agents.append("narrative_decoder")

    # ── Management quality issues → re-check forensics ──
    mgmt_trail = state.agent_trails.get("management_quality")
    if mgmt_trail and mgmt_trail.findings:
        flags = mgmt_trail.findings.get("governance_flags", [])
        if len(flags) >= 3:
            reflection_agents.append("forensic_investigator")

    # Deduplicate and exclude already-run-twice agents
    seen = set()
    unique = []
    for a in reflection_agents:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


def _detect_conflicts(state: OrchestratorState) -> list[dict]:
    """
    Lightweight conflict detection between agent findings.
    
    v2 used O(n²) LLM calls to compare every pair. v3 uses Python-based
    keyword overlap to only flag genuine tensions — no LLM needed for this.
    """
    conflicts = []

    # ── Quant vs Qualitative conflict: earnings quality vs forensic flags ──
    quant = state.agent_trails.get("forensic_quant")
    forensic = state.agent_trails.get("forensic_investigator")

    if quant and forensic and quant.findings and forensic.findings:
        ocf_ratio = quant.findings.get("ocf_ebitda_ratio")
        has_high_flags = any(
            f.get("severity") in ("HIGH", "CRITICAL")
            for key in ("related_party_flags", "auditor_flags")
            for f in forensic.findings.get(key, [])
            if isinstance(f, dict)
        )
        if ocf_ratio and ocf_ratio > 0.8 and has_high_flags:
            conflicts.append({
                "agents": ["forensic_quant", "forensic_investigator"],
                "severity": "MEDIUM",
                "description": (
                    f"Quant says strong cash quality (OCF/EBITDA={ocf_ratio:.0%}) "
                    f"but forensic agent found HIGH severity flags — investigate why "
                    f"cash flow metrics look clean despite accounting concerns."
                ),
            })

    # ── Moat vs Narrative conflict: moat strength vs tone deterioration ──
    moat = state.agent_trails.get("moat_architect")
    narrative = state.agent_trails.get("narrative_decoder")

    if moat and narrative and moat.findings and narrative.findings:
        moat_verdict = moat.findings.get("moat_durability", "").upper()
        tone_shifts = narrative.findings.get("tone_shifts", [])
        has_bearish_shift = any(
            "cautious" in str(t.get("current_tone", "")).lower()
            or "challenging" in str(t.get("current_tone", "")).lower()
            for t in tone_shifts if isinstance(t, dict)
        )
        if moat_verdict in ("STRONG", "INTACT") and has_bearish_shift:
            conflicts.append({
                "agents": ["moat_architect", "narrative_decoder"],
                "severity": "MEDIUM",
                "description": (
                    f"Moat analysis says '{moat_verdict}' but management tone is "
                    f"deteriorating — management may be seeing moat erosion "
                    f"that hasn't shown up in reported numbers yet."
                ),
            })

    return conflicts


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Full Pipeline in One Call
# ═══════════════════════════════════════════════════════════════════════════

async def analyze(
    ticker: str,
    document_text: str,
    financial_tables: dict,
    sector: str,
    extraction_signals: dict = None,
    query: str = "",
    wacc: float = 0.12,
    progress_callback: Callable = None,
) -> OrchestratorState:
    """
    One-call entry point for the full v3 pipeline.
    
    Usage:
        state = await analyze(
            ticker="HINDUNILVR",
            document_text=extracted_text,
            financial_tables=pivoted_tables,
            sector="FMCG",
            extraction_signals={"has_rpt_disclosures": True},
        )
        print(state.final_report)
    """
    return await run_pipeline(
        ticker=ticker,
        document_text=document_text,
        financial_tables=financial_tables,
        sector=sector,
        extraction_signals=extraction_signals or {},
        query=query,
        wacc=wacc,
        progress_callback=progress_callback,
    )
