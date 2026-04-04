"""
novus_v3/orchestrator/cio_v3.py — v3 CIO Orchestrator

Coordinates all v3 agents with:
  0. Lead Analyst Briefing: Generates dynamic, context-aware frameworks for each agent based on client mandate and macro reality.
  1. Parallel execution of independent agents using the dynamic frameworks.
  2. Data routing: quant agents get structured data, LLM agents get tools.
  3. Reflection: high-severity findings re-trigger relevant agents.
  4. Conflict detection: cross-check agent findings for contradictions.
  5. Synthesis: PM agent merges everything into a single thesis.
  6. Full audit trail preserved for every step.
"""

import json
import time
import asyncio
import re
from typing import Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Dedicated ThreadPool to ensure 10 agents can run simultaneously 
# without queuing behind other default executor tasks.
_agent_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="AgentPool")

from core.llm_client import LLMClient, get_llm_client
from core.agent_base_v3 import AuditTrail, AgentV3
from agents.all_agents import (
    ForensicInvestigatorV3,
    NarrativeDecoderV3,
    MoatArchitectV3,
    CapitalAllocatorV3,
    ManagementQualityV3,
    ForensicQuantV3,
    PMSynthesisV3,
    ALL_AGENTS,
)


@dataclass
class OrchestratorState:
    ticker: str
    sector: str
    query: str
    fiscal_year: str = ""

    client_profile: str = "Standard Institutional Mandate: Focus on sustainable growth, reasonable valuations, and clean accounting."
    macro_context: str = "Neutral macroeconomic environment."
    agent_frameworks: dict = field(default_factory=dict) 

    document_text: str = ""
    financial_tables: dict = field(default_factory=dict)
    extraction_signals: dict = field(default_factory=dict)

    wacc: float = 0.12
    terminal_growth: float = 0.05
    market_cap: Optional[float] = None

    agent_trails: dict[str, AuditTrail] = field(default_factory=dict)
    conflicts: list[dict] = field(default_factory=list)

    final_thesis: Optional[AuditTrail] = None
    final_report: str = ""


EXECUTION_PHASES = [
    {
        "phase": "investigation",
        "parallel": True,
        "agents": [
            "forensic_quant",          
            "forensic_investigator",   
            "narrative_decoder",       
            "moat_architect",          
            "capital_allocator",       
            "management_quality",      
        ],
    },
    {
        "phase": "reflection",
        "parallel": False,
        "agents": [],   
    },
    {
        "phase": "synthesis",
        "parallel": False,
        "agents": ["pm_synthesis"],
    },
]


async def _generate_dynamic_frameworks(state: OrchestratorState, llm: LLMClient) -> dict:
    prompt = f"""You are the Director of Research for an Indian Equity Fund.
Target Company: {state.ticker} ({state.sector})
Client Mandate: {state.client_profile}
Current India Macro Reality: {state.macro_context}
Specific User Query: {state.query}

We are dispatching 5 qualitative agents to analyze this company's filings. 
Based entirely on the target sector, the macroeconomic context, and the client's specific mandate, write a strict 2-3 sentence 'custom_framework' (focus area) for EACH agent.

Rule: If the client is conservative, instruct the forensic agent to lower materiality thresholds. If the macro is inflationary, instruct the moat architect to heavily scrutinize pricing power and raw material pass-through. Give specific, tailored directions.

Output valid JSON ONLY matching this exact structure:
{{
  "forensic_investigator": "focus instructions...",
  "narrative_decoder": "focus instructions...",
  "moat_architect": "focus instructions...",
  "capital_allocator": "focus instructions...",
  "management_quality": "focus instructions..."
}}"""

    try:
        response = await asyncio.to_thread(
            llm.call_simple,
            "You are a Lead Analyst at a top-tier institutional equity fund. Output valid JSON only.",
            prompt,
        )
        clean = response.strip()
        if '```json' in clean:
            clean = clean.split('```json', 1)[1].rsplit('```', 1)[0]
        elif '```' in clean:
            clean = clean.split('```', 1)[1].rsplit('```', 1)[0]
            
        frameworks = json.loads(clean.strip())
        return frameworks
    except Exception as e:
        print(f"> [CIO] ⚠️ Lead Analyst failed to generate dynamic frameworks: {e}")
        return {}


async def run_pipeline(
    ticker: str,
    document_text: str,
    financial_tables: dict,
    sector: str,
    extraction_signals: dict,
    query: str = "",
    client_profile: str = "Standard Institutional Mandate",
    macro_context: str = "Neutral macroeconomic environment",
    wacc: float = 0.12,
    terminal_growth: float = 0.05,
    market_cap: float = None,
    progress_callback: Callable = None,
    llm: LLMClient = None,
) -> OrchestratorState:
    
    # ── Dual LLM Routing ──
    # V3: Fast, structured tool-calling — for all ReAct investigation agents
    # R1: Deep chain-of-thought reasoning — exclusively for PM Synthesis
    v3_llm = get_llm_client(use_r1=False)
    r1_llm = get_llm_client(use_r1=True)
    print(f"> [CIO] Model routing: ReAct agents → V3 | PM Synthesis → R1")

    state = OrchestratorState(
        ticker=ticker,
        sector=sector,
        query=query,
        client_profile=client_profile,
        macro_context=macro_context,
        document_text=document_text,
        financial_tables=financial_tables,
        extraction_signals=extraction_signals,
        wacc=wacc,
        terminal_growth=terminal_growth,
        market_cap=market_cap,
    )

    if progress_callback:
        progress_callback("lead_analyst_planning", [], [])
        
    state.agent_frameworks = await _generate_dynamic_frameworks(state, v3_llm)
    print(f"> [CIO] Lead Analyst generated frameworks for {len(state.agent_frameworks)} agents.")

    phase1 = EXECUTION_PHASES[0]
    phase1_agents = phase1["agents"].copy()

    # ── STAGED BLACKBOARD: Run Quant First ──
    if "forensic_quant" in phase1_agents:
        if progress_callback:
            progress_callback("investigation", ["forensic_quant"], [])
        await _run_agents_parallel(state, ["forensic_quant"], v3_llm, progress_callback)
        
        # Cross-pollinate the anomalies into the Prompt Composer's dynamic mandate
        quant_trail = state.agent_trails.get("forensic_quant")
        if quant_trail and quant_trail.findings:
            anomaly = quant_trail.findings.get("anomaly_flag")
            if anomaly:
                alert = (
                    f"\n\n## CRITICAL QUANT ALERT\n"
                    f"The Quantitative engine flagged an anomaly: {anomaly}\n"
                    f"Prioritize investigating this phenomenon."
                )
                for name in phase1_agents:
                    if name != "forensic_quant":
                        state.agent_frameworks[name] = state.agent_frameworks.get(name, "") + alert
                        
        phase1_agents.remove("forensic_quant")

    if phase1_agents:
        if progress_callback:
            progress_callback("investigation", phase1_agents, list(state.agent_trails.keys()))
        await _run_agents_parallel(state, phase1_agents, v3_llm, progress_callback)

    reflection_agents = _determine_reflection_needs(state)
    if reflection_agents:
        if progress_callback:
            progress_callback("reflection", reflection_agents, list(state.agent_trails.keys()))
        await _run_agents_parallel(state, reflection_agents, v3_llm, progress_callback)

    if progress_callback:
        progress_callback("conflict_check", [], list(state.agent_trails.keys()))
    state.conflicts = _detect_conflicts(state)

    if progress_callback:
        progress_callback("synthesis", ["pm_synthesis"], list(state.agent_trails.keys()))

    agent_outputs = {}
    for name, trail in state.agent_trails.items():
        if trail.findings:
            agent_outputs[name] = trail.findings
    if state.conflicts:
        agent_outputs["_conflicts"] = state.conflicts

    pm_signals = {**extraction_signals, "_agent_outputs": agent_outputs}
    
    pm = PMSynthesisV3()
    thesis_trail = await asyncio.to_thread(
        pm.execute,
        ticker=ticker,
        document_text=document_text,  
        financial_tables=financial_tables,
        sector=sector,
        extraction_signals=pm_signals,
        llm=r1_llm,  # R1 for deep reasoning synthesis
        dynamic_mandate=state.agent_frameworks.get("pm_synthesis", "")
    )
    
    state.agent_trails["pm_synthesis"] = thesis_trail
    state.final_thesis = thesis_trail
    state.final_report = thesis_trail.to_analyst_note() if hasattr(thesis_trail, 'to_analyst_note') else str(thesis_trail.findings)

    if progress_callback:
        progress_callback("complete", [], list(state.agent_trails.keys()))

    return state


async def _run_agents_parallel(
    state: OrchestratorState,
    agent_names: list[str],
    llm: LLMClient,
    progress_callback: Callable = None,
):
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
        custom_mandate = state.agent_frameworks.get(name, "")

        try:
            if name == "forensic_quant":
                trail = await loop.run_in_executor(
                    _agent_executor,
                    lambda: agent.execute(
                        ticker=state.ticker,
                        financial_tables=state.financial_tables,
                        wacc=state.wacc,
                        terminal_growth=state.terminal_growth,
                        market_cap=state.market_cap,
                    ),
                )
            else:
                trail = await asyncio.wait_for(
                    loop.run_in_executor(
                        _agent_executor,
                        lambda: agent.execute(
                            ticker=state.ticker,
                            document_text=state.document_text,
                            financial_tables=state.financial_tables,
                            sector=state.sector,
                            extraction_signals=state.extraction_signals,
                            llm=llm,
                            dynamic_mandate=custom_mandate, 
                        ),
                    ),
                    timeout=180.0, 
                )
            return name, trail

        except asyncio.TimeoutError:
            return name, AuditTrail(
                agent_name=name, ticker=state.ticker,
                data_gaps=[f"Agent '{name}' timed out after 180s"],
                confidence=0.0,
            )
        except Exception as e:
            return name, AuditTrail(
                agent_name=name, ticker=state.ticker,
                data_gaps=[f"Agent '{name}' crashed: {e}"],
                confidence=0.0,
            )

    tasks = [_run_one(name) for name in agent_names]
    completed = set(state.agent_trails.keys())

    for coro in asyncio.as_completed(tasks):
        name, trail = await coro
        state.agent_trails[name] = trail
        completed.add(name)
        print(f"> [CIO] ✅ Agent {name} COMPLETED (conf: {trail.confidence})")

        if progress_callback:
            from utils.formatters import format_dict_as_markdown
            active = [n for n in agent_names if n not in completed]
            agent_outputs = {}
            for n, t in state.agent_trails.items():
                if t.findings:
                    agent_outputs[n] = "\n".join(format_dict_as_markdown(t.findings, indent=0))
                elif t.data_gaps:
                    agent_outputs[n] = "**Data Gaps:**\n" + "\n".join(f"- {g}" for g in t.data_gaps)
            
            progress_callback("investigation", active, list(completed), agent_outputs=agent_outputs)


def _determine_reflection_needs(state: OrchestratorState) -> list[str]:
    reflection_agents = []

    forensic_trail = state.agent_trails.get("forensic_investigator")
    if forensic_trail and forensic_trail.findings and isinstance(forensic_trail.findings, dict):
        high_severity = False
        for key in ["related_party_flags", "auditor_flags", "contingent_liabilities"]:
            items = forensic_trail.findings.get(key, [])
            if any(isinstance(f, dict) and f.get("severity") in ("HIGH", "CRITICAL") for f in items):
                high_severity = True
                break
        if high_severity:
            reflection_agents.append("forensic_quant")

    capital_trail = state.agent_trails.get("capital_allocator")
    if capital_trail and capital_trail.findings and isinstance(capital_trail.findings, dict):
        empire = capital_trail.findings.get("empire_building", {})
        if isinstance(empire, dict) and empire.get("unrelated_acquisitions"):
            reflection_agents.append("narrative_decoder")

    mgmt_trail = state.agent_trails.get("management_quality")
    if mgmt_trail and mgmt_trail.findings and isinstance(mgmt_trail.findings, dict):
        flags = mgmt_trail.findings.get("governance_flags", [])
        if isinstance(flags, list) and len(flags) >= 3:
            reflection_agents.append("forensic_investigator")

    return list(dict.fromkeys(reflection_agents)) 


def _detect_conflicts(state: OrchestratorState) -> list[dict]:
    conflicts = []

    quant = state.agent_trails.get("forensic_quant")
    forensic = state.agent_trails.get("forensic_investigator")

    if quant and forensic and isinstance(quant.findings, dict) and isinstance(forensic.findings, dict):
        ocf_ratio = quant.findings.get("ocf_ebitda_ratio")
        has_high_flags = any(
            isinstance(f, dict) and f.get("severity") in ("HIGH", "CRITICAL")
            for key in ("related_party_flags", "auditor_flags")
            for f in forensic.findings.get(key, [])
        )
        if isinstance(ocf_ratio, (int, float)) and ocf_ratio > 0.8 and has_high_flags:
            conflicts.append({
                "agents": ["forensic_quant", "forensic_investigator"],
                "severity": "MEDIUM",
                "description": f"Quant says strong cash quality (OCF/EBITDA={ocf_ratio:.0%}) but forensic agent found HIGH severity accounting flags.",
            })

    moat = state.agent_trails.get("moat_architect")
    narrative = state.agent_trails.get("narrative_decoder")

    if moat and narrative and isinstance(moat.findings, dict) and isinstance(narrative.findings, dict):
        moat_verdict = str(moat.findings.get("moat_durability", "")).upper()
        tone_shifts = narrative.findings.get("tone_shifts", [])
        
        has_bearish_shift = any(
            isinstance(t, dict) and ("cautious" in str(t.get("current_tone", "")).lower() or "challenging" in str(t.get("current_tone", "")).lower())
            for t in tone_shifts if isinstance(t, dict)
        )
        if moat_verdict in ("STRONG", "INTACT") and has_bearish_shift:
            conflicts.append({
                "agents": ["moat_architect", "narrative_decoder"],
                "severity": "MEDIUM",
                "description": f"Moat analysis says '{moat_verdict}' but management tone is actively deteriorating in concalls.",
            })

    return conflicts


async def analyze(
    ticker: str,
    document_text: str,
    financial_tables: dict,
    sector: str,
    extraction_signals: dict = None,
    query: str = "",
    client_profile: str = "Standard Institutional Mandate",
    macro_context: str = "Neutral macroeconomic environment",
    wacc: float = 0.12,
    progress_callback: Callable = None,
) -> OrchestratorState:
    
    return await run_pipeline(
        ticker=ticker,
        document_text=document_text,
        financial_tables=financial_tables,
        sector=sector,
        extraction_signals=extraction_signals or {},
        query=query,
        client_profile=client_profile,
        macro_context=macro_context,
        wacc=wacc,
        progress_callback=progress_callback,
    )
