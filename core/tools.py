"""
novus_v3/core/tools.py — Tool Registry + Shared Financial Analysis Tools

Every v3 agent gets access to these document/data tools.
Agents can ALSO register their own specialized tools.

Design principle: The LLM decides WHAT to investigate.
Python computes the MATH. The LLM NARRATES the findings.
"""

import json
import re
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass
from rag_engine import query as rag_query


# ═══════════════════════════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict        # JSON Schema
    handler: Callable       # Python function to execute


class ToolRegistry:
    """Registry of callable tools available to an agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self                         # allow chaining

    def to_api_format(self) -> list[dict]:
        """Format for DeepSeek / OpenAI function-calling API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found"})
        try:
            result = tool.handler(**arguments)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, default=str)
            return str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Shared Tools — available to EVERY agent
# ═══════════════════════════════════════════════════════════════════════════

def build_shared_tools(document_text: str, financial_tables: dict, ticker: str = "") -> ToolRegistry:
    """
    Core toolkit every v3 agent gets.  Individual agents extend this
    with their own specialized tools via build_agent_tools().
    """
    reg = ToolRegistry()

    # ── 1. Document keyword search ────────────────────────────────────
    reg.register(Tool(
        name="search_document",
        description=(
            "Search the annual report / earnings transcript for a topic. "
            "Returns the 3 most relevant passages. "
            "Use for: 'related party transactions', 'auditor qualification', "
            "'goodwill impairment', 'contingent liabilities', 'segment revenue', "
            "'management guidance', 'capex plans', 'debt maturity profile', etc."
        ),
        parameters=_schema({
            "query":       ("string",  True,  "What to search for"),
            "max_results": ("integer", False, "1-5, default 3"),
        }),
        handler=lambda query, max_results=3: _search_doc(document_text, query, max_results, ticker),
    ))

    # ── 2. Get a specific page / note ─────────────────────────────────
    reg.register(Tool(
        name="get_page_content",
        description=(
            "Retrieve text around a page or note reference. "
            "Indian annual reports bury critical disclosures in notes 30-50. "
            "Use when you see 'Refer Note 42' or 'as per Schedule III'."
        ),
        parameters=_schema({
            "reference": ("string", True, "e.g. 'note 42', 'page 188', 'schedule III'"),
        }),
        handler=lambda reference: _get_page(document_text, reference),
    ))

    # ── 3. Financial line-item lookup ─────────────────────────────────
    reg.register(Tool(
        name="get_metric",
        description=(
            "Get a financial line item across all available years. "
            "Tables: profit_loss, balance_sheet, cash_flow. "
            "Supports fuzzy matching — e.g. 'Revenue' matches 'Revenue from Operations'."
        ),
        parameters=_schema({
            "line_item": ("string", True,  "e.g. 'Revenue from Operations', 'Trade Receivables'"),
            "table":     ("string", True,  "profit_loss | balance_sheet | cash_flow"),
        }),
        handler=lambda line_item, table: _get_metric(financial_tables, line_item, table),
    ))

    # ── 4. Python-computed ratio (Law 2 enforcer) ─────────────────────
    reg.register(Tool(
        name="compute_ratio",
        description=(
            "Compute a financial ratio using PYTHON. You MUST use this tool "
            "for ALL numerical calculations. Never compute ratios yourself. "
            "Example: compute_ratio('Other Income', 'Profit before tax', 'profit_loss', 'Mar 2024')"
        ),
        parameters=_schema({
            "numerator":   ("string", True,  "Numerator line item"),
            "denominator": ("string", True,  "Denominator line item"),
            "table":       ("string", True,  "profit_loss | balance_sheet | cash_flow"),
            "year":        ("string", True,  "e.g. 'Mar 2024'"),
        }),
        handler=lambda numerator, denominator, table, year: _compute_ratio(
            financial_tables, numerator, denominator, table, year
        ),
    ))

    # ── 5. Year-over-year comparison ──────────────────────────────────
    reg.register(Tool(
        name="compare_years",
        description=(
            "Compare a metric between two fiscal years. "
            "Returns values, absolute change, and % change — computed in Python."
        ),
        parameters=_schema({
            "metric": ("string", True,  "Line item name"),
            "year1":  ("string", True,  "Earlier year, e.g. 'Mar 2023'"),
            "year2":  ("string", True,  "Later year, e.g. 'Mar 2024'"),
            "table":  ("string", True,  "profit_loss | balance_sheet | cash_flow"),
        }),
        handler=lambda metric, year1, year2, table: _compare_years(
            financial_tables, metric, year1, year2, table
        ),
    ))

    # ── 6. Statistical anomaly detection ──────────────────────────────
    reg.register(Tool(
        name="detect_anomaly",
        description=(
            "Scan a line item across all years for anomalies: "
            "sudden spikes (>30% YoY), drops, or divergence from another item. "
            "Use to detect channel stuffing (receivables vs revenue divergence), "
            "aggressive capitalisation (CWIP vs depreciation), etc."
        ),
        parameters=_schema({
            "line_item":    ("string", True,  "Primary metric to scan"),
            "table":        ("string", True,  "profit_loss | balance_sheet | cash_flow"),
            "compare_with": ("string", False, "Optional second metric for divergence check"),
        }),
        handler=lambda line_item, table, compare_with=None: _detect_anomaly(
            financial_tables, line_item, table, compare_with
        ),
    ))

    # ── 7. Multi-year CAGR ────────────────────────────────────────────
    reg.register(Tool(
        name="compute_cagr",
        description=(
            "Compute the Compound Annual Growth Rate of a line item "
            "between two years. Returns percentage. Uses Python math."
        ),
        parameters=_schema({
            "line_item": ("string", True, "Line item name"),
            "table":     ("string", True, "profit_loss | balance_sheet | cash_flow"),
            "from_year": ("string", True, "Start year"),
            "to_year":   ("string", True, "End year"),
        }),
        handler=lambda line_item, table, from_year, to_year: _compute_cagr(
            financial_tables, line_item, table, from_year, to_year
        ),
    ))

    # ── 8. List available years and line items ────────────────────────
    reg.register(Tool(
        name="list_available_data",
        description=(
            "List all available years and line-item names for a given table. "
            "Call this FIRST if you're unsure what data is available."
        ),
        parameters=_schema({
            "table": ("string", True, "profit_loss | balance_sheet | cash_flow"),
        }),
        handler=lambda table: _list_available(financial_tables, table),
    ))

    return reg


# ═══════════════════════════════════════════════════════════════════════════
# Tool Implementations (Pure Python — no LLM involvement)
# ═══════════════════════════════════════════════════════════════════════════

def _schema(fields: dict) -> dict:
    """Shorthand for building JSON Schema parameters."""
    properties = {}
    required = []
    for name, (typ, req, desc) in fields.items():
        properties[name] = {"type": typ, "description": desc}
        if req:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _fuzzy_get(data: dict, key: str):
    """Try exact match, then case-insensitive substring match."""
    if key in data:
        return data[key]
    key_lower = key.lower()
    for k, v in data.items():
        if key_lower in k.lower() or k.lower() in key_lower:
            return v
    return None


def _search_doc(text: str, query: str, max_results: int = 3, ticker: str = "") -> list[dict]:
    """BM25-style keyword search over document paragraphs, with an option to use semantic RAG."""
    if ticker:
        results = rag_query(ticker, query, top_k=max_results)
        if results:
            return [{"passage": r["text"][:1000], "score": r["relevance"], "position": "rag_chunk"} for r in results]
    
    # Fallback to dumb BM25 search
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return [{"passage": "Empty query", "score": 0}]

    paragraphs = re.split(r'\n\s*\n', text)
    scored = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < 30:
            continue
        lower = para.lower()
        score = sum(lower.count(t) for t in terms)
        # Boost for exact phrase match
        if query.lower() in lower:
            score += 10
        if score > 0:
            scored.append((score, i, para))

    scored.sort(key=lambda x: -x[0])
    return [
        {"passage": p[:1000], "score": s, "position": f"para_{idx}"}
        for s, idx, p in scored[:max_results]
    ] or [{"passage": "No relevant content found.", "score": 0}]


def _get_page(text: str, reference: str) -> dict:
    ref = reference.lower().strip()
    # Try multiple patterns
    for pattern in [ref, ref.replace("note ", "note no. "), ref.replace("note ", "notes ")]:
        idx = text.lower().find(pattern)
        if idx != -1:
            start = max(0, idx - 200)
            end = min(len(text), idx + 3000)
            return {"content": text[start:end], "found": True, "position": idx}
    return {"content": f"'{reference}' not found in document.", "found": False}


def _get_metric(tables: dict, line_item: str, table: str) -> dict:
    tbl = tables.get(table, {})
    results = {}
    for year, items in tbl.items():
        if isinstance(items, dict):
            val = _fuzzy_get(items, line_item)
            if val is not None:
                results[year] = val
    return {"line_item": line_item, "table": table, "values": results}


def _compute_ratio(tables: dict, num: str, den: str, table: str, year: str) -> dict:
    tbl = tables.get(table, {})
    year_data = tbl.get(year, {})
    n = _fuzzy_get(year_data, num)
    d = _fuzzy_get(year_data, den)
    if n is None or d is None:
        return {"error": f"Missing: {num}={n}, {den}={d}", "year": year}
    if d == 0:
        return {"error": "Denominator is zero", "numerator_value": n}
    ratio = round(n / d, 4)
    return {
        "numerator": num, "n_value": n,
        "denominator": den, "d_value": d,
        "ratio": ratio, "pct": f"{ratio*100:.2f}%", "year": year,
    }


def _compare_years(tables: dict, metric: str, y1: str, y2: str, table: str) -> dict:
    tbl = tables.get(table, {})
    v1 = _fuzzy_get(tbl.get(y1, {}), metric)
    v2 = _fuzzy_get(tbl.get(y2, {}), metric)
    result = {"metric": metric, "year1": y1, "val1": v1, "year2": y2, "val2": v2}
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 != 0:
        result["abs_change"] = round(v2 - v1, 2)
        result["pct_change"] = round(((v2 - v1) / abs(v1)) * 100, 2)
    return result


def _detect_anomaly(tables: dict, item: str, table: str, compare: str = None) -> dict:
    data = _get_metric(tables, item, table)
    vals = data.get("values", {})
    years = sorted(vals.keys())
    anomalies = []
    for i in range(1, len(years)):
        p, c = vals.get(years[i-1]), vals.get(years[i])
        if isinstance(p, (int, float)) and isinstance(c, (int, float)) and p != 0:
            chg = ((c - p) / abs(p)) * 100
            if abs(chg) > 30:
                anomalies.append({
                    "year": years[i], "prev": p, "curr": c,
                    "change_pct": round(chg, 1),
                    "type": "SPIKE" if chg > 30 else "DROP",
                })

    result = {"line_item": item, "values": vals, "anomalies": anomalies}

    if compare:
        comp = _get_metric(tables, compare, table).get("values", {})
        ratios = []
        for y in years:
            v1, v2 = vals.get(y), comp.get(y)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v2 != 0:
                ratios.append({"year": y, "ratio": round(v1 / v2, 4)})
        if len(ratios) >= 2:
            first, last = ratios[0]["ratio"], ratios[-1]["ratio"]
            drift = ((last - first) / abs(first)) * 100 if first else 0
            result["divergence"] = {
                "vs": compare, "ratios": ratios,
                "drift_pct": round(drift, 1), "is_diverging": abs(drift) > 20,
            }
    return result


def _compute_cagr(tables: dict, item: str, table: str, y1: str, y2: str) -> dict:
    data = _get_metric(tables, item, table).get("values", {})
    v1, v2 = data.get(y1), data.get(y2)
    if not (isinstance(v1, (int, float)) and isinstance(v2, (int, float))):
        return {"error": f"Missing data: {y1}={v1}, {y2}={v2}"}
    if v1 <= 0 or v2 <= 0:
        return {"error": "Cannot compute CAGR with non-positive values"}
    # Estimate years between
    try:
        yr1 = int(re.search(r'(\d{4})', y1).group(1))
        yr2 = int(re.search(r'(\d{4})', y2).group(1))
        n = yr2 - yr1
    except (AttributeError, ValueError):
        n = 1
    if n <= 0:
        return {"error": f"Invalid year range: {y1} to {y2}"}
    cagr = ((v2 / v1) ** (1 / n) - 1) * 100
    return {"cagr_pct": round(cagr, 2), "from": v1, "to": v2, "years": n}


def _list_available(tables: dict, table: str) -> dict:
    tbl = tables.get(table, {})
    years = sorted(tbl.keys())
    items = set()
    for yr_data in tbl.values():
        if isinstance(yr_data, dict):
            items.update(yr_data.keys())
    return {"table": table, "years": years, "line_items": sorted(items)}
