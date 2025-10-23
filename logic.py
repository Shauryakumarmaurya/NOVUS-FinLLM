#logic.py
import os
import google.generativeai as genai
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time 
import json as _json
import re as _re
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
gemini_api_key = os.getenv("GEMINI_API_KEY") # Gemini API key from environment variable
fmp_api_key = os.getenv("FMP_API_KEY") # Financial Modeling Prep API key (if needed)
# Enable verbose Gemini input/output logging only when explicitly requested
_ENABLE_GEMINI_DEBUG_LOGS = os.getenv("ENABLE_GEMINI_DEBUG_LOGS", "false").lower() in ("1", "true", "yes", "on")

# Configure the Gemini API client
genai.configure(api_key=gemini_api_key) 
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite') # Using a modern, efficient model

# A dictionary to hold all prompts for the AI model.
PROMPTS = {
    "financial_assumptions":  "You are an equity analyst. Based *only* on the qualitative management commentary, guidance, and risks mentioned in the provided text, generate financial assumptions for the next three fiscal years. Do not perform any calculations. Provide the output as a clean JSON object ONLY, with no other text or explanations. The structure must be exactly: {\"base_case\": {\"revenue_growth_cagr\": 15, \"ebitda_margin\": 28.5, \"tax_rate\": 25}, \"bull_case\": {\"revenue_growth_cagr\": 20, \"ebitda_margin\": 29.5, \"tax_rate\": 25}, \"bear_case\": {\"revenue_growth_cagr\": 10, \"ebitda_margin\": 27.0, \"tax_rate\": 25}}. Use reasonable, text-supported estimates for growth, margin, and tax rates.",# old version changed into new workflow
    "detailed_financial_assumptions_text": "You are an expert equity analyst. Below are historical financial statements and recent earnings call transcripts for a company. Your task is to generate a detailed qualitative analysis for the Bull, Base, and Bear case financial assumptions for the next three years. Use the historical data as a quantitative baseline and the transcript commentary to justify any acceleration, deceleration, or changes in trends. IMPORTANT: Before the content of each case section, output EXACTLY this cue on its own line so the app can split sections: ##NEWSECTION{base_case}## for Base Case, ##NEWSECTION{bull_case}## for Bull Case, and ##NEWSECTION{bear_case}## for Bear Case. Then immediately provide that case's analysis. Do not include any other text between the cues and the content. Within each section, list the key metrics (revenue_growth_cagr, ebitda_margin, tax_rate), their projected values, and a detailed justification, along with a confidence score (out of 10) for each number you predict, cite specific quotes from the text AND reference historical data points where relevant. Keep your justifications short(max 50 words per metric).",
    "extract_assumptions_json": "bosed *ONLY* the text provided below, extract the financial assumptions for the base_case, bull_case, and bear_case. Provide the output as a clean JSON object ONLY, with no other text, markdown, or explanations.In case there is a range output the midpoint of the range. The structure must be exactly:{\"base_case\": {\"revenue_growth_cagr\": 15, \"ebitda_margin\": 28.5, \"tax_rate\": 25}, \"bull_case\": {\"revenue_growth_cagr\": 20, \"ebitda_margin\": 29.5, \"tax_rate\": 25}, \"bear_case\": {\"revenue_growth_cagr\": 10, \"ebitda_margin\": 27.0, \"tax_rate\": 25}}. Use reasonable, text-supported estimates for growth, margin, and tax rates.",
    "business_model": "You are an equity research analyst. Summarize the company's business model in simple, investor-friendly terms based on the provided text. Include these exact markdown headings: \"📌 Core Products & Services\", \"🎯 Target Markets / Customers\", \"💸 Revenue Model & Geography\", and \"📈 Scale & Competitive Positioning\".",
    "key_quarterly_updates": "Extract the 5-7 most important operational or financial updates from this concall text. Focus on: Growth drivers, Orders/capacity/margins, Strategy changes, and direct Quotes or signals from management. Present as a bulleted list in Markdown.",
    "management_commentary": "Summarize management's guidance and tone for the next 1-2 quarters from the provided text. Format your answer in Markdown under these exact headings: \"🔹 Forward-Looking Statements\", \"🔹 Management Tone & Confidence\" (classify tone as Optimistic/Cautious/Neutral and support with quotes), and \"🔹 Capex / Risk / Guidance Highlights\".",
    "risks_uncertainties": "List the key risks or uncertainties based on the concall text. Categorize them under these exact Markdown headings if possible: \"Execution Risks\", \"Demand-side or Macro Risks\", and \"Regulatory / External Risks\".",
    "prompt_set": "Based on the provided concall text, generate 3-5 company-specific, smart, and non-generic prompts an investor could ask an LLM to explore further."
}

# --- Core Logic Functions ---
# --- New Function for Chat-Based Report Generation ---(not currently used)---
def generate_report_in_chat(uploaded_files):
    chat = gemini_model.start_chat()
    results = {}
    
    # --- Message 1 ---
    print("Getting management commentary...")
    response1 = chat.send_message([PROMPTS["management_commentary"]] + uploaded_files)
    results["managementCommentary"] = response1.text
    time.sleep(5) # 2. Pause for 5 seconds

    # --- Message 2 ---
    print("Getting key quarterly updates...")
    response2 = chat.send_message(PROMPTS["key_quarterly_updates"])
    results["keyQuarterlyUpdates"] = response2.text
    time.sleep(5) # 2. Pause for 5 seconds
     
    # --- message 3 ---
    # --- Message  (The final JSON request) ---
    print("Generating financial assumptions...")
    final_prompt = "Based only on the qualitative management commentary..." # Use the forceful prompt
    response3 = chat.send_message(final_prompt)
    results["financialAssumptions"] = response3.text
    # No need to sleep after the last call

    # ... continue this pattern for all your steps ...

    return results


# not being used currently
def _parse_statements_from_soup(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """Helper function to extract tables from a BeautifulSoup object."""
    # Find the Profit & Loss table
    profit_loss_section = soup.find('section', id='profit-loss')
    profit_loss_html = None
    if profit_loss_section:
        profit_loss_table = profit_loss_section.find('table', class_='data-table')
        if profit_loss_table:
            profit_loss_html = str(profit_loss_table)
    
    # Find the Balance Sheet table
    balance_sheet_section = soup.find('section', id='balance-sheet')
    balance_sheet_html = None
    if balance_sheet_section:
        balance_sheet_table = balance_sheet_section.find('table', class_='data-table')
        if balance_sheet_table:
            balance_sheet_html = str(balance_sheet_table)

    if not profit_loss_html:
        print("Warning: Profit & Loss section/table not found on the page.")
    if not balance_sheet_html:
        print("Warning: Balance Sheet section/table not found on the page.")
        
    return profit_loss_html, balance_sheet_html
# --- Updated Function with Robustness Improvements ---
def get_yearly_financial_statements_html(ticker: str) -> tuple[str | None, str | None]:
    """
    Fetches and extracts yearly financial statements for a given ticker.
    It automatically tries both consolidated and standalone URLs and validates
    that the found tables actually contain data.
    """
    urls_to_try = [
        f"https://www.screener.in/company/{ticker}/consolidated/",
        f"https://www.screener.in/company/{ticker}/"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for url in urls_to_try:
        try:
            print(f"Attempting to fetch data from: {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            pl_html, bs_html = _parse_statements_from_soup(soup)
            
            # If we found the HTML for the tables, proceed to validate them
            if pl_html and bs_html:
                # --- NEW VALIDATION STEP ---
                # Check if the extracted table actually has data (year columns)
                # by doing a quick test extraction.
                pl_data = extract_financial_data_from_html(pl_html)
                
                if pl_data and pl_data.keys():
                    print("✅ Successfully found valid financial statements.")
                    return pl_html, bs_html
                else:
                    # This happens if the table exists but is empty
                    print("⚠️ Found tables, but they contain no data. Trying next URL...")
            else:
                # This happens if the <section> or <table> tags weren't found
                print("Could not find the required tables, trying next URL...")

        except requests.exceptions.RequestException as e:
            print(f"❌ A network error occurred for {url}: {e}. Trying next URL...")
            continue
    
    print(f"Error: Failed to retrieve valid data for {ticker} from all attempted sources.")
    return None, None

def extract_financial_data_from_html(html: str) -> dict[str, dict[str, float]]:
    """
    Parses the HTML of a financial statement table and extracts the data.
    (This function is good as is, no changes needed).
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', class_='data-table')
    
    if not table:
        print("Error: No data table found in the provided HTML.")
        return {}

    # Extract headers (years)
    headers = table.find('thead').find_all('th')[1:]
    years = [header.get_text(strip=True) for header in headers]

    data = {year: {} for year in years}

    # Extract rows
    rows = table.find('tbody').find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        if not cols or len(cols) < 2:
            continue

        item_name = cols[0].get_text(strip=True).replace('.', '')
        if not item_name:
            continue

        for i, col in enumerate(cols[1:]):
            if i < len(years):
                year = years[i]
                value_str = col.get_text(strip=True).replace(',', '').replace('₹', '')
                try:
                    value = float(value_str) if value_str not in ('-', '') else 0.0
                except ValueError:
                    value = 0.0
                data[year][item_name] = value

    return data


# best data orgzanization tool
def compact_financial_data_for_llm(pl_input, bs_input, years_count: int | None = None, return_json: bool = True):
    """
    Build a compact, clean JSON of financial data for LLM consumption using pandas & numpy.

    Inputs can be either:
    - HTML strings (as scraped from Screener tables), or
    - Dicts shaped like: { "YearLabel": { "Metric": value, ... }, ... } returned by extract_financial_data_from_html.

    Behavior:
    - Normalizes year columns; parses numeric years from labels like 'Mar 2023'.
    - If a TTM column exists, it is mapped to (max_year + 1) to keep chronological order stable years later.
    - Drops OPM% row if it exists and is all zeros across years.
    - Sorts columns by ascending year for both P&L and Balance Sheet independently.
        - Returns compact JSON with separate sections for profit_and_loss and balance_sheet.
        - years_count: number of historical years to include (most recent N), plus TTM if available.
            Example: years_count=4 -> last 4 years + TTM.

    return_json=True returns a JSON string; otherwise returns a Python dict.
    """

    def _ensure_dict(data_or_html):
        # Convert HTML to dict via existing extractor if needed
        if isinstance(data_or_html, str):
            return extract_financial_data_from_html(data_or_html)
        if isinstance(data_or_html, dict):
            return data_or_html
        return {}

    def _normalize_item_name(name: str) -> str:
        # Clean up item names: remove trailing '+' and extra spaces, unify percent formatting
        if not isinstance(name, str):
            return name
        n = name.replace('\u00a0', ' ').strip()  # non-breaking spaces
        n = n.rstrip('+').strip()
        # Standardize percent labels spacing
        n = n.replace(' %', '%')
        n = n.replace('% ', '%')
        return n

    def _dict_to_dataframe(year_to_items: dict) -> pd.DataFrame:
        # Convert dict-of-years -> items-> values into DataFrame with rows as items, cols as years
        if not year_to_items:
            return pd.DataFrame()
        # Build a DataFrame by combining per-year series
        frames = []
        for year_label, items in year_to_items.items():
            if not isinstance(items, dict):
                continue
            s = pd.Series(items, name=str(year_label))
            frames.append(s)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=1)
        # Normalize item names in the index
        df.index = [ _normalize_item_name(ix) for ix in df.index ]
        # Coerce to numeric values
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

    def _parse_year(label: str) -> tuple[int | None, bool]:
        """Return (year_int, is_ttm). If TTM detected, (None, True)."""
        if label is None:
            return None, False
        lbl = str(label).strip()
        if lbl.upper() == 'TTM' or 'TTM' in lbl.upper():
            return None, True
        # Extract a 4-digit year (handles 'Mar 2023' etc.)
        m = _re.search(r'(19|20)\d{2}', lbl)
        if m:
            return int(m.group(0)), False
        return None, False

    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        cols = list(df.columns)
        years = []
        ttm_present = False
        for c in cols:
            y, is_ttm = _parse_year(c)
            if is_ttm:
                ttm_present = True
            else:
                years.append(y)
        # Filter valid years
        years = [y for y in years if isinstance(y, int)]
        if not years:
            # If we can't parse any years, keep as-is but ensure strings
            df.columns = [str(c) for c in df.columns]
            return df
        max_year = max(years)
        # Build mapping: original -> normalized year label
        col_map = {}
        for c in cols:
            y, is_ttm = _parse_year(c)
            if is_ttm:
                # Map TTM to max_year + 1
                col_map[c] = str(max_year + 1)
            elif isinstance(y, int):
                col_map[c] = str(y)
            else:
                # Unknown label: leave as string (placed before sorting end)
                col_map[c] = str(c)
        df = df.rename(columns=col_map)
        # Sort columns numerically; non-integers will be placed at the end preserving order
        def _key(c):
            try:
                return (0, int(c))
            except Exception:
                return (1, c)
        # Drop duplicate columns keeping the last occurrence (e.g., multiple TTMs over time)
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        df = df.reindex(columns=sorted(df.columns, key=_key))
        # Store which column is TTM for downstream labeling
        if ttm_present:
            df.attrs['ttm_col'] = str(max_year + 1)
        else:
            df.attrs['ttm_col'] = None
        return df

    def _drop_opm_if_all_zero(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Possible row labels variants: 'OPM%', 'OPM %', 'OPM % (Operating Profit Margin)'
        idx_matches = [ix for ix in df.index if 'OPM' in str(ix).upper() and '%' in str(ix)]
        for ix in idx_matches:
            row = df.loc[ix]
            if np.nan_to_num(row.values.astype(float), nan=0.0).sum() == 0.0:
                df = df.drop(index=ix)
        return df

    def _to_clean_json(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"years": [], "data": {}}
        # Ensure numeric columns sorted
        df_norm = _normalize_columns(df)
        # Replace +/- inf with NaN then to None for JSON
        df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
        # Determine TTM column (if any)
        ttm_col = df_norm.attrs.get('ttm_col', None)
        # Build list of numeric year columns excluding TTM
        numeric_cols = []
        for c in df_norm.columns:
            try:
                ci = int(c)
                if ttm_col is not None and str(c) == str(ttm_col):
                    continue
                numeric_cols.append(ci)
            except Exception:
                # ignore non-numeric labels
                pass
        numeric_cols = sorted(set(numeric_cols))
        # Select last N years if requested
        if years_count is not None and years_count > 0 and numeric_cols:
            selected_years = numeric_cols[-years_count:]
        else:
            selected_years = numeric_cols
        # Compose final column order: selected years (as strings), then TTM (if present)
        final_cols = [str(y) for y in selected_years]
        if ttm_col is not None and str(ttm_col) in df_norm.columns:
            final_cols.append(str(ttm_col))
        # Reindex to selected columns if we reduced anything
        if set(final_cols) != set(df_norm.columns):
            # Keep only those that exist
            existing = [c for c in final_cols if c in df_norm.columns]
            df_norm = df_norm.reindex(columns=existing)
        years_order = list(df_norm.columns)
        # Convert to pure Python types; keep floats/ints
        data_obj = {}
        for item in df_norm.index:
            series = df_norm.loc[item]
            values = []
            for v in series.tolist():
                if pd.isna(v):
                    values.append(None)
                else:
                    # Cast very small -0.0 to 0.0 cleanly
                    vv = float(v)
                    if abs(vv) < 1e-12:
                        vv = 0.0
                    values.append(vv)
            data_obj[str(item)] = values
        # Build output years with TTM label for the final column if applicable
        years_cast = []
        for idx, y in enumerate(years_order):
            if ttm_col is not None and str(y) == str(ttm_col):
                years_cast.append('TTM')
            else:
                try:
                    years_cast.append(int(y))
                except Exception:
                    years_cast.append(str(y))
        return {"years": years_cast, "data": data_obj}

    # 1) Convert inputs to DataFrames
    pl_dict = _ensure_dict(pl_input)
    bs_dict = _ensure_dict(bs_input)

    pl_df = _dict_to_dataframe(pl_dict)
    bs_df = _dict_to_dataframe(bs_dict)

    # 2) Clean up: drop OPM% rows if all zeros (P&L mostly)
    pl_df = _drop_opm_if_all_zero(pl_df)

    # 3) Normalize columns (years incl. TTM -> max_year + 1) and sort
    pl_json = _to_clean_json(pl_df)
    bs_json = _to_clean_json(bs_df)

    result = {
        "profit_and_loss": pl_json,
        "balance_sheet": bs_json,
    }

    if return_json:
        try:
            return _json.dumps(result, ensure_ascii=False)
        except Exception:
            # Fallback if any non-serializable sneaks in
            return _json.dumps(_json.loads(_json.dumps(result, default=str)))
    return result


    """
    Calculates future financial projections based on AI assumptions and current data.
    You will write the logic for this function.
    """
    print("Calculating financial projections...")
    #  Add your Python code here to generate the projection tables.
    # The function should return an HTML string containing the formatted tables.
    # You can reuse or adapt the logic from your original JavaScript function.
    # For now, we'll return a placeholder message.

# --- Helper Functions for the Backend ---

def extract_text_from_pdfs_from_bytes(files_data) -> str:
    """Extracts and combines text from a list of PDF file byte streams."""
    combined_text = ""
    for file_data in files_data:
        try:
            pdf_document = fitz.open(stream=file_data, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                combined_text += page.get_text()
        except Exception as e:
            print(f"Error processing PDF file: {e}")
            continue
    return combined_text



# Assuming gemini_model is configured elsewhere

def call_gemini(prompt, text_to_analyze, send_financials=False, financial_data=None, extra_context=None,):
    """Calls the Gemini API with a specific prompt and text, optionally including financial data."""
    
    # --- Refined Prompt Construction ---
    # Determine the content and its description based on the arguments.
    if send_financials and financial_data:
        description = "transcripts and historical financial statements"
        content_to_send = f"{text_to_analyze}\n\n---\n\n{financial_data}"
    else:
        description = "transcripts"
        content_to_send = text_to_analyze

    # Append prior step's output if provided to aid chaining
    if extra_context:
        content_to_send = f"{content_to_send}\n\n---\n\nPrior context from previous step:\n{extra_context}"

    # Assemble the final prompt
    full_prompt = f"{prompt}\n\nHere are the {description} to analyze:\n\n---\n\n{content_to_send}"
    # --- End of Refined Construction ---

    try:
        # Debug append: record input payload prior to call (if enabled)
        if _ENABLE_GEMINI_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "gemini_input_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n\n=== GEMINI_CALL INPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write("-- send_financials: " + str(bool(send_financials)) + "\n")
                    _f.write("-- extra_context: " + ("yes" if bool(extra_context) else "no") + "\n")
                    _f.write("-- prompt:\n" + (prompt if isinstance(prompt, str) else _json.dumps(prompt, ensure_ascii=False)) + "\n")
                    _f.write("-- content_to_analyze (possibly markdown):\n" + (text_to_analyze if isinstance(text_to_analyze, str) else _json.dumps(text_to_analyze, ensure_ascii=False)) + "\n")
                    if send_financials and financial_data is not None:
                        _f.write("-- financial_data (truncated to 10k chars):\n")
                        _fd = financial_data if isinstance(financial_data, str) else _json.dumps(financial_data, ensure_ascii=False)
                        _f.write(str(_fd)[:10000] + ("...\n" if len(str(_fd)) > 10000 else "\n"))
                    _f.write("=== END INPUT ===\n")
            except Exception as _e0:
                print(f"[debug] Failed to log Gemini input: {_e0}")

        response = gemini_model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000
            )
        )
        # Debug append: record raw output (if enabled)
        if _ENABLE_GEMINI_DEBUG_LOGS:
            try:
                _debug_path = os.path.join(os.path.dirname(__file__), "gemini_input_debug.txt")
                with open(_debug_path, "a", encoding="utf-8") as _f:
                    _f.write("\n=== GEMINI_CALL OUTPUT @ " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                    _f.write((response.text or "") + "\n")
                    _f.write("=== END OUTPUT ===\n")
            except Exception as _e1:
                print(f"[debug] Failed to log Gemini output: {_e1}")

        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return f"Error: Could not generate content from AI. Details: {e}"


def calculate_financial_projections(
    assumptions,
    financial_data,
    qualitative_text,
    compact_financials=None,            # optional: output from compact_financial_data_for_llm (str or dict)
    march31_prices: dict[int, float] | None = None,  # optional: output from get_march31_prices_last_5_years
    ticker: str | None = None,          # optional: to fetch latest price if needed
    latest_price: float | None = None,  # optional: override latest price
    sector_median_pe: float | None = None  # optional: bounds sanity
):
    """
    Calculates future financial projections and interleaves them with qualitative analysis,
    returning a single HTML string.
    """
    print("Calculating financial projections and combining with analysis...")
    
    # Helper: basic Markdown-ish to HTML converter (bullets, ordered lists, paragraphs, bold/inline-code)
    def _mdish_to_html(block: str) -> str:
        text = (block or "").strip()
        if not text:
            return ""
        lines = text.splitlines()
        html_parts = []
        in_ul = False
        in_ol = False
        # Support -, *, +, bullet (•), en/em dash as bullets
        bullet_re = _re.compile(r"^\s*[\-\*\+\u2022\u2013\u2014]\s+(.*)$")
        num_re = _re.compile(r"^\s*(\d+)[\.)]\s+(.*)$")

        def close_lists():
            nonlocal in_ul, in_ol
            if in_ul:
                html_parts.append("</ul>")
                in_ul = False
            if in_ol:
                html_parts.append("</ol>")
                in_ol = False

        for raw in lines:
            if not raw.strip():
                # Blank line -> paragraph break
                close_lists()
                continue
            m_b = bullet_re.match(raw)
            m_n = num_re.match(raw)
            if m_b:
                if not in_ul:
                    close_lists()
                    html_parts.append("<ul>")
                    in_ul = True
                content = m_b.group(1)
                # inline formatting: **bold**, `code`
                content = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
                content = _re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
                html_parts.append(f"<li>{content}</li>")
                continue
            if m_n:
                if not in_ol:
                    close_lists()
                    html_parts.append("<ol>")
                    in_ol = True
                content = m_n.group(2)
                content = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
                content = _re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
                html_parts.append(f"<li>{content}</li>")
                continue
            # Normal paragraph line
            close_lists()
            content = raw.strip()
            content = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = _re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            html_parts.append(f"<p>{content}</p>")

        close_lists()
        return "".join(html_parts)

    # Helper: robustly parse the qualitative assumptions text into per-case sections
    def parse_qualitative_assumptions_sections(full_text: str) -> dict:
        if not full_text:
            return {}
        # Strip global code fences if model wrapped entire output in ```
        def strip_all_fences(txt: str) -> str:
            t = txt.strip()
            # Remove leading/trailing fenced blocks and any language tags
            t = _re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", t)
            t = _re.sub(r"\n```\s*$", "", t)
            # Remove any remaining lone fences
            t = t.replace("```", "")
            return t
        text = strip_all_fences(full_text.replace("\r\n", "\n").replace("\r", "\n"))

        # Preferred path: split using explicit NEWSECTION cues and map to cases
        cue_pattern = _re.compile(r"^\s*##NEWSECTION\{(base_case|bull_case|bear_case)\}##\s*$", _re.IGNORECASE | _re.MULTILINE)
        segments = []  # list of (label_or_None, content)
        last = 0
        for m in cue_pattern.finditer(text):
            if m.start() > last:
                segments.append((None, text[last:m.start()]))
            segments.append((m.group(1).lower(), ""))  # marker; content will follow until next cue
            last = m.end()
        if last < len(text):
            segments.append((None, text[last:]))

        if segments:
            acc = {"base_case": [], "bull_case": [], "bear_case": []}
            current = None
            for label, content in segments:
                if label in acc:
                    current = label
                    continue
                if current:
                    acc[current].append(content)
            by_case = {"base_case": None, "bull_case": None, "bear_case": None}
            for k, chunks in acc.items():
                block = strip_all_fences("".join(chunks).strip())
                if block:
                    html_block = _mdish_to_html(block)
                    by_case[k] = f"<div class=\"qualitative-analysis\"><h3 class=\"text-gray-800 font-semibold mb-2\">Assumptions & Rationale</h3>{html_block}</div>"
            if any(by_case.values()):
                return by_case

        lines = text.split("\n")
        # Identify all markdown headings
        heading_idx = []  # (idx, line)
        heading_re = _re.compile(r"^\s{0,3}#{2,6}\s+(.+)$")
        for i, ln in enumerate(lines):
            if heading_re.match(ln):
                heading_idx.append((i, ln.strip()))
        # Map to cases
        case_map = {"base_case": None, "bull_case": None, "bear_case": None}
        # Flexible matcher: line contains '<case> ... case' in any form
        def is_case_heading(line: str, case_word: str) -> bool:
            lw = line.lower()
            # allow variations like "Base Case", "Base-Case", "Assumptions – Base Case"
            lw = lw.replace("–", "-")  # normalize en-dash
            return (case_word in lw) and ("case" in lw)

        # Find start index for each case from markdown headings first
        case_starts = {}
        for idx, ln in heading_idx:
            if is_case_heading(ln, "base") and case_starts.get("base_case") is None:
                case_starts["base_case"] = idx
            if is_case_heading(ln, "bull") and case_starts.get("bull_case") is None:
                case_starts["bull_case"] = idx
            if is_case_heading(ln, "bear") and case_starts.get("bear_case") is None:
                case_starts["bear_case"] = idx

        # If not found via markdown headings, attempt plain-line headings (no #)
        plain_positions = []
        if not case_starts.get("base_case") or not case_starts.get("bull_case") or not case_starts.get("bear_case"):
            plain_heading_re = _re.compile(r"^(?:\s*(?:#{1,6}\s*)?)?(?:\*\*|__|`)?\s*(?:the\s+)?(base|bull|bear)[\s\-–]?case(?:\s+assumptions?)?\s*:?\s*(?:\*\*|__|`)?\s*$", _re.IGNORECASE)
            for i, ln in enumerate(lines):
                s = (ln or "").strip()
                m = plain_heading_re.match(s)
                if m:
                    word = m.group(1).lower()
                    key = f"{word}_case"
                    if not case_starts.get(key):
                        case_starts[key] = i
                    plain_positions.append(i)

        # Helper to find next heading after a given index
        # Consider both markdown and plain headings for section boundaries
        heading_positions = sorted(set([i for i, _ in heading_idx] + plain_positions))
        def next_heading_after(i_start: int) -> int:
            for pos in heading_positions:
                if pos > i_start:
                    return pos
            return len(lines)

        def _strip_fences(block: str) -> str:
            b = block.strip()
            # Remove enclosing triple backtick fences if present
            fence_re = _re.compile(r"^```[a-zA-Z0-9]*\s*\n([\s\S]*?)\n```\s*$")
            m = fence_re.match(b)
            if m:
                return m.group(1).strip()
            # Remove stray opening/closing fences
            b = _re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", b)
            b = _re.sub(r"\n```\s*$", "", b)
            return b

        for key, start in case_starts.items():
            if start is None:
                continue
            end = next_heading_after(start)
            # Exclude the heading line itself
            block = "\n".join(lines[start + 1:end]).strip()
            block = _strip_fences(block)
            # Some models include subheadings; keep them as paragraphs
            html_block = _mdish_to_html(block)
            case_map[key] = f"<div class=\"qualitative-analysis\"><h3 class=\"text-gray-800 font-semibold mb-2\">Assumptions & Rationale</h3>{html_block}</div>"
        # Fallback: if nothing detected, keep old regex approach per-case headings with '### <Case> Assumptions'
        if all(v is None for v in case_map.values()):
            for ck, human in [("base_case", "Base Case"), ("bull_case", "Bull Case"), ("bear_case", "Bear Case")]:
                pattern = _re.compile(rf"^\s*#{2,6}\s*{_re.escape(human)}[^\n]*?(?:Assumptions)?\s*$([\s\S]*?)(?=^\s*#{1,6}\s|\Z)", _re.IGNORECASE | _re.MULTILINE)
                m = pattern.search(text)
                if m:
                    html_block = _mdish_to_html(_strip_fences(m.group(1).strip()))
                    case_map[ck] = f"<div class=\"qualitative-analysis\"><h3 class=\"text-gray-800 font-semibold mb-2\">Assumptions & Rationale</h3>{html_block}</div>"
        return case_map

    # (All your other helper functions like calculate_revenue_growth, etc. remain here unchanged)
    # ...
    def calculate_revenue_growth(revenue, growth_rate, years):
        growth_rate_decimal = growth_rate / 100
        projections = []
        for year in range(1, years + 1):
            projected_revenue = revenue * ((1 + growth_rate_decimal) ** year)
            projections.append(round(projected_revenue))
        return projections

    def calculate_ebitda(revenues, ebitda_margin):
        ebitda_values = []
        for revenue in revenues:
            ebitda = revenue * (ebitda_margin / 100)
            ebitda_values.append(round(ebitda))
        return ebitda_values

    def calculate_pbt(ebitda_values, interest, depreciation):
        pbt_values = []
        for ebitda in ebitda_values:
            pbt = ebitda - interest - depreciation
            pbt_values.append(round(pbt))
        return pbt_values

    def calculate_pat(pbt_values, tax_rate):
        pat_values = []
        for pbt in pbt_values:
            pat = pbt * (1 - tax_rate / 100)
            pat_values.append(round(pat))
        return pat_values

    def calculate_number_of_shares(pat_year0, eps_year0):
        if eps_year0 == 0:
            return 0
        return round(pat_year0 / eps_year0)

    def calculate_eps(pat_values, number_of_shares):
        eps_values = []
        for pat in pat_values:
            if number_of_shares == 0:
                eps = 0
            else:
                eps = pat / number_of_shares
            eps_values.append(round(eps, 2))
        return eps_values
    # --- Main Logic ---

    year0_ebitda = financial_data['Operating Profit'] + financial_data['Depreciation']
    num_shares = calculate_number_of_shares(financial_data['Net Profit+'], financial_data['EPS in Rs'])

    # Resolve latest price for market cap sizing and Year 0 display
    resolved_latest_price = None
    if latest_price is not None and np.isfinite(latest_price):
        resolved_latest_price = float(latest_price)
    elif march31_prices:
        # pick the most recent available fiscal year price
        try:
            latest_year = max(k for k, v in march31_prices.items() if v is not None)
            resolved_latest_price = float(march31_prices[latest_year]) if latest_year else None
        except Exception:
            resolved_latest_price = None
    elif ticker:
        # try today's price via helper; ignore if it fails
        try:
            resolved_latest_price = float(get_price_on_date(ticker, date.today().isoformat()))
        except Exception:
            resolved_latest_price = None

    # Determine company size for PE sensitivity
    try:
        comp_size = classify_market_cap_category(resolved_latest_price, num_shares) if resolved_latest_price else "midcap"
    except Exception:
        comp_size = "midcap"

    # Initialize HTML string with CSS styling (added style for the new analysis div)
    html_output = """
    <style>
        body { font-family: sans-serif; }
        .container { display: flex; flex-direction: column; gap: 30px; } /* Increased gap */
        .projection-table { border-collapse: collapse; width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px; } /* Added margin */
        .projection-table th, .projection-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .projection-table th { background-color: #f2f2f2; font-weight: bold; }
        .projection-table td:first-child { font-weight: bold; }
        .projection-table tr:nth-child(even) { background-color: #f9f9f9; }
        h2 { color: #333; border-bottom: 2px solid #f2f2f2; padding-bottom: 5px; }
        .qualitative-analysis { background-color: #fdfdfd; border-left: 4px solid #007bff; padding: 15px; margin-top: 10px; font-size: 0.9em; line-height: 1.6; } /* New style */
    </style>
    <div class="container">
    """

    # Parse qualitative assumptions text once into per-case sections
    sections_by_case = parse_qualitative_assumptions_sections(qualitative_text)

    # Loop through each scenario: base, bull, and bear
    for case_name, case_assumptions in assumptions.items():
        title = case_name.replace('_', ' ').title()
        # Avoid duplicating the word 'Case' in the heading
        display_title = title if 'case' in title.lower() else f"{title} Case"

        growth_rate = case_assumptions['revenue_growth_cagr']
        ebitda_margin = case_assumptions['ebitda_margin']
        tax_rate = case_assumptions['tax_rate']

        # Extend projections to 3 years to match your prompt's request
        revenues_proj = calculate_revenue_growth(financial_data['Sales+'], growth_rate, 3)
        ebitdas_proj = calculate_ebitda(revenues_proj, ebitda_margin)
        pbts_proj = calculate_pbt(ebitdas_proj, financial_data['Interest'], financial_data['Depreciation'])
        pats_proj = calculate_pat(pbts_proj, tax_rate)
        eps_proj = calculate_eps(pats_proj, num_shares)

        # Compute EPS CAGR over the 3 projected years, starting from Year 0 actual EPS
        try:
            eps0 = float(financial_data['EPS in Rs'])
            eps_last = float(eps_proj[-1]) if eps_proj else eps0
            # Avoid zero/negative EPS issues
            if eps0 > 0 and eps_last > 0:
                eps_cagr = ((eps_last / eps0) ** (1 / 3) - 1) * 100.0
            else:
                eps_cagr = 0.0
        except Exception:
            eps_cagr = 0.0

        # Build past PE list if we have compact financials and prices, else fallback to neutral 20x
        past_pe_list = None
        if compact_financials is not None and march31_prices is not None:
            try:
                past_pe_list = build_past_pe_list_from_compact(compact_financials, march31_prices)
            except Exception:
                past_pe_list = None
        if not past_pe_list:
            # Provide a benign default base PE of 20 to keep projections reasonable
            this_year = date.today().year
            past_pe_list = [{"year": this_year - 3, "pe": 20.0, "eps": None},
                            {"year": this_year - 2, "pe": 20.0, "eps": None},
                            {"year": this_year - 1, "pe": 20.0, "eps": None}]

        try:
            projected_pe = estimate_future_pe(past_pe_list, eps_cagr, sector_median_pe=sector_median_pe, company_size=comp_size)
            if not np.isfinite(projected_pe) or projected_pe <= 0:
                projected_pe = 20.0
        except Exception:
            projected_pe = 20.0

        # Project stock prices per projected year as EPS * projected PE
        projected_prices = [round(float(e) * float(projected_pe), 2) if e is not None else None for e in eps_proj]

        # Year 0 price to display (prefer March 31 latest FY price, else resolved latest price)
        year0_price = None
        try:
            if march31_prices:
                yr0 = max(march31_prices.keys()) if march31_prices else None
                v = march31_prices.get(yr0) if yr0 else None
                year0_price = float(v) if v is not None else None
        except Exception:
            year0_price = None
        if year0_price is None:
            year0_price = resolved_latest_price

        data_rows = {
            "Revenue": [financial_data['Sales+']] + revenues_proj,
            "EBITDA": [year0_ebitda] + ebitdas_proj,
            "PBT": [financial_data['Profit before tax']] + pbts_proj,
            "PAT": [financial_data['Net Profit+']] + pats_proj,
            "EPS": [financial_data['EPS in Rs']] + eps_proj,
            "Stock Price": [year0_price] + projected_prices
        }

        # Build the HTML for the current case
        html_output += f"<div><h2>{display_title}</h2>"
        html_output += '<table class="projection-table">'
        html_output += "<tr><th>Metric</th><th>Year 0 (Actual)</th><th>Year 1 (Proj)</th><th>Year 2 (Proj)</th><th>Year 3 (Proj)</th></tr>"
        
        for metric, values in data_rows.items():
            # Nicely format ints and floats; allow None as '-' for readability
            def _fmt(v):
                if v is None:
                    return "-"
                try:
                    if isinstance(v, (int, np.integer)):
                        return f"{int(v):,}"
                    else:
                        return f"{float(v):,.2f}"
                except Exception:
                    return str(v)
            formatted_values = [_fmt(v) for v in values]
            html_output += f"<tr><td>{metric}</td><td>{formatted_values[0]}</td><td>{formatted_values[1]}</td><td>{formatted_values[2]}</td><td>{formatted_values[3]}</td></tr>"
        
        html_output += "</table>"

        # Append the qualitative analysis for the current case (if available)
        qualitative_section_html = sections_by_case.get(case_name)
        if not qualitative_section_html:
            qualitative_section_html = "<div class='qualitative-analysis'><p style='color:#a00;'>Could not find the qualitative assumptions for this case.</p></div>"
        html_output += qualitative_section_html
        
        html_output += "</div>" # Close the div for this case

    html_output += "</div>" # Close container div
    return html_output

    # --- Main Logic ---

    # Calculate base year (Year 0) EBITDA
    # EBITDA = Operating Profit (EBIT) + Depreciation
    year0_ebitda = financial_data['Operating Profit'] + financial_data['Depreciation']
    
    # Calculate the number of shares (assumed to be constant)
    num_shares = calculate_number_of_shares(financial_data['Net Profit+'], financial_data['EPS in Rs'])

    # Initialize HTML string with CSS styling
    html_output = """
    <style>
        body { font-family: sans-serif; }
        .container { display: flex; flex-direction: column; gap: 20px; }
        .projection-table { border-collapse: collapse; width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .projection-table th, .projection-table td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        .projection-table th { background-color: #f2f2f2; font-weight: bold; }
        .projection-table td:first-child { font-weight: bold; }
        .projection-table tr:nth-child(even) { background-color: #f9f9f9; }
        h2 { color: #333; border-bottom: 2px solid #f2f2f2; padding-bottom: 5px; }
    </style>
    <div class="container">
    """

    # Loop through each scenario: base, bull, and bear
    for case_name, case_assumptions in assumptions.items():
        title = case_name.replace('_', ' ').title()

        # Get assumptions for the current case
        growth_rate = case_assumptions['revenue_growth_cagr']
        ebitda_margin = case_assumptions['ebitda_margin']
        tax_rate = case_assumptions['tax_rate']

        # Perform calculations for Year 1 and Year 2
        revenues_proj = calculate_revenue_growth(financial_data['Sales+'], growth_rate, 2)
        ebitdas_proj = calculate_ebitda(revenues_proj, ebitda_margin)
        pbts_proj = calculate_pbt(ebitdas_proj, financial_data['Interest'], financial_data['Depreciation'])
        pats_proj = calculate_pat(pbts_proj, tax_rate)
        eps_proj = calculate_eps(pats_proj, num_shares)

        # Combine Year 0 data with projections for the table
        data_rows = {
            "Revenue": [financial_data['Sales+']] + revenues_proj,
            "EBITDA": [year0_ebitda] + ebitdas_proj,
            "PBT": [financial_data['Profit before tax']] + pbts_proj,
            "PAT": [financial_data['Net Profit+']] + pats_proj,
            "EPS": [financial_data['EPS in Rs']] + eps_proj
        }

        # Build the HTML table for the current case
        html_output += f"<div><h2>{title} Projections</h2>"
        html_output += '<table class="projection-table">'
        html_output += "<tr><th>Metric</th><th>Year 0 (Actual)</th><th>Year 1 (Projected)</th><th>Year 2 (Projected)</th></tr>"
        
        for metric, values in data_rows.items():
            # Format numbers with commas for better readability
            formatted_values = [f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" for v in values]
            html_output += f"<tr><td>{metric}</td><td>{formatted_values[0]}</td><td>{formatted_values[1]}</td><td>{formatted_values[2]}</td></tr>"
        
        html_output += "</table></div>"

    html_output += "</div>" # Close container div
    return html_output

def estimate_future_pe(
    past_pe_list,              # Input history; supports:
                               # - list[float] of past P/E ratios (old behavior)
                               # - list[dict] with keys like 'pe' and 'eps' (recommended)
                               # - list[tuple|list] as (year, pe, eps) or (year, eps) (best-effort)
    future_eps_cagr,           # projected EPS CAGR (in %)
    sector_median_pe=None,     # optional sanity check
    company_size="midcap"      # "smallcap", "midcap", or "largecap"
):
    """
    Estimate future P/E ratio combining:
    - Base P/E from historical median
    - Adjustment by delta between projected EPS CAGR and trailing 3Y EPS CAGR

    Input flexibility (past_pe_list):
    - If a plain list of numbers is provided, it's treated as P/E values and past EPS CAGR
      can't be computed (falls back gracefully to 0%).
    - If dicts are provided, tries keys: 'eps', 'EPS', 'eps_in_rs' for EPS and 'pe', 'PE' for P/E.
    - If tuples/lists are provided, attempts to read (year, pe, eps) or (year, eps) ordering.

    Returns: float projected P/E, rounded to 2 decimals.
    """

    def _extract_series(past_list):
        """Extract ordered PE and EPS series from heterogeneous input.
        Returns (pe_values, eps_values, years) where each is a list ordered oldest->newest when possible.
        """
        pe_vals = []
        eps_vals = []
        years = []

        # Helper to append keeping alignment
        def _append(year, pe, eps):
            years.append(year)
            pe_vals.append(pe)
            eps_vals.append(eps)

        # Try to detect structure
        for item in (past_list or []):
            if isinstance(item, dict):
                y = item.get('year') or item.get('fy') or item.get('FY')
                pe = item.get('pe') if 'pe' in item else item.get('PE')
                eps = (
                    item.get('eps') if 'eps' in item else
                    item.get('EPS') if 'EPS' in item else
                    item.get('eps_in_rs') if 'eps_in_rs' in item else
                    item.get('EPS in Rs')
                )
                # Only record when at least one of pe/eps exists
                if pe is not None or eps is not None:
                    _append(y, pe, eps)
            elif isinstance(item, (tuple, list)):
                # Heuristics: (year, pe, eps) or (year, eps) or (pe, eps) or (year, pe)
                y = None
                pe = None
                eps = None
                if len(item) >= 3:
                    y = item[0]
                    pe = item[1]
                    eps = item[2]
                elif len(item) == 2:
                    # If first looks like a year, treat as (year, eps) and leave pe unknown
                    first = item[0]
                    second = item[1]
                    if isinstance(first, (int, float)) and 1900 <= int(first) <= 2100:
                        y = int(first)
                        # Second could be eps; we can't infer pe reliably
                        eps = second
                    else:
                        # Treat as (pe, eps)
                        pe = first
                        eps = second
                elif len(item) == 1:
                    # Single number list -> likely P/E
                    pe = item[0]
                _append(y, pe, eps)
            else:
                # Plain number: assume P/E
                try:
                    val = float(item)
                    _append(None, val, None)
                except Exception:
                    # Unknown shape; skip
                    pass

        # If years available, sort by year ascending to get oldest->newest
        if any(y is not None for y in years):
            order = sorted(range(len(years)), key=lambda i: (float('inf') if years[i] is None else years[i]))
            pe_vals = [pe_vals[i] for i in order]
            eps_vals = [eps_vals[i] for i in order]
            years = [years[i] for i in order]

        return pe_vals, eps_vals, years

    def _compute_trailing_cagr(eps_series, trailing_years=3):
        """Compute CAGR over the last `trailing_years` between the most recent and the value N years back.
        Falls back to geometric mean of year-over-year growth if exact span not available.
        Returns CAGR in percent.
        """
        # Filter to finite, positive EPS (CAGR undefined for <=0)
        seq = [float(x) for x in eps_series if x is not None and np.isfinite(x)]
        if len(seq) < 2:
            return 0.0
        # Use last point and N-back if possible
        last = seq[-1]
        idx_back = max(0, len(seq) - 1 - trailing_years)
        span = (len(seq) - 1) - idx_back
        first = seq[idx_back]
        if first > 0 and last > 0 and span > 0:
            try:
                cagr = (last / first) ** (1.0 / span) - 1.0
                return float(cagr * 100.0)
            except Exception:
                pass
        # Fallback: log-return average of available consecutive positive pairs
        logs = []
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            if a is not None and b is not None and a > 0 and b > 0:
                try:
                    logs.append(np.log(b / a))
                except Exception:
                    continue
        if not logs:
            return 0.0
        geo = np.exp(np.mean(logs)) - 1.0
        return float(geo * 100.0)

    # ---- Extract series from input
    pe_series, eps_series, _years = _extract_series(past_pe_list)

    # 1) Base P/E (median of available PE numbers); ignore None
    pe_clean = [float(v) for v in pe_series if v is not None and np.isfinite(v)]
    base_pe = np.median(pe_clean) if pe_clean else (np.median(past_pe_list) if past_pe_list else 0.0)

    # 2) Compute trailing EPS CAGR over last 3 years when EPS is available
    past_eps_cagr = _compute_trailing_cagr(eps_series, trailing_years=3) if any(v is not None for v in eps_series) else 0.0

    # 3) Sensitivity constant 'k' based on company size
    size_map = {"smallcap": 1.2, "midcap": 1.0, "largecap": 0.6}
    k = size_map.get(company_size.lower(), 1.0)

    # 4) Calculate growth delta (convert % to decimal)
    growth_delta = (float(future_eps_cagr) - float(past_eps_cagr)) / 100.0

    # 5) Projected P/E with guardrails
    projected_pe = base_pe * (1.0 + k * growth_delta)
    projected_pe = np.clip(projected_pe, base_pe * 0.7, base_pe * 1.3) if base_pe > 0 else projected_pe

    # 6) Optional sanity check vs sector median
    if sector_median_pe is not None and np.isfinite(sector_median_pe):
        lower_bound, upper_bound = sector_median_pe * 0.7, sector_median_pe * 1.3
        projected_pe = np.clip(projected_pe, lower_bound, upper_bound)

    return round(float(projected_pe), 2)

def get_price_on_date(ticker: str, target_date: str):
    """
    Fetches the stock price for a given ticker on a specific date.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'RELIANCE.NS').
    - target_date (str): The target date in 'YYYY-MM-DD' format.

    Returns:
    - float: The stock's closing price on the target date.
    """
    # Fetch historical data for the past 1 year
    data = yf.download(ticker, period="1y")
    
    # Convert the index to datetime
    data.index = pd.to_datetime(data.index)
    
    # Filter data for the target date
    price_on_date = data.loc[data.index.date == pd.to_datetime(target_date).date(), 'Close']
    
    if not price_on_date.empty:
        return price_on_date.iloc[0]
    else:
        raise ValueError(f"No data available for {ticker} on {target_date}")


def get_march31_prices_last_5_years(ticker: str, years_count: int = 5) -> dict[int, float]:
    """
    Return a dictionary of stock prices on March 31 for the latest N fiscal years.

    Tries to use existing get_price_on_date for dates within the last year, and
    falls back to a direct yfinance window fetch for older years or market holidays.

    Keys: fiscal year as int (e.g., 2025), Values: closing price (float).
    """
    results: dict[int, float] = {}

    today = date.today()
    # Latest completed March 31 relative to today
    latest_fy = today.year if date(today.year, 3, 31) <= today else today.year - 1
    target_years = [latest_fy - i for i in range(years_count)]

    def _price_via_yf_window(tkr: str, d: date, back_days: int = 10) -> float | None:
        """Fetch price using a small window around the target date; pick the last close on/before d."""
        start_dt = d - timedelta(days=back_days)
        end_dt = d + timedelta(days=1)  # include target day
        try:
            df = yf.download(tkr, start=start_dt.isoformat(), end=end_dt.isoformat(), progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            # Filter rows on or before target date, pick last available
            df_on_or_before = df[df.index.date <= d]
            if df_on_or_before.empty:
                return None
            return float(df_on_or_before['Close'].iloc[-1])
        except Exception:
            return None

    for yr in target_years:
        target = date(yr, 3, 31)
        got_price = None
        # 1) Try the provided helper for dates within the last ~1 year (its limitation)
        try:
            got_price = float(get_price_on_date(ticker, target.isoformat()))
        except Exception:
            got_price = None

        # 2) If still None, try walking back a few days via our helper (market holidays)
        if got_price is None:
            # Use a compact window to handle holidays/weekends and older years
            got_price = _price_via_yf_window(ticker, target, back_days=14)

        # 3) As a final fallback, expand the window
        if got_price is None:
            got_price = _price_via_yf_window(ticker, target, back_days=35)

        if got_price is not None:
            results[yr] = round(got_price, 4)
        else:
            # Leave missing entries out or set to None; choose explicit None for clarity
            results[yr] = None

    return results


def classify_market_cap_category(latest_price: float, number_of_shares: float | int) -> str:
    """
    Classify the company as 'smallcap', 'midcap', or 'largecap' based on market cap (INR crores).

    Calculation:
    - market_cap_rs = latest_price * number_of_shares
    - market_cap_cr = market_cap_rs / 1e7   # 1 crore = 10,000,000

    Bands:
    - < 5000 cr -> 'smallcap'
    - 5000 to 20000 cr (inclusive) -> 'midcap'
    - > 20000 cr -> 'largecap'
    """
    try:
        price = float(latest_price)
        shares = float(number_of_shares)
    except Exception as e:
        raise ValueError(f"Invalid inputs. latest_price and number_of_shares must be numeric. Details: {e}")

    if not np.isfinite(price) or not np.isfinite(shares) or price <= 0 or shares <= 0:
        raise ValueError("latest_price and number_of_shares must be positive finite numbers")

    market_cap_rs = price * shares
    market_cap_cr = market_cap_rs / 1e7  # convert to crore rupees

    if market_cap_cr < 5000:
        return "smallcap"
    elif market_cap_cr <= 20000:
        return "midcap"
    else:
        return "largecap"


def compute_yearly_pe_from_compact(financials_compact, march31_prices: dict[int, float]) -> dict[int, float]:
    """
    Build a dictionary of P/E ratios by fiscal year using:
    - compact_financial_data_for_llm output (string JSON or dict)
    - get_march31_prices_last_5_years output (dict[int, float])

    Returns: dict[year:int -> pe:float], only for years where both price and positive EPS exist.
    """
    # Normalize compact input to dict
    if isinstance(financials_compact, str):
        try:
            fin = _json.loads(financials_compact)
        except Exception as e:
            raise ValueError(f"Invalid compact financials JSON: {e}")
    elif isinstance(financials_compact, dict):
        fin = financials_compact
    else:
        raise ValueError("financials_compact must be a JSON string or dict")

    pl = (fin or {}).get("profit_and_loss") or {}
    years = list(pl.get("years") or [])
    data = pl.get("data") or {}

    if not years or not data:
        return {}

    # Identify EPS row key with preference order
    eps_key_candidates = [
        "EPS in Rs",
        "EPS",
    ]
    # Fallback: any key containing 'eps' (case-insensitive)
    eps_key = None
    for k in eps_key_candidates:
        if k in data:
            eps_key = k
            break
    if eps_key is None:
        for k in data.keys():
            if isinstance(k, str) and "eps" in k.lower():
                eps_key = k
                break
    if eps_key is None:
        return {}

    eps_series = list(data.get(eps_key) or [])
    # Align EPS to years; skip the 'TTM' slot
    pe_by_year: dict[int, float] = {}
    for idx, yr in enumerate(years):
        # Skip TTM or non-integer year labels
        if isinstance(yr, str):
            continue
        try:
            yr_int = int(yr)
        except Exception:
            continue
        # Guard index range
        if idx >= len(eps_series):
            continue
        eps_val = eps_series[idx]
        if eps_val is None:
            continue
        try:
            eps_f = float(eps_val)
        except Exception:
            continue
        if not np.isfinite(eps_f) or eps_f <= 0:
            continue
        price = march31_prices.get(yr_int)
        if price is None:
            continue
        try:
            p = float(price)
        except Exception:
            continue
        if not np.isfinite(p) or p <= 0:
            continue
        pe = p / eps_f
        pe_by_year[yr_int] = round(float(pe), 2)

    return pe_by_year


def build_past_pe_list_from_compact(financials_compact, march31_prices: dict[int, float]) -> list[dict]:
    """
    Convenience helper that returns a list of dicts with year, pe, eps suitable for
    estimate_future_pe (enables trailing EPS CAGR computation):
    [ {"year": 2021, "pe": 22.1, "eps": 12.3}, ... ]
    """
    # Reuse compute_yearly_pe_from_compact for P/E values
    pe_by_year = compute_yearly_pe_from_compact(financials_compact, march31_prices)
    # Also extract EPS by year from compact
    if isinstance(financials_compact, str):
        fin = _json.loads(financials_compact)
    else:
        fin = financials_compact
    pl = (fin or {}).get("profit_and_loss") or {}
    years = list(pl.get("years") or [])
    data = pl.get("data") or {}
    # Find EPS key again
    eps_key = None
    for k in ("EPS in Rs", "EPS"):
        if k in data:
            eps_key = k
            break
    if eps_key is None:
        for k in data.keys():
            if isinstance(k, str) and "eps" in k.lower():
                eps_key = k
                break
    eps_series = list(data.get(eps_key) or [])

    out = []
    for idx, yr in enumerate(years):
        if isinstance(yr, str):
            continue
        try:
            yr_int = int(yr)
        except Exception:
            continue
        if yr_int not in pe_by_year:
            continue
        if idx >= len(eps_series):
            continue
        eps_val = eps_series[idx]
        try:
            eps_f = float(eps_val) if eps_val is not None else None
        except Exception:
            eps_f = None
        out.append({"year": yr_int, "pe": pe_by_year[yr_int], "eps": eps_f})

    # Sort oldest -> newest
    out.sort(key=lambda d: d.get("year", 0))
    return out


