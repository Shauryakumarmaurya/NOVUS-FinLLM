# tasks.py
import json
import time
from rq import get_current_job
from logic import (
    extract_text_from_pdfs_from_bytes,
    get_yearly_financial_statements_html,
    extract_financial_data_from_html,
    call_gemini,
    calculate_financial_projections,
    PROMPTS,
    compact_financial_data_for_llm,
    get_march31_prices_last_5_years
)

PROGRESS_STAGES = [
    "extract_pdfs",
    "fetch_financials",
    "business_model",
    "quarterly_updates",
    "management_commentary",
    "risks",
    "prompt_set",
    "assumptions",
    "projections",
    "assemble"
]

def _update_progress(stage: str, extra: dict | None = None):
    job = get_current_job()
    if not job:
        return
    job.meta['stage'] = stage
    if 'stages' not in job.meta:
        job.meta['stages'] = PROGRESS_STAGES
    if extra:
        job.meta.update(extra)
    job.save_meta()

def generate_financial_report(ticker, files_data):
    """Background task to generate financial report (RQ worker)."""
    try:
        _update_progress('extract_pdfs')
        combined_text = extract_text_from_pdfs_from_bytes(files_data)
        if not combined_text:
            raise ValueError("Could not extract text from the provided PDF files.")

        _update_progress('fetch_financials')
        profit_and_loss_html, balance_sheet_html = get_yearly_financial_statements_html(ticker)
        if not profit_and_loss_html or not balance_sheet_html:
            raise ValueError(f"Could not fetch financial statements for ticker: {ticker}")
        financial_data = extract_financial_data_from_html(profit_and_loss_html)# profit and loss
        financial_data_bs = extract_financial_data_from_html(balance_sheet_html)# balance sheet
        financial_data_json_for_llm= compact_financial_data_for_llm(financial_data, financial_data_bs, years_count=4, return_json=True)
        _update_progress('business_model')
        # Include a compact snapshot of the scraped financials to aid the first LLM step
        business_model_md = call_gemini(
            PROMPTS["business_model"],
            combined_text,
            
        )
      

        _update_progress('quarterly_updates')
        quarterly_updates_md = call_gemini(
            PROMPTS["key_quarterly_updates"],
            combined_text,
        )
      

        _update_progress('management_commentary')
        mgmt_commentary_md = call_gemini(
            PROMPTS["management_commentary"],
            combined_text,
        )
        

        _update_progress('risks')
        risks_md = call_gemini(
            PROMPTS["risks_uncertainties"],
            combined_text,
        )
       

     

        _update_progress('assumptions')
        Qualitative_assumptions_md = call_gemini(
            PROMPTS["detailed_financial_assumptions_text"],
            combined_text,
            send_financials=True,
            financial_data=financial_data_json_for_llm
        )
        
        time.sleep(2)
        financial_assumptions_json_str = call_gemini(
            PROMPTS["extract_assumptions_json"],
            Qualitative_assumptions_md
        )

        projections_html = ""
        assumptions = {}
        try:
            raw = financial_assumptions_json_str
            if '```json' in raw:
                raw = raw.split('```json', 1)[1].rsplit('```', 1)[0]
            raw = raw.strip()
            assumptions = json.loads(raw)
        except Exception as e:
            projections_html = "<p style='color:red'>Failed to parse assumptions JSON.</p>"
            _update_progress('assumptions', {"assumptions_parse_error": str(e)})

        if assumptions:
            _update_progress('projections')
            years = [key for key in financial_data.keys() if 'ttm' not in key.lower()]
            latest_year = max(years)
            # Fetch March 31 prices for last 5 fiscal years to support PE-based price projections
            try:
                march31_prices = get_march31_prices_last_5_years(ticker, years_count=5)
            except Exception:
                march31_prices = None

            projections_html = calculate_financial_projections(
                assumptions,
                financial_data[latest_year],
                Qualitative_assumptions_md,
                compact_financials=financial_data_json_for_llm,
                march31_prices=march31_prices,
                ticker=ticker,
                latest_price=None,
                sector_median_pe=None,
            )
        

        _update_progress('prompt_set')
        prompt_set_md = call_gemini(
            PROMPTS["prompt_set"],
            combined_text,
            extra_context=Qualitative_assumptions_md
        )
        _update_progress('assemble')
        report_data = {
            "businessModel": business_model_md,
            "keyQuarterlyUpdates": quarterly_updates_md,
            "managementCommentary": mgmt_commentary_md,
            "financialProjections": projections_html,
            "risksUncertainties": risks_md,
            "promptSet": prompt_set_md,
            "status": "completed"
        }
        return report_data
    except Exception as e:
        _update_progress('failed', {"error": str(e)})
        # Raising lets RQ mark the job as failed (better visibility)
        raise