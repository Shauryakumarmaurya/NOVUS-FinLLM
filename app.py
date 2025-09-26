# app.py
import os
import google.generativeai as genai
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time 

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
app = Flask(__name__, static_folder='static')
gemini_api_key = os.getenv("GEMINI_API_KEY") # Gemini API key from environment variable
fmp_api_key = os.getenv("FMP_API_KEY") # Financial Modeling Prep API key (if needed)

# Configure the Gemini API client
genai.configure(api_key=gemini_api_key) 
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite') # Using a modern, efficient model

# A dictionary to hold all prompts for the AI model.
PROMPTS = {
    "financial_assumptions":  "You are an equity analyst. Based *only* on the qualitative management commentary, guidance, and risks mentioned in the provided text, generate financial assumptions for the next three fiscal years. Do not perform any calculations. Provide the output as a clean JSON object ONLY, with no other text or explanations. The structure must be exactly: {\"base_case\": {\"revenue_growth_cagr\": 15, \"ebitda_margin\": 28.5, \"tax_rate\": 25}, \"bull_case\": {\"revenue_growth_cagr\": 20, \"ebitda_margin\": 29.5, \"tax_rate\": 25}, \"bear_case\": {\"revenue_growth_cagr\": 10, \"ebitda_margin\": 27.0, \"tax_rate\": 25}}. Use reasonable, text-supported estimates for growth, margin, and tax rates.",
    "business_model": "You are an equity research analyst. Summarize the company's business model in simple, investor-friendly terms based on the provided text. Include these exact markdown headings: \"📌 Core Products & Services\", \"🎯 Target Markets / Customers\", \"💸 Revenue Model & Geography\", and \"📈 Scale & Competitive Positioning\".",
    "key_quarterly_updates": "Extract the 5-7 most important operational or financial updates from this concall text. Focus on: Growth drivers, Orders/capacity/margins, Strategy changes, and direct Quotes or signals from management. Present as a bulleted list in Markdown.",
    "management_commentary": "Summarize management's guidance and tone for the next 1-2 quarters from the provided text. Format your answer in Markdown under these exact headings: \"🔹 Forward-Looking Statements\", \"🔹 Management Tone & Confidence\" (classify tone as Optimistic/Cautious/Neutral and support with quotes), and \"🔹 Capex / Risk / Guidance Highlights\".",
    "risks_uncertainties": "List the key risks or uncertainties based on the concall text. Categorize them under these exact Markdown headings if possible: \"Execution Risks\", \"Demand-side or Macro Risks\", and \"Regulatory / External Risks\".",
    "prompt_set": "Based on the provided concall text, generate 3-5 company-specific, smart, and non-generic prompts an investor could ask an LLM to explore further."
}

# --- ⚙️ YOUR ACTION REQUIRED: ADD YOUR FINANCIAL LOGIC HERE ⚙️ ---


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

def extract_text_from_pdfs(pdf_files):
    """Extracts text from a list of uploaded PDF files."""
    combined_text = ""
    for pdf_file in pdf_files:
        try:
            # Open the PDF file from the in-memory file stream
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                combined_text += page.get_text() + "\n\n"
            pdf_document.close()
        except Exception as e:
            print(f"Error processing file {pdf_file.filename}: {e}")
            # Optionally, you can decide how to handle a corrupt file
            # For now, we'll just skip it
    return combined_text

def call_gemini(prompt, text_to_analyze):
    """Calls the Gemini API with a specific prompt and text."""
    full_prompt = f"{prompt}\n\nHere is the transcripts to analyze:\n\n---\n\n{text_to_analyze}"
    try:
        response = gemini_model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=1000
                    )
                    )

        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return f"Error: Could not generate content from AI. Details: {e}"
def calculate_financial_projections(assumptions, financial_data):
    """
    Calculates future financial projections for three cases (Base, Bull, Bear)
    based on assumptions and current financial data, returning an HTML string.
    """
    print("Calculating financial projections...")

    # Helper sub-functions (as provided in the prompt)
    def calculate_revenue_growth(revenue, growth_rate, years):
        growth_rate_decimal = growth_rate / 100
        projections = []
        for year in range(1, years + 1):
            projected_revenue = revenue * ((1 + growth_rate_decimal) ** year)
            projections.append(round(projected_revenue)) # Round to nearest integer
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
            # Assuming interest and depreciation remain constant as per base data
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

# --- Main API Endpoint ---

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'files' not in request.files or 'ticker' not in request.form:
        return jsonify({"error": "Missing files or ticker symbol"}), 400

    ticker = request.form['ticker']
    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    # 1. Extract text from PDFs
    combined_text = extract_text_from_pdfs(files)
    if not combined_text:
        return jsonify({"error": "Could not extract text from the provided PDF files."}), 500

    # 2. Fetch financial data
    profit_and_loss_html, balance_sheet_html = get_yearly_financial_statements_html(ticker)
    if not profit_and_loss_html or not balance_sheet_html:
        return jsonify({"error": f"Could not fetch financial statements for ticker: {ticker}"}), 500
    financial_data = extract_financial_data_from_html(profit_and_loss_html)

    # 3. Call Gemini API for all text analysis sections
    business_model_md = call_gemini(PROMPTS["business_model"], combined_text)
    quarterly_updates_md = call_gemini(PROMPTS["key_quarterly_updates"], combined_text)
    mgmt_commentary_md = call_gemini(PROMPTS["management_commentary"], combined_text)
    risks_md = call_gemini(PROMPTS["risks_uncertainties"], combined_text)
    prompt_set_md = call_gemini(PROMPTS["prompt_set"], combined_text)
    
    # 4. Get financial assumptions from Gemini
    financial_assumptions_json_str = call_gemini(PROMPTS["financial_assumptions"], combined_text)
    
    # 5. Calculate projections 
    import json
    projections_html = "" # Initialize to avoid reference errors
    try:
        # --- ⭐️ CORRECTED CODE STARTS HERE ⭐️ ---
        # Clean the string: remove backticks, "json" label, and strip whitespace
        if '```json' in financial_assumptions_json_str:
            clean_json_str = financial_assumptions_json_str.split('```json', 1)[1].rsplit('```', 1)[0]
        else:
            clean_json_str = financial_assumptions_json_str
        
        clean_json_str = clean_json_str.strip()
        
        # Now, parse the cleaned string
        assumptions = json.loads(clean_json_str)
        projections_html = calculate_financial_projections(assumptions, financial_data['Mar 2025'])
        # --- ⭐️ CORRECTED CODE ENDS HERE ⭐️ ---

    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error: Failed to decode JSON. Raw response was:\n{financial_assumptions_json_str}\nError details: {e}")
        projections_html = "<p style='color: red;'>Error: Could not parse financial assumptions from AI. The model returned an invalid format.</p>"


    # 6. Assemble the final report into a JSON object
    report_data = {
        "businessModel": business_model_md,
        "keyQuarterlyUpdates": quarterly_updates_md,
        "managementCommentary": mgmt_commentary_md,
        "financialProjections": projections_html,
        "risksUncertainties": risks_md,
        "promptSet": prompt_set_md
    }

    return jsonify(report_data)

# ... (keep the rest of the file as is) ...


# --- Route to Serve the Frontend ---
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# --- Run the App ---
if __name__ == '__main__':

    app.run(port=5000, debug=True ,host='0.0.0.0')
