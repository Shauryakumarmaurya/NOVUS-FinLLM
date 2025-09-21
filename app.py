# app.py
import os
import google.generativeai as genai
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
app = Flask(__name__, static_folder='static')
gemini_api_key = os.getenv("GEMINI_API_KEY") # Gemini API key from environment variable
fmp_api_key = os.getenv("FMP_API_KEY") # Financial Modeling Prep API key (if needed)

# Configure the Gemini API client
genai.configure(api_key=gemini_api_key) 
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using a modern, efficient model

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


def get_yearly_financial_statements_html(ticker: str) -> tuple[str | None, str | None]:
    """
    Fetches and extracts the yearly Profit & Loss and Balance Sheet statements 
    for a given stock ticker from screener.in.

    The web can be unreliable, so this function includes error handling for network 
    issues and changes in website structure.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'RELIANCE').

    Returns:
        tuple[str | None, str | None]: A tuple containing two elements:
            - The HTML string of the Profit & Loss table, or None if not found.
            - The HTML string of the Balance Sheet table, or None if not found.
    """
    try:
        # The URL is constructed for the consolidated statements page.
        url = f"https://www.screener.in/company/{ticker}/consolidated/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Fetching data from {url}...")
        response = requests.get(url, headers=headers)
        # Raise an exception for HTTP errors (e.g., 404, 500)
        response.raise_for_status()  

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Extract Profit & Loss Statement ---
        # The financial statements are in sections with specific IDs.
        profit_loss_section = soup.find('section', id='profit-loss')
        profit_loss_html = None
        if profit_loss_section:
            # The table is within a specific div inside the section.
            profit_loss_table = profit_loss_section.find('table', class_='data-table')
            if profit_loss_table:
                profit_loss_html = str(profit_loss_table)
            else:
                print("Warning: Profit & Loss data table not found on the page.")
        else:
            print("Warning: Profit & Loss section not found on the page.")

        # --- Extract Balance Sheet Statement ---
        balance_sheet_section = soup.find('section', id='balance-sheet')
        balance_sheet_html = None
        if balance_sheet_section:
            balance_sheet_table = balance_sheet_section.find('table', class_='data-table')
            if balance_sheet_table:
                balance_sheet_html = str(balance_sheet_table)
            else:
                print("Warning: Balance Sheet data table not found on the page.")
        else:
            print("Warning: Balance Sheet section not found on the page.")
            
        return profit_loss_html, balance_sheet_html

    except requests.exceptions.RequestException as e:
        print(f"Error: A network error occurred while fetching the data: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

def get_financial_data(ticker):
    """
    Scrapes key financial data for the most recent quarter for AI analysis.
    This function remains to ensure the projection logic does not break.
    """
    print(f"Scraping LATEST QUARTER data for {ticker} from Screener.in...")
    try:
        url = f"https://www.screener.in/company/{ticker}/consolidated/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        quarters_table = soup.select_one("section#quarters table")
        if not quarters_table:
            raise ValueError("Could not find the 'Quarterly Results' table.")

        header_row = quarters_table.find('thead').find_all('th')
        last_column_index = len(header_row) - 1

        def get_metric(table, metric_name):
            for row in table.select("tbody tr"):
                first_cell = row.find("td")
                if not first_cell:
                    continue

                # Extract text even if it's inside <button> or nested tags
                row_name = first_cell.get_text(strip=True)

                if metric_name.lower() in row_name.lower():
                    # All numeric cells after the label
                    value_cells = row.find_all("td")[1:]
                    last_value = value_cells[-1].get_text(strip=True).replace(",", "").replace("−", "-")

                    try:
                        return float(last_value)
                    except ValueError:
                        return 0.0
            return 0.0

        return {
            "totalRevenue": get_metric(quarters_table, "Sales"),          # not "Revenue"
            "ebitda": get_metric(quarters_table, "Operating Profit"),
            "pbt": get_metric(quarters_table, "Profit before tax"),
            "pat": get_metric(quarters_table, "Net Profit")               # not "PAT"
        }

    except Exception as e:
        print(f"An error occurred during quarterly scraping: {e}")
        return {"totalRevenue": 0, "ebitda": 0, "pbt": 0, "pat": 0}




def get_yearly_financial_data(ticker):
    """
    Scrapes YEARLY Profit & Loss and Balance Sheet data from Screener.in
    and extracts key metrics (Sales, EBITDA, PBT, PAT).
    """
    print(f"Scraping YEARLY statements for {ticker} from Screener.in...")
    try:
        url = f"https://www.screener.in/company/{ticker}/consolidated/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')

        # Locate the yearly Profit & Loss table
        profit_loss_section = soup.find('section', id='profit-loss')
        if not profit_loss_section:
            raise ValueError("Could not find the Profit & Loss section.")

        pl_table = profit_loss_section.find("table")
        if not pl_table:
            raise ValueError("Could not find the Profit & Loss table.")

        # --- Helper function to get last available value for a metric ---
        def get_metric(table, metric_name):
            for row in table.select("tbody tr"):
                first_cell = row.find("td")
                if not first_cell:
                    continue

                row_name = first_cell.get_text(strip=True)
                if metric_name.lower() in row_name.lower():
                    # All numeric cells after the label
                    value_cells = row.find_all("td")[1:]
                    if not value_cells:
                        return 0.0
                    last_value = value_cells[-1].get_text(strip=True).replace(",", "").replace("−", "-")
                    try:
                        return float(last_value)
                    except ValueError:
                        return 0.0
            return 0.0

        # Extract metrics from Profit & Loss
        return {
            "totalRevenue": get_metric(pl_table, "Sales"),          # yearly sales
            "ebitda": get_metric(pl_table, "Operating Profit"),     # yearly EBITDA
            "pbt": get_metric(pl_table, "Profit before tax"),       # yearly PBT
            "pat": get_metric(pl_table, "Net Profit")               # yearly PAT
        }

    except Exception as e:
        print(f"An error occurred during yearly scraping: {e}")
        return {"totalRevenue": 0, "ebitda": 0, "pbt": 0, "pat": 0}


def extract_financial_data_from_html(profit_and_loss_html: str, balance_sheet_html: str) -> dict:
    """
    Extracts key financial metrics for the most recent year from HTML tables.

    Args:
        profit_and_loss_html (str): The HTML string of the Profit & Loss table.
        balance_sheet_html (str): The HTML string of the Balance Sheet table.

    Returns:
        dict: A dictionary containing the extracted financial metrics.
    """
    
    def _get_metric_from_table(table_soup, metric_name, is_pnl=False):
        """Helper function to find and extract a metric from a parsed table."""
        try:
            for row in table_soup.select("tbody tr"):
                # Find the cell with the metric name
                first_cell = row.find("td")
                if not first_cell:
                    continue

                row_name = first_cell.get_text(strip=True)

                # Check if the desired metric name is in this row's title
                if metric_name.lower() in row_name.lower():
                    # Get all numerical value cells for the row
                    value_cells = row.find_all("td")[1:]
                    if not value_cells:
                        return 0.0

                    # For P&L statement, the last column is TTM, so we take the second to last.
                    # For Balance Sheet, the last column is the most recent year.
                    target_index = -2 if is_pnl else -1
                    
                    # Ensure we don't get an index error if there are not enough columns
                    if len(value_cells) >= abs(target_index):
                         last_value_str = value_cells[target_index].get_text(strip=True)
                    else: # Fallback to the last available column
                        last_value_str = value_cells[-1].get_text(strip=True)


                    # Clean and convert the value to a float
                    if not last_value_str:
                        return 0.0
                    cleaned_value = last_value_str.replace(",", "").replace("−", "-")
                    return float(cleaned_value)
            
            # Return 0.0 if the metric was not found
            return 0.0

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse value for '{metric_name}'. Error: {e}")
            return 0.0
    
    # Parse the HTML strings into BeautifulSoup objects
    pl_soup = BeautifulSoup(profit_and_loss_html, 'html.parser')
    bs_soup = BeautifulSoup(balance_sheet_html, 'html.parser')

    # Define the metrics to be extracted
    financial_data = {
        # --- From Profit & Loss ---
        "sales": _get_metric_from_table(pl_soup, "Sales", is_pnl=True),
        "operating_profit": _get_metric_from_table(pl_soup, "Operating Profit", is_pnl=True),
        "interest": _get_metric_from_table(pl_soup, "Interest", is_pnl=True),
        "depreciation": _get_metric_from_table(pl_soup, "Depreciation", is_pnl=True),
        "pbt": _get_metric_from_table(pl_soup, "Profit before tax", is_pnl=True),
        "net_profit": _get_metric_from_table(pl_soup, "Net Profit", is_pnl=True),
        "eps": _get_metric_from_table(pl_soup, "EPS in Rs", is_pnl=True),

        # --- From Balance Sheet ---
        "borrowings": _get_metric_from_table(bs_soup, "Borrowings"),
        "equity_capital": _get_metric_from_table(bs_soup, "Equity Capital"),
        "reserves": _get_metric_from_table(bs_soup, "Reserves"),
    }

    return financial_data





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
    full_prompt = f"{prompt}\n\nHere is the text to analyze:\n\n---\n\n{text_to_analyze}"
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
    year0_ebitda = financial_data['operating_profit'] + financial_data['depreciation']
    
    # Calculate the number of shares (assumed to be constant)
    num_shares = calculate_number_of_shares(financial_data['net_profit'], financial_data['eps'])

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
        revenues_proj = calculate_revenue_growth(financial_data['sales'], growth_rate, 2)
        ebitdas_proj = calculate_ebitda(revenues_proj, ebitda_margin)
        pbts_proj = calculate_pbt(ebitdas_proj, financial_data['interest'], financial_data['depreciation'])
        pats_proj = calculate_pat(pbts_proj, tax_rate)
        eps_proj = calculate_eps(pats_proj, num_shares)

        # Combine Year 0 data with projections for the table
        data_rows = {
            "Revenue": [financial_data['sales']] + revenues_proj,
            "EBITDA": [year0_ebitda] + ebitdas_proj,
            "PBT": [financial_data['pbt']] + pbts_proj,
            "PAT": [financial_data['net_profit']] + pats_proj,
            "EPS": [financial_data['eps']] + eps_proj
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
    financial_data = extract_financial_data_from_html(profit_and_loss_html, balance_sheet_html)

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
        projections_html = calculate_financial_projections(assumptions, financial_data)
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
    app.run(host='0.0.0.0', port=5000, debug=False)