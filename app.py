import os
import functools
from utils.logger import get_logger
logger = get_logger(__name__)
from flask import Flask, Blueprint, request, jsonify, send_from_directory, send_file
from dotenv import load_dotenv
from redis_config import get_queue, get_redis  # updated import path (same directory)
from tasks import generate_financial_report, generate_financial_report_from_rag
from rq.job import Job
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import io
from pdf_export import generate_quant_pdf # updated import path (same directory)

# --- Lifted Lazy Imports ---
from flask import Response
from flask import send_file
from llm_clients import client, deepseek_model_name
from rag_engine import get_collection_stats
from rag_engine import ingest_documents
from rag_engine import query as rag_query, get_collection_stats
from screener_scraper import fetch_screener_tables
from weasyprint import HTML
import asyncio
import datetime
import glob
import io
import json as _json
import queue
import threading
import time as _time


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static')
# --- Blueprint Configuration ---
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')


# --- Security: CORS restricted to allowed origins ---
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5000,http://127.0.0.1:3000,http://127.0.0.1:5000"
).split(",")
CORS(app, origins=ALLOWED_ORIGINS)

# --- Security: Rate Limiting ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="memory://",
)

# --- Security: API Key Middleware ---
API_KEY = os.getenv("NOVUS_API_KEY")  # Set in .env; if unset, auth is disabled (dev mode)

def require_api_key(f):
    """Decorator to enforce X-API-Key header on protected endpoints."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:  # Only enforce if NOVUS_API_KEY is set
            provided = request.headers.get("X-API-Key", "")
            if provided != API_KEY:
                return jsonify({"error": "Unauthorized — invalid or missing X-API-Key header"}), 401
        return f(*args, **kwargs)
    return decorated

# --- Security: Path Traversal Protection ---
ALLOWED_INGEST_PREFIXES = [
    p.strip() for p in
    os.getenv("ALLOWED_INGEST_PATHS", os.path.expanduser("~/Desktop")).split(",")
    if p.strip()
]
@api_v1.route('/generate_report', methods=['POST'])
@require_api_key
@limiter.limit("10 per hour")
def generate_report():
    """Start background task for report generation"""
    if 'files' not in request.files or 'ticker' not in request.form:
        return jsonify({"error": "Missing files or ticker symbol"}), 400

    ticker = request.form['ticker']
    files = request.files.getlist('files')

    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    # Convert files to bytes for background processing
    files_data = []
    for file in files:
        files_data.append(file.read())

    # Queue the background task
    queue = get_queue()
    job = queue.enqueue(
        generate_financial_report,
        ticker,
        files_data,
        #retry=3,               # auto retries (requires rq.Retry)
        ttl=3600,               # job expires after 1h if not started
        result_ttl=3600,        # keep result for 1h
        failure_ttl=7200,       # keep failure info 2h
        job_timeout=600         # hard timeout 10 min
    )

    return jsonify({
        "job_id": job.id,
        "status": "queued",
        "message": "Report generation started. Use the job_id to check status."
    })


@api_v1.route('/analyze_rag', methods=['POST'])
@require_api_key
@limiter.limit("10 per hour")
def analyze_rag():
    """
    RAG-Only Analysis: Just provide a ticker.
    The system pulls all context from ChromaDB + Screener.in automatically.
    Body: { "ticker": "HUL" }
    """
    data = request.get_json()
    if not data or 'ticker' not in data:
        return jsonify({"error": "Missing 'ticker' in request body"}), 400

    ticker = data['ticker'].upper()

    # Queue the background task
    queue = get_queue()
    job = queue.enqueue(
        generate_financial_report_from_rag,
        ticker,
        ttl=3600,
        result_ttl=3600,
        failure_ttl=7200,
        job_timeout=900,          # 15 min timeout (RAG queries + LLM calls)
    )

    return jsonify({
        "job_id": job.id,
        "status": "queued",
        "ticker": ticker,
        "mode": "rag_only",
        "message": f"RAG-only analysis for {ticker} started. Use /job_status/{job.id} to check progress."
    })

@api_v1.route('/job_status/<job_id>')
@limiter.exempt
def job_status(job_id):
    """Check the status of a background job"""
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:
        return jsonify({"error": "job not found"}), 404

    status = job.get_status()
    # Map RQ statuses to frontend-friendly ones
    ui_status = (
        "completed" if status == "finished" else
        "processing" if status == "started" else
        status
    )
    payload = {
        "job_id": job.id,
        "status": ui_status,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
        "progress": getattr(job, "meta", {}),
    }
    if status == "finished":
        payload["result"] = job.result
    if status == "failed":
        payload["error"] = job.exc_info
    return jsonify(payload)

@app.route('/')
def serve_frontend():
    """Serve the main frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health():
    try:
        get_redis().ping()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500

@api_v1.route('/screener_data', methods=['GET'])
def get_screener_data():
    """Fetch synchronous numerical table data from Screener.in"""
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Missing ticker parameter"}), 400
        
    try:
        data = fetch_screener_tables(ticker)
        if "error" in data:
            return jsonify(data), 500
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── RAG Chat Endpoint ────────────────────────────────────────────────────────

@api_v1.route('/chat', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def chat():
    """
    RAG-powered chat: Ask questions about a company using stored documents.
    Body: { "ticker": "HINDUNILVR", "question": "What is HUL's competitive moat?", "history": [] }
    """
    data = request.get_json()
    if not data or 'ticker' not in data or 'question' not in data:
        return jsonify({"error": "Missing 'ticker' or 'question'"}), 400

    ticker = data['ticker'].upper()
    question = data['question'].strip()
    history = data.get('history', [])  # list of {role, content} dicts

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:

        # Check if we have data for this ticker
        stats = get_collection_stats(ticker)
        if stats['total_chunks'] == 0:
            return jsonify({
                "answer": f"No documents found for **{ticker}** in the RAG store. Please ingest documents first using the /ingest_local endpoint.",
                "sources": [],
                "chunks_used": 0
            })

        # Query RAG for relevant chunks
        results = rag_query(ticker, question, top_k=8)

        # Build context from retrieved chunks
        context_parts = []
        sources = []
        seen = set()
        for r in results:
            meta = r.get('metadata', {})
            filename = meta.get('filename', 'unknown')
            doc_type = meta.get('doc_type', 'unknown')
            section = meta.get('section', '')
            source_key = f"{filename}|{section}"
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "filename": filename,
                    "doc_type": doc_type,
                    "section": section,
                    "relevance": round(1 - r.get('distance', 0.5), 2)
                })
            context_parts.append(
                f"[Source: {filename} | Type: {doc_type} | Section: {section}]\n{r['text']}"
            )

        rag_context = "\n\n---\n\n".join(context_parts)

        # Build messages for DeepSeek
        system_prompt = f"""You are Novus, an expert financial analyst assistant. You answer questions about {ticker} using ONLY the provided document context. 

Rules:
- Base your answers ONLY on the provided context. If the context doesn't contain enough information, say so clearly.
- CRITICAL: If you see context from "Specialist Agents", "FSAQUANT", "CAPITALALLOCATOR", or any internal MAS logs, IGNORE IT. You must form your own independent opinion purely from raw company filings.
- Cite specific sources when making claims (e.g., "According to the Q3 2024 transcript...").
- Be precise with numbers and financial metrics.
- Format your response in clean markdown with headers, bullet points, and bold text for key figures.
- Keep answers concise but comprehensive."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 6 messages max)
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current question with RAG context
        user_message = f"""Based on the following document excerpts for {ticker}, please answer my question.

## Retrieved Document Context
{rag_context}

## My Question
{question}"""

        messages.append({"role": "user", "content": user_message})

        # Call DeepSeek
        response = client.chat.completions.create(
            model=deepseek_model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )

        answer = response.choices[0].message.content

        return jsonify({
            "answer": answer,
            "sources": sources[:5],
            "chunks_used": len(results),
            "ticker": ticker
        })

    except Exception as e:
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500


# ── RAG Local Folder Ingestion ────────────────────────────────────────────────

@app.route('/ingest_local', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def ingest_local():
    """
    Ingest all PDFs from a local folder into the RAG vector store.
    Body: { "ticker": "RELIANCE", "folder_path": "/Users/.../Fin k10 copy" }
    """

    data = request.get_json()
    if not data or 'ticker' not in data:
        return jsonify({"error": "Missing 'ticker' in request body"}), 400

    ticker = data['ticker']
    folder_path = data.get('folder_path', os.path.expanduser('~/Desktop/Fin k10 copy'))

    # --- Security: Path traversal protection ---
    ALLOWED_INGEST_ROOTS = [os.path.expanduser("~/Desktop"), "/data/uploads"]
    if not any(os.path.commonpath([folder_path, root]) == root for root in ALLOWED_INGEST_ROOTS):
        return jsonify({"error": "Folder path not permitted"}), 403

    if not os.path.isdir(folder_path):
        return jsonify({"error": f"Folder not found: {folder_path}"}), 404

    # Find all PDFs in the folder (recursive)
    pdf_paths = glob.glob(os.path.join(folder_path, '**', '*.pdf'), recursive=True)
    pdf_paths += glob.glob(os.path.join(folder_path, '**', '*.PDF'), recursive=True)

    if not pdf_paths:
        return jsonify({"error": f"No PDF files found in {folder_path}"}), 404

    # Read files into (filename, bytes) pairs
    files_data = []
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as f:
                filename = os.path.basename(pdf_path)
                files_data.append((filename, f.read()))
        except Exception as e:
            logger.info(f"[Ingest] Skipped {pdf_path}: {e}")

    if not files_data:
        return jsonify({"error": "Could not read any PDF files"}), 500

    # Ingest into RAG
    try:
        result = ingest_documents(ticker, files_data)
        return jsonify({
            "status": "success",
            "ticker": ticker,
            "folder": folder_path,
            "files_processed": len(files_data),
            "filenames": [f[0] for f in files_data],
            **result,
        })
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {str(e)}"}), 500


@app.route('/rag_stats/<ticker>')
def rag_stats(ticker):
    """Get RAG store stats for a ticker."""
    try:
        stats = get_collection_stats(ticker.upper())
        return jsonify({"ticker": ticker.upper(), **stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/list_local_pdfs', methods=['POST'])
def list_local_pdfs():
    """List PDFs in a local folder (preview before ingesting)."""

    data = request.get_json()
    folder_path = data.get('folder_path', os.path.expanduser('~/Desktop/Fin k10 copy'))

    if not os.path.isdir(folder_path):
        return jsonify({"error": f"Folder not found: {folder_path}"}), 404

    pdf_paths = glob.glob(os.path.join(folder_path, '**', '*.pdf'), recursive=True)
    pdf_paths += glob.glob(os.path.join(folder_path, '**', '*.PDF'), recursive=True)

    files_info = []
    for p in sorted(set(pdf_paths)):
        stat = os.stat(p)
        files_info.append({
            "filename": os.path.basename(p),
            "path": p,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
        })

    return jsonify({
        "folder": folder_path,
        "total_pdfs": len(files_info),
        "files": files_info,
    })

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """Generate a professional, print-ready PDF using WeasyPrint."""
    data = request.get_json()
    if not data or 'content_html' not in data:
        return jsonify({"error": "Missing 'content_html' in request"}), 400
        
    ticker = data.get('ticker', 'REPORT').upper()
    content_html = data['content_html']
    
    # Wrap in minimal HTML with "Light-Mode Quant" strict typography CSS
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            @page {
                margin: 2cm;
                @top-right {
                    content: "Novus FinLLM Report - __TICKER__";
                    font-family: monospace;
                    font-size: 8pt;
                    color: #666;
                }
                @bottom-center {
                    content: counter(page) " / " counter(pages);
                    font-family: monospace;
                    font-size: 8pt;
                    color: #666;
                }
            }
            body {
                font-family: "Georgia", serif;
                font-size: 11pt;
                line-height: 1.5;
                color: #111;
                background: #fff;
            }
            h1, h2, h3, h4 {
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                color: #000;
                border-bottom: 1px solid #ccc;
                padding-bottom: 5px;
            }
            pre, code {
                font-family: "Courier New", Courier, monospace;
                font-size: 9pt;
                background: #f7f7f7;
                padding: 2px 4px;
                border-radius: 3px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 1.5em 0;
                font-family: "Courier New", Courier, monospace;
                font-size: 9pt;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f3f4f6;
                font-family: "Helvetica Neue", Helvetica, sans-serif;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #fafafa;
            }
            .calc-badge {
                font-family: monospace;
                font-weight: bold;
                background: #eee;
                padding: 1px 3px;
                border: 1px solid #ccc;
            }
        </style>
    </head>
    <body>
        <div style="font-family: monospace; text-align: center; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 20px;">
            <h1 style="border:none; margin:0;">Novus Institutional Report</h1>
            <div>__TICKER__</div>
        </div>
        __CONTENT_HTML__
    </body>
    </html>
    """.replace("__TICKER__", ticker).replace("__CONTENT_HTML__", content_html)
    
    try:
        
        pdf_bytes = HTML(string=html_template).write_pdf()
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'{ticker}_Novus_Analysis.pdf'
        )
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500




app.register_blueprint(api_v1)

if __name__ == '__main__':
    app.run(port=5001, debug=True, host='0.0.0.0')
