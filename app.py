import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from redis_config import get_queue, get_redis  # updated import path (same directory)
from tasks import generate_financial_report
from rq.job import Job

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static')

@app.route('/generate_report', methods=['POST'])
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

@app.route('/job_status/<job_id>')
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

if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')