"""
server.py - local test UI for 5-view teeth reconstruction.

Flask stays lightweight: it never imports the heavy ML stack. Each reconstruction
runs as an isolated subprocess (reconstruct_api.py) whose stdout is tailed for
live progress. Single-user local tool - no auth, no concurrency guarantees.

Run:  .venv/Scripts/python.exe webui/server.py
Then open http://127.0.0.1:5000
"""

import os
import shutil
import subprocess
import sys
import time

from flask import (Flask, jsonify, request, send_file,
                   send_from_directory, abort)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBUI_DIR = os.path.join(REPO_DIR, "webui")
JOBS_DIR = os.path.join(WEBUI_DIR, "jobs")
SAMPLE_IMG_DIR = os.path.join(REPO_DIR, "seg", "valid", "image")
VIEWS = ["upper", "lower", "left", "right", "frontal"]
# sample photo naming: "<tag>-<0..4>.png" mapped to upper/lower/left/right/frontal
VIEW_INDEX = {"upper": 0, "lower": 1, "left": 2, "right": 3, "frontal": 4}

os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(WEBUI_DIR, "static"),
            static_url_path="/static")

# job_id -> {"proc": Popen, "dir": path}
JOBS = {}


def _job_dir(job_id):
    d = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(d):
        abort(404)
    return d


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/samples")
def list_samples():
    """Available sample tags (from seg/valid/image/<tag>-<0..4>.png)."""
    tags = set()
    if os.path.isdir(SAMPLE_IMG_DIR):
        for f in os.listdir(SAMPLE_IMG_DIR):
            if f.endswith(".png") and "-" in f:
                tags.add(f.split("-")[0])
    return jsonify(sorted(tags))


@app.route("/api/sample-image/<tag>/<view>")
def sample_image(tag, view):
    idx = VIEW_INDEX.get(view)
    if idx is None:
        abort(404)
    path = os.path.join(SAMPLE_IMG_DIR, "{}-{}.png".format(tag, idx))
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/api/reconstruct", methods=["POST"])
def reconstruct():
    job_id = time.strftime("%Y%m%d-%H%M%S")
    job_path = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    saved = {}
    # Either uploaded files, or a chosen sample tag copied server-side.
    sample_tag = request.form.get("sample_tag")
    if sample_tag:
        for view in VIEWS:
            src = os.path.join(SAMPLE_IMG_DIR,
                               "{}-{}.png".format(sample_tag, VIEW_INDEX[view]))
            if not os.path.exists(src):
                return jsonify({"error": "sample {} missing view {}".format(
                    sample_tag, view)}), 400
            dst = os.path.join(job_path, view + ".png")
            shutil.copyfile(src, dst)
            saved[view] = dst
    else:
        for view in VIEWS:
            if view not in request.files:
                return jsonify({"error": "missing photo: " + view}), 400
            f = request.files[view]
            dst = os.path.join(job_path, view + ".png")
            f.save(dst)
            saved[view] = dst

    log_path = os.path.join(job_path, "progress.log")
    log_f = open(log_path, "w", encoding="utf-8")
    cmd = [
        sys.executable, os.path.join(WEBUI_DIR, "reconstruct_api.py"),
        "--upper", saved["upper"], "--lower", saved["lower"],
        "--left", saved["left"], "--right", saved["right"],
        "--frontal", saved["frontal"], "--out", job_path,
    ]
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=log_f,
                            stderr=subprocess.STDOUT)
    JOBS[job_id] = {"proc": proc, "dir": job_path, "log": log_path}
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    d = _job_dir(job_id)
    log_path = os.path.join(d, "progress.log")
    tail = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = "".join(lines[-40:])

    state = "running"
    if "RECON_DONE" in tail:
        state = "done"
    elif "RECON_ERROR" in tail:
        state = "error"
    else:
        job = JOBS.get(job_id)
        if job and job["proc"].poll() is not None and state == "running":
            # process exited without a DONE marker -> treat as error
            state = "error"

    resp = {"state": state, "log": tail}
    if state == "done":
        resp["upper_url"] = "/api/mesh/{}/upper".format(job_id)
        resp["lower_url"] = "/api/mesh/{}/lower".format(job_id)
    return jsonify(resp)


@app.route("/api/mesh/<job_id>/<which>")
def mesh(job_id, which):
    d = _job_dir(job_id)
    name = {"upper": "Pred_Upper_Mesh_Tag=result.obj",
            "lower": "Pred_Lower_Mesh_Tag=result.obj"}.get(which)
    if not name:
        abort(404)
    path = os.path.join(d, "result", name)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="text/plain",
                     as_attachment=False, download_name=which + ".obj")


if __name__ == "__main__":
    print("Teeth reconstruction UI -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
