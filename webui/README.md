# webui — Local test bench (5 photos → 3D model)

**Local** web interface to try out the 3D teeth reconstruction from 5 intra-oral
photos: upload the photos (or pick a sample set), press **Ricostruisci**, and watch
the two 3D arches spin in the browser, with `.obj` download.

The computation stays **in Python on your own PC** (CPU). The interface is only the
shop window: Flask receives the photos and launches the pipeline as an **isolated
subprocess**, so the heavy stack (TensorFlow 2.6 / Ray / open3d) never shares a
process with Flask.

> **Why not Cloudflare Pages.** Pages/Workers serve static files plus small JS/WASM
> Functions: they **cannot** run Python, TensorFlow, Ray or open3d, nor keep running
> for minutes. The *frontend* could live on Pages, but the *engine* needs a machine
> with Python (this PC, or HuggingFace Spaces / Modal / a VM). For now: everything
> runs locally.

---

## How to start it

```bash
# from the repository root
.venv/Scripts/python.exe webui/server.py
# then open http://127.0.0.1:5000
```

In the page: fill the 5 slots (upper, lower, left, right, frontal) **or** pick a
sample set (0 / 1) and press **Carica**, then **Ricostruisci**.

- ⏱️ One reconstruction takes **~7 minutes on CPU** (measured on tag 0: 425 s). The
  progress bar shows the live log (segmentation → EM → mesh).
- The pipeline assumes **all teeth are present** (28 = 14 upper + 14 lower, third
  molars excluded). Missing teeth cannot be selected from the interface yet.

---

## Layout

| File | What it does |
|---|---|
| `server.py` | Flask app: receives the photos, starts the subprocess, exposes its state (polling) and serves the meshes. Never imports the ML stack. |
| `reconstruct_api.py` | CLI run as a subprocess: 5 photos → 2 `.obj`. Reuses `loadMuEigValSigma`, `run_emopt` and `create_mesh_from_emopt_h5File` from `main.py`. |
| `static/index.html` | Frontend: 5 photo slots, sample loading, progress, three.js viewer (CDN). |
| `jobs/` | Local runs (photos + generated meshes). **Git-ignored.** |

---

## Python environment (one-time setup)

The original stack is from 2021 and needs **Python 3.9** (not 3.11+). `.venv/` is not
versioned; to recreate it:

```bash
py -3.9 -m venv .venv
.venv/Scripts/python.exe -m pip install -U pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
# Flask for the webui
.venv/Scripts/python.exe -m pip install flask
```

Four version pins are required (the usual conflicts of a stack this old), already
applied in the working venv:

- `numpy==1.19.5` and `matplotlib==3.5.3` — TF 2.6 requires numpy ~1.19.
- `pydantic==1.10.13` — Ray 2.0.1 uses the pydantic **v1** API.
- `dash==2.9.3` — open3d 0.16 imports `dash` at startup; 4.x demands pydantic v2.
- `werkzeug==2.2.3` — paired with Flask 2.2.5 (3.x breaks the import).

---

## Status (current)

- ✅ Python 3.9 environment working; pipeline verified **end-to-end** on CPU (tag 0
  completed in 425 s → valid upper/lower meshes).
- ✅ webui complete: upload of 5 photos / sample sets, live progress, three.js 3D
  viewer with arch toggles and `.obj` download.

### Possible next steps
- Missing-teeth selection (pass a `tooth_exist_mask` from the interface).
- Overlay of the projection onto the 5 shots (like `visualization.py`).
- Cut the runtime (shorter grid search / fewer EM iterations) or move the engine to a
  GPU machine for online use.
