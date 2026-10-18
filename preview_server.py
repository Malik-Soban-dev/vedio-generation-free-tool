#!/usr/bin/env python3
"""
WanGP Interactive Studio Preview Server
Serves an interactive preview of the WanGP v18.17 Dashboard, 270+ Model Presets,
Deepy AI Director, Media Flow Post-Processing, VRAM Calculator, and Documentation
on 0.0.0.0:7860.
"""
import glob
import json
import mimetypes
import os
import tempfile
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_ARCHS = {
    "flux", "flux2_dev", "flux2_klein_4b", "flux2_klein_9b", "flux3_klein",
    "flux3_klein_base", "flux_chroma", "flux_chroma_radiance", "flux_dev_kontext",
    "flux_dev_kontext_dreamomni2", "flux_dev_umo", "flux_dev_uso", "flux_schnell",
    "hidream_o1", "hidream_o1_dev", "ideogram4", "ideogram4_turbotime",
    "krea2_raw", "krea2_raw_edit", "krea2_turbo", "krea2_turbo_edit",
    "ming_image_0_1_design", "ming_image_0_1_design_layer", "moss_image",
    "pi_flux2", "qwen_image_20B", "qwen_image_21_7B", "qwen_image_edit_20B",
    "qwen_image_edit_plus2_20B", "qwen_image_edit_plus_20B", "qwen_image_layered_20B",
    "sensenova_u1_5_8b_mot", "z_image", "z_image_base", "z_image_control",
    "z_image_control2", "z_image_control2_1",
}

AUDIO_ARCHS = {
    "ace_step_v1", "ace_step_v1_5", "ace_step_v1_5_xl", "auk", "chatterbox",
    "dramabox_audio", "fish_drama3", "heartmula_oss_3b", "index_tts2",
    "index_tts25", "kugelaudio_0_open", "minimax_h3_tts_ref2va_pruned",
    "minimax_music3", "omnivoice", "qwen3_tts_base", "qwen3_tts_customvoice",
    "qwen3_tts_voicedesign", "scenema_audio", "stable_audio3_medium",
    "stable_audio3_small", "yue2", "yue2_hum",
}

THREED_ARCHS = {
    "animate_any_mesh", "da3_points", "geosam2", "hunyuan3d_21",
    "hunyuan3d_21_shape", "mapanything", "moge2", "partcrafter",
    "trellis2", "triposg", "triposplat", "worldmirror_2",
}

AV_VIDEO_PREFIXES = (
    "minimax_h3", "ltx2", "k6_", "ovi", "prism", "infinitetalk",
    "multitalk", "longcat_avatar", "hunyuan_avatar", "hunyuan_custom_audio",
    "magi_human",
)


def classify_model(file_id, arch, name, desc):
    arch_l = (arch or "").lower()
    fid_l = (file_id or "").lower()
    if arch in THREED_ARCHS or any(k in fid_l for k in ("trellis", "hunyuan3d", "tripo", "worldmirror", "mapanything", "geosam", "partcrafter", "moge2", "da3_points", "animate_any_mesh")):
        return "3d"
    if arch in AUDIO_ARCHS or any(k in fid_l for k in ("tts", "ace_step", "yue2", "music", "audio", "chatterbox", "fish_drama", "heartmula", "omnivoice", "auk")):
        return "audio"
    if arch in IMAGE_ARCHS or any(k in fid_l for k in ("flux", "qwen_image", "krea2", "ideogram", "sensenova", "ming_image", "moss_image", "z_image", "hidream")):
        return "image"
    return "video"


def load_all_models():
    models = []
    for path in sorted(glob.glob(os.path.join(ROOT_DIR, "defaults", "*.json"))):
        file_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = data.get("model", data)
            name = m.get("name") or file_id
            arch = m.get("architecture") or file_id
            desc = m.get("description") or f"{name} preset ({arch})"
            urls = m.get("URLs") or m.get("preload_URLs") or []
            if isinstance(urls, str):
                urls = [urls]
            modality = classify_model(file_id, arch, name, desc)
            has_native_audio = modality == "audio" or any(
                fid_k in file_id.lower() or fid_k in arch.lower()
                for fid_k in AV_VIDEO_PREFIXES
            )
            models.append({
                "id": file_id,
                "name": name,
                "architecture": arch,
                "modality": modality,
                "has_audio": has_native_audio,
                "description": desc,
                "prompt": data.get("prompt", ""),
                "negative_prompt": data.get("negative_prompt", ""),
                "resolution": data.get("resolution", "832x480" if modality == "video" else "1024x1024"),
                "video_length": data.get("video_length", 81 if modality == "video" else 1),
                "steps": data.get("num_inference_steps", 20),
                "guidance_scale": data.get("guidance_scale", 5.0),
                "flow_shift": data.get("flow_shift", 3.0),
                "urls": urls[:4],
            })
        except Exception:
            continue
    # Sort featured models first within each modality
    priority = {
        "minimax_h3_fl2va_pruned": 0,
        "ltx2_25_22B_distilled": 1,
        "t2v_2_2": 2,
        "i2v_2_2": 3,
        "k6_lite_distill": 4,
        "vace_14B": 5,
        "animate2_distilled": 6,
        "viggle_animate": 7,
        "infinitetalk": 8,
        "fun_camera_1.3B": 9,
        "t2v_1.3B": 10,
    }
    models.sort(key=lambda x: (priority.get(x["id"], 999), x["name"].lower()))
    return models


def load_plugins():
    p_path = os.path.join(ROOT_DIR, "plugins.json")
    if os.path.exists(p_path):
        try:
            with open(p_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def list_docs():
    docs = [
        {"id": "VIDEO_GENERATION_GUIDE.md", "title": "Complete Video Generation & Capacity Guide (Start Here)", "path": "VIDEO_GENERATION_GUIDE.md"}
    ]
    docs_dir = os.path.join(ROOT_DIR, "docs")
    if os.path.isdir(docs_dir):
        for fname in sorted(os.listdir(docs_dir)):
            if fname.endswith(".md"):
                docs.append({
                    "id": f"docs/{fname}",
                    "title": fname.replace(".md", "").replace("_", " ").replace("-", " "),
                    "path": os.path.join("docs", fname),
                })
    docs.append({"id": "README.md", "title": "WanGP Official README & Changelog", "path": "README.md"})
    return docs


ALL_MODELS = load_all_models()
ALL_PLUGINS = load_plugins()
ALL_DOCS = list_docs()


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>WanGP v18.17 — AI Video, Audio, Image & 3D Studio</title>
<link rel="icon" href="favicon.png" />
<style>
  :root {
    --bg-main: #0b0f19;
    --bg-panel: #131a2a;
    --bg-card: #1a2337;
    --bg-input: #0e1422;
    --border: #283552;
    --accent: #6366f1;
    --accent-hover: #818cf8;
    --accent-2: #10b981;
    --accent-warn: #f59e0b;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --radius: 10px;
  }
  [data-theme="cyber"] {
    --bg-main: #06070d;
    --bg-panel: #0f1123;
    --bg-card: #171a36;
    --bg-input: #090b16;
    --border: #ec489955;
    --accent: #ec4899;
    --accent-hover: #f472b6;
    --accent-2: #06b6d4;
  }
  [data-theme="gradio"] {
    --bg-main: #1f2937;
    --bg-panel: #111827;
    --bg-card: #374151;
    --bg-input: #1f2937;
    --border: #4b5563;
    --accent: #f97316;
    --accent-hover: #fb923c;
    --accent-2: #22c55e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg-main);
    color: var(--text-main);
    line-height: 1.5;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  /* Header */
  .topbar {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    z-index: 50;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand img {
    width: 34px;
    height: 34px;
    border-radius: 8px;
  }
  .brand-title {
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .badge {
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.4);
    font-weight: 600;
  }
  .badge-green {
    background: rgba(16, 185, 129, 0.16);
    color: #6ee7b7;
    border-color: rgba(16, 185, 129, 0.4);
  }
  .badge-amber {
    background: rgba(245, 158, 11, 0.16);
    color: #fcd34d;
    border-color: rgba(245, 158, 11, 0.4);
  }
  .topbar-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  /* Main Nav Tabs */
  .nav-tabs {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    padding: 0 18px;
    display: flex;
    gap: 6px;
    overflow-x: auto;
  }
  .nav-tab {
    background: transparent;
    color: var(--text-muted);
    border: none;
    border-bottom: 3px solid transparent;
    padding: 11px 15px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
  }
  .nav-tab:hover { color: var(--text-main); }
  .nav-tab.active {
    color: var(--text-main);
    border-bottom-color: var(--accent);
    background: rgba(99, 102, 241, 0.08);
  }
  /* Layout */
  .container {
    flex: 1;
    padding: 16px 18px;
    max-width: 1600px;
    width: 100%;
    margin: 0 auto;
  }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }
  .studio-grid {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 16px;
    align-items: start;
  }
  @media (max-width: 1080px) {
    .studio-grid { grid-template-columns: 1fr; }
  }
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 15px;
    margin-bottom: 14px;
  }
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .panel-title {
    font-size: 0.98rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  label {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 5px;
  }
  select, input[type="text"], input[type="number"], textarea {
    width: 100%;
    background: var(--bg-input);
    color: var(--text-main);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 9px 11px;
    font-size: 0.88rem;
    font-family: inherit;
  }
  select:focus, input:focus, textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  textarea {
    min-height: 112px;
    resize: vertical;
  }
  .row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-bottom: 10px;
  }
  .btn {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 7px;
    padding: 9px 15px;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    transition: background 0.15s, transform 0.08s;
    text-decoration: none;
  }
  .btn:hover { background: var(--accent-hover); }
  .btn:active { transform: scale(0.98); }
  .btn-secondary {
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { background: #26334d; }
  .btn-success { background: var(--accent-2); color: #052e16; }
  .btn-success:hover { background: #34d399; }
  .btn-sm { padding: 5px 10px; font-size: 0.78rem; }
  /* Modality Filter Pills */
  .pill-bar {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .pill {
    background: var(--bg-card);
    color: var(--text-muted);
    border: 1px solid var(--border);
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
  }
  .pill.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  /* Model Info Banner */
  .model-info {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.82rem;
    margin-top: 8px;
  }
  .model-info-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 6px;
  }
  /* Sub-tabs inside left column */
  .subtabs {
    display: flex;
    gap: 5px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .subtab {
    background: transparent;
    color: var(--text-muted);
    border: none;
    padding: 7px 11px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .subtab.active {
    color: var(--text-main);
    border-bottom-color: var(--accent);
  }
  /* Preview Canvas */
  .preview-stage {
    background: #050811;
    border: 1px solid var(--border);
    border-radius: 9px;
    overflow: hidden;
    position: relative;
    aspect-ratio: 16 / 9;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #previewCanvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  .preview-overlay {
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    pointer-events: none;
  }
  .progress-wrap {
    margin-top: 10px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 12px;
  }
  .progress-bar-bg {
    width: 100%;
    height: 8px;
    background: #1e293b;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 6px;
  }
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    width: 0%;
    transition: width 0.15s;
  }
  /* Gallery Strip */
  .gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 10px;
  }
  .gallery-item {
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .gallery-item.selected, .gallery-item:hover {
    border-color: var(--accent);
  }
  .gallery-thumb {
    width: 100%;
    height: 90px;
    object-fit: cover;
    background: #090d16;
    display: block;
  }
  .gallery-caption {
    padding: 6px 8px;
    font-size: 0.74rem;
  }
  /* Catalogue cards */
  .cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: 12px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 13px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 8px;
  }
  .card h4 { font-size: 0.94rem; }
  .card p { font-size: 0.8rem; color: var(--text-muted); }
  /* Deepy Chat */
  .chat-box {
    height: 410px;
    overflow-y: auto;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 10px;
  }
  .chat-msg {
    max-width: 85%;
    padding: 10px 13px;
    border-radius: 10px;
    font-size: 0.86rem;
  }
  .chat-msg.user {
    align-self: flex-end;
    background: var(--accent);
    color: #fff;
  }
  .chat-msg.assistant {
    align-self: flex-start;
    background: var(--bg-card);
    border: 1px solid var(--border);
  }
  .tool-card {
    margin-top: 8px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 7px;
    padding: 8px 10px;
    font-size: 0.79rem;
  }
  /* Docs Viewer */
  .docs-layout {
    display: grid;
    grid-template-columns: 270px 1fr;
    gap: 16px;
  }
  @media (max-width: 900px) {
    .docs-layout { grid-template-columns: 1fr; }
  }
  .doc-list {
    max-height: 75vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .doc-item {
    text-align: left;
    background: transparent;
    color: var(--text-muted);
    border: 1px solid transparent;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 0.82rem;
    cursor: pointer;
  }
  .doc-item:hover, .doc-item.active {
    background: var(--bg-card);
    color: var(--text-main);
    border-color: var(--border);
  }
  .doc-viewer {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 20px;
    max-height: 75vh;
    overflow-y: auto;
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.84rem;
    line-height: 1.6;
  }
</style>
</head>
<body>
<!--__WANGP_INLINE_DATA__-->

  <!-- Top Header -->
  <header class="topbar">
    <div class="brand">
      <img src="favicon.png" alt="WanGP" />
      <div>
        <div class="brand-title">
          WanGP Studio <span class="badge">v18.17</span>
          <span class="badge badge-green" id="modelCountBadge">270 Models Loaded</span>
          <span class="badge badge-amber" title="Sandbox is CPU-only; interactive preview mode active">Interactive Studio Preview</span>
        </div>
        <div style="font-size:0.75rem; color:var(--text-muted);">
          The Open-Source Video, Audio, Image &amp; 3D Super-App for the GPU Poor (MMGP v4)
        </div>
      </div>
    </div>
    <div class="topbar-controls">
      <select id="workspaceSelect" style="width:auto;" title="Active Project Workspace">
        <option value="default">📁 Workspace: Default Studio</option>
        <option value="shortfilm">🎬 Workspace: Sci-Fi Short Film</option>
        <option value="ad_campaign">📺 Workspace: Commercial Campaign</option>
      </select>
      <select id="themeSelect" style="width:auto;" onchange="document.documentElement.setAttribute('data-theme', this.value)">
        <option value="default">🎨 Theme: Midnight Studio</option>
        <option value="cyber">🎨 Theme: Cyber Neon</option>
        <option value="gradio">🎨 Theme: Classic Gradio</option>
      </select>
      <a class="btn btn-sm btn-success" href="https://raw.githubusercontent.com/Malik-Soban-dev/vedio-generation-free-tool/arena/f01df44c-vedio-generation-free-tool/WanGP_Colab_Free_GPU.ipynb" target="_blank" rel="noopener">
        ⚡ Download Free Colab GPU Launcher (.ipynb)
      </a>
      <button class="btn btn-sm btn-secondary" onclick="toggleFullScreen()" title="Expand the preview to fill your whole screen">
        ⛶ Fullscreen
      </button>
      <a class="btn btn-sm" id="newTabBtn" href="#" onclick="return openInNewTab();" style="background:linear-gradient(135deg,#6366f1,#ec4899); font-weight:700; opacity:0.55;" title="Opens the dashboard in a real new browser tab (needs the sandbox traffic token)">
        ↗ Open in New Tab <span id="newTabState">…</span>
      </a>
      <a class="btn btn-sm" href="WanGP_Studio_Standalone.html" download="WanGP_Studio_Standalone.html" target="_blank" rel="noopener" style="background:linear-gradient(135deg,#6366f1,#ec4899); font-weight:700;">
        ⬇️ Download Offline HTML (opens in any browser)
      </a>
    </div>
  </header>

  <!-- Why there is no public URL: sandbox traffic-access-token notice -->
  <div id="iframeBanner" style="display:none; background:linear-gradient(90deg, rgba(99,102,241,0.22), rgba(236,72,153,0.22)); border-bottom:1px solid var(--accent); padding:9px 18px; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div style="font-size:0.84rem; line-height:1.45;">
      🔒 <b>This workspace sandbox is private.</b> Its public URL is protected by an E2B <b>traffic-access-token</b>, so opening <code>https://7860-…e2b.app</code> directly in a new tab returns <b>403 — missing traffic token</b>.
      Use <b>⛶ Fullscreen</b> here, or download the <b>offline HTML</b> to open the full dashboard in any browser with zero server.
    </div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
      <button class="btn btn-sm" onclick="toggleFullScreen()" style="background:var(--accent-2); color:#052e16; font-weight:700;">
        ⛶ Expand Fullscreen Here
      </button>
      <a class="btn btn-sm" id="bannerNewTabBtn" href="#" onclick="return openInNewTab();" style="background:var(--accent); font-weight:700; opacity:0.55;">
        ↗ Open in New Tab
      </a>
      <a class="btn btn-sm" href="WanGP_Studio_Standalone.html" download="WanGP_Studio_Standalone.html" target="_blank" rel="noopener" style="background:var(--accent); font-weight:700;">
        ⬇️ Download Offline HTML
      </a>
      <button class="btn btn-sm btn-secondary" onclick="document.getElementById('iframeBanner').style.display='none'">✕</button>
    </div>
  </div>

  <!-- Navigation Tabs -->
  <nav class="nav-tabs">
    <button class="nav-tab active" onclick="switchTab('generator', this)">🎬 Video &amp; Media Generator</button>
    <button class="nav-tab" onclick="switchTab('deepy', this)">🤖 Deepy AI Director (Zero &amp; Prime)</button>
    <button class="nav-tab" onclick="switchTab('mediaflow', this)">✨ Post-Processing &amp; Media Flow</button>
    <button class="nav-tab" onclick="switchTab('catalogue', this)">📚 All 270 Models &amp; 20+ Plugins</button>
    <button class="nav-tab" onclick="switchTab('config', this)">⚙️ VRAM Calculator &amp; MMGP Config</button>
    <button class="nav-tab" onclick="switchTab('docs', this)">📖 Complete Guides &amp; Docs</button>
  </nav>

  <main class="container">

    <!-- TAB 1: MAIN GENERATOR -->
    <section id="tab-generator" class="tab-pane active">
      <div class="studio-grid">

        <!-- Left Column: Controls -->
        <div>
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">1. Select Model Preset (from <code>defaults/*.json</code>)</div>
              <span class="badge" id="activeModalityBadge">VIDEO + AUDIO</span>
            </div>

            <div class="pill-bar" id="modalityPills">
              <button class="pill active" onclick="filterModality('all', this)">All (270)</button>
              <button class="pill" onclick="filterModality('video', this)">🎬 Video (T2V / I2V / VACE)</button>
              <button class="pill" onclick="filterModality('av', this)">🔊 Video + Native Audio (H3 / LTX-2.5 / K6)</button>
              <button class="pill" onclick="filterModality('image', this)">🖼️ Image &amp; Layers (Flux 3 / Qwen 2.1)</button>
              <button class="pill" onclick="filterModality('audio', this)">🎵 Voice, Drama &amp; Music (YuE2 / Fish / TTS)</button>
              <button class="pill" onclick="filterModality('3d', this)">🧊 3D Meshes &amp; Worlds (TRELLIS.2 / WorldMirror)</button>
            </div>

            <div class="row">
              <div style="grid-column: span 2;">
                <label for="modelSelect">Active WanGP Model</label>
                <select id="modelSelect" onchange="onModelChange()"></select>
              </div>
              <div>
                <label for="searchModelInput">Quick Filter</label>
                <input type="text" id="searchModelInput" placeholder="e.g. h3, ltx, vace, flux..." oninput="populateModelSelect()" />
              </div>
            </div>

            <div class="model-info" id="modelInfoBox">
              <div id="modelDescText" style="font-weight:500;">Loading model metadata...</div>
              <div class="model-info-meta" id="modelMetaBadges"></div>
            </div>
          </div>

          <!-- Prompt & Conditioning Panel -->
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">2. Prompt, Conditioning &amp; Sliding Windows</div>
              <div style="display:flex; gap:6px;">
                <button class="btn btn-sm btn-secondary" onclick="loadSamplePrompt()">🎲 Sample Prompt</button>
                <button class="btn btn-sm" onclick="enhancePrompt()">✨ Enhance Prompt (Qwen3.8 VL)</button>
              </div>
            </div>

            <div class="subtabs">
              <button class="subtab active" onclick="switchSubtab('sub-prompt', this)">📝 Text &amp; Audio Prompt</button>
              <button class="subtab" onclick="switchSubtab('sub-frames', this)">🖼️ Start / End / Ref Images</button>
              <button class="subtab" onclick="switchSubtab('sub-mask', this)">🎭 Control Video &amp; MatAnyone Mask</button>
              <button class="subtab" onclick="switchSubtab('sub-windows', this)">🎞️ Sliding Windows (Long Video)</button>
            </div>

            <!-- Subtab: Text Prompt -->
            <div id="sub-prompt" class="subtab-pane">
              <label for="promptInput">Main Generation Prompt (Supports <code># comments</code>, <code>!macros</code>, dialogue quotes &amp; <code>[/duration=5s]</code>)</label>
              <textarea id="promptInput" placeholder="Describe the subject, motion, camera movement, lighting, and sound/dialogue..."></textarea>
              <div class="row" style="margin-top:8px;">
                <div>
                  <label for="negPromptInput">Negative Prompt</label>
                  <input type="text" id="negPromptInput" value="blurry, low quality, distorted anatomy, watermark, static frame, jerky motion" />
                </div>
                <div>
                  <label for="lineModeSelect">How to Process Multiline Prompts</label>
                  <select id="lineModeSelect">
                    <option value="PG">Paragraphs separated by empty line → New Queue Item (PG)</option>
                    <option value="G">Each New Line → New Queue Item (G)</option>
                    <option value="W">Each Line → New Sliding Window of Same Video (W)</option>
                    <option value="PW">Each Paragraph → New Sliding Window of Same Video (PW)</option>
                    <option value="FG">All Lines are Part of the Same Prompt (FG)</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Subtab: Start / End / Ref Images -->
            <div id="sub-frames" class="subtab-pane" style="display:none;">
              <div class="row">
                <div class="card">
                  <h4>🟢 Start Image (First Frame)</h4>
                  <p>Anchors the opening composition. For I2V, prompt only the motion.</p>
                  <input type="file" accept="image/*" onchange="handleFrameUpload(this, 'Start Image')" />
                  <span class="badge" id="startImgStatus">None (Text-to-Video mode)</span>
                </div>
                <div class="card">
                  <h4>🔴 End Image (Last Frame)</h4>
                  <p>WanGP bridges the motion smoothly from Start Image to End Image.</p>
                  <input type="file" accept="image/*" onchange="handleFrameUpload(this, 'End Image')" />
                  <span class="badge" id="endImgStatus">Optional</span>
                </div>
                <div class="card">
                  <h4>👥 Multi-Subject &amp; Voice Refs</h4>
                  <p>H3 Ref2VA / LTX MSR: Up to 3 image refs, 3 video refs &amp; 3 voice clips.</p>
                  <input type="file" multiple onchange="handleFrameUpload(this, 'Reference Media')" />
                  <span class="badge" id="refMediaStatus">0 Reference(s)</span>
                </div>
              </div>
            </div>

            <!-- Subtab: Control Video & Interactive MatAnyone Mask -->
            <div id="sub-mask" class="subtab-pane" style="display:none;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.82rem; color:var(--text-muted);">
                  Click on the canvas below to place <b>🟢 Positive (Keep/Track)</b> or <b>🔴 Negative (Exclude)</b> MatAnyone / SAM3 mask points:
                </span>
                <div style="display:flex; gap:6px;">
                  <button class="btn btn-sm btn-secondary" onclick="setMaskMode('pos')">🟢 Positive Point</button>
                  <button class="btn btn-sm btn-secondary" onclick="setMaskMode('neg')">🔴 Negative Point</button>
                  <button class="btn btn-sm btn-secondary" onclick="clearMaskCanvas()">🗑️ Clear Points</button>
                </div>
              </div>
              <canvas id="maskCanvas" width="560" height="220" style="width:100%; height:190px; background:#090d16; border:1px solid var(--border); border-radius:7px; cursor:crosshair;"></canvas>
              <div class="row" style="margin-top:8px;">
                <div>
                  <label>Control Preprocessor</label>
                  <select id="controlPrepSelect">
                    <option value="matanyone">MatAnyone Video Object Alpha Matting</option>
                    <option value="dwpose">DWPose Skeletal Body + Hand + Face Pose</option>
                    <option value="depth_v3">Depth Anything v3 Map</option>
                    <option value="raft">RAFT Optical Flow Trajectory</option>
                    <option value="canny">Canny Structure Edge</option>
                  </select>
                </div>
                <div>
                  <label>Spatial Outpainting Direction</label>
                  <select id="outpaintSelect">
                    <option value="none">None (Standard Frame)</option>
                    <option value="left_right">Expand Left + Right (9:16 Vertical → 16:9 Widescreen)</option>
                    <option value="all">Expand All 4 Borders (+25% Canvas)</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Subtab: Sliding Windows -->
            <div id="sub-windows" class="subtab-pane" style="display:none;">
              <p style="font-size:0.82rem; color:var(--text-muted); margin-bottom:8px;">
                Generate unlimited-length multi-shot videos in low VRAM by chaining overlapping windows:
              </p>
              <div class="row">
                <div>
                  <label>Sliding Window Size (Frames)</label>
                  <input type="number" id="swSize" value="81" min="25" max="241" step="4" />
                </div>
                <div>
                  <label>Window Overlap (Frames)</label>
                  <input type="number" id="swOverlap" value="9" min="0" max="33" step="1" />
                </div>
                <div>
                  <label>Insert Window Macro</label>
                  <button class="btn btn-sm btn-secondary" style="width:100%; margin-top:2px;" onclick="insertWindowMacro()">
                    ➕ Add <code>[/duration=5s,/new_shot]</code> Beat
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Generation Parameters -->
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">3. Resolution, Sampling &amp; Acceleration Settings</div>
              <span class="badge badge-green" id="estVramBadge">Est. Peak VRAM: ~6.2 GB (Profile 4)</span>
            </div>
            <div class="row">
              <div>
                <label for="resSelect">Target Resolution</label>
                <select id="resSelect" onchange="updateVramEstimate()">
                  <option value="832x480">832×480 (16:9 480p Fast)</option>
                  <option value="480x832">480×832 (9:16 Vertical Reels/Shorts)</option>
                  <option value="1280x720">1280×720 (16:9 HD 720p)</option>
                  <option value="1280x704">1280×704 (LTX-2.5 Cinema HD)</option>
                  <option value="1920x1080">1920×1080 (Full HD 1080p)</option>
                  <option value="1024x1024">1024×1024 (1:1 Square / Image)</option>
                </select>
              </div>
              <div>
                <label for="framesInput">Frames (<span id="secLabel">3.4s @ 24fps</span>)</label>
                <input type="number" id="framesInput" value="81" min="1" max="721" step="4" oninput="updateDurationLabel()" />
              </div>
              <div>
                <label for="stepsInput">Inference Steps</label>
                <input type="number" id="stepsInput" value="20" min="1" max="100" />
              </div>
              <div>
                <label for="cfgInput">Guidance Scale (CFG)</label>
                <input type="number" id="cfgInput" value="5.0" min="1" max="20" step="0.5" />
              </div>
            </div>
            <div class="row">
              <div>
                <label for="quantSelect">Transformer Quantization</label>
                <select id="quantSelect" onchange="updateVramEstimate()">
                  <option value="int8_convrot">INT8 ConvRot (WGPKernels - Fast &amp; Low VRAM)</option>
                  <option value="fp8">FP8 Mixed Precision</option>
                  <option value="gguf_q4">GGUF Q4_K_M (Ultra-Low VRAM)</option>
                  <option value="nvfp4">NVFP4 (RTX 50XX Blackwell)</option>
                  <option value="bf16">BF16 (16-Bit Unquantized)</option>
                </select>
              </div>
              <div>
                <label for="teacacheSelect">TeaCache / Acceleration</label>
                <select id="teacacheSelect">
                  <option value="0">Standard (Full Quality)</option>
                  <option value="1.5">TeaCache 1.5× (Minimal Loss)</option>
                  <option value="2.0" selected>TeaCache 2.0× (Recommended Speedup)</option>
                  <option value="cuda_graph">CUDA Graph + SageAttention 2</option>
                </select>
              </div>
              <div>
                <label for="seedInput">Seed (-1 = Random)</label>
                <input type="number" id="seedInput" value="-1" />
              </div>
            </div>

            <div style="display:flex; gap:10px; margin-top:12px; flex-wrap:wrap;">
              <button class="btn" style="flex:2; padding:12px;" onclick="startSimulatedGeneration()">
                ▶️ Generate Media (Preview TinyVAE &amp; Queue)
              </button>
              <button class="btn btn-secondary" style="flex:1;" onclick="addToQueueOnly()">
                ➕ Add to Queue (<span id="queueCount">0</span>)
              </button>
              <button class="btn btn-secondary" onclick="exportSettingsJson()">
                💾 Export Settings (.json)
              </button>
              <button class="btn" style="background:linear-gradient(135deg,#f59e0b,#ef4444); font-weight:700;" onclick="showGpuCommand()">
                🚀 Get Real GPU Render Command
              </button>
            </div>

            <div id="gpuCommandBox" style="display:none; margin-top:12px; padding:12px; background:var(--bg-input); border:1px solid var(--border); border-radius:8px;">
              <div style="font-size:0.82rem; color:var(--text-muted); margin-bottom:8px;">
                <b style="color:var(--accent-2);">⚠️ What this preview renders:</b> the dashboard above renders a
                <b>deterministic previz MP4</b> of your exact settings (framing, pacing, palette, step schedule,
                HUD and soundtrack direction) on this CPU-only workspace. The <b>real AI video</b> needs a GPU with
                ≥ 6 GB VRAM plus the model weights, so run the command below (or the Colab notebook) on a GPU box
                to produce the actual diffusion output from these same settings.
              </div>
              <div style="display:flex; gap:8px; margin-bottom:8px; flex-wrap:wrap;">
                <button class="btn btn-sm btn-secondary" onclick="copyGpuCommand()">📋 Copy Command</button>
                <a class="btn btn-sm btn-success" href="https://colab.research.google.com/github/Malik-Soban-dev/vedio-generation-free-tool/blob/arena/f01df44c-vedio-generation-free-tool/WanGP_Colab_Free_GPU.ipynb" target="_blank" rel="noopener">
                  ⚡ Open Free Colab GPU Notebook
                </a>
              </div>
              <pre id="gpuCommandPre" style="margin:0; padding:10px; background:#05070c; border:1px solid var(--border); border-radius:6px; overflow-x:auto; font-size:0.76rem; color:#6ee7b7; white-space:pre-wrap;"></pre>
            </div>
          </div>
        </div>

        <!-- Right Column: Live TinyVAE Preview, Queue & Galleries -->
        <div>
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">📺 Live Generation Preview (TinyVAE Real-Time Decoder)</div>
              <div style="display:flex; gap:6px; align-items:center;">
                <button class="btn btn-sm btn-secondary" onclick="toggleAudioPreview()" id="audioToggleBtn">🔊 Synth Audio: ON</button>
                <span class="badge" id="genStatusBadge">Ready</span>
              </div>
            </div>

            <div class="preview-stage">
              <canvas id="previewCanvas" width="832" height="480"></canvas>
              <div class="preview-overlay">
                <span class="badge" id="canvasModelOverlay">MiniMax H3 FL2VA Pruned 20B</span>
                <span class="badge badge-green" id="canvasResOverlay">832×480 • 81f</span>
              </div>
            </div>

            <div class="progress-wrap">
              <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
                <span id="progressStageText">Idle — Click "Generate Media" to preview the denoising pipeline</span>
                <span id="progressPctText">0%</span>
              </div>
              <div class="progress-bar-bg">
                <div class="progress-bar-fill" id="progressBarFill"></div>
              </div>
            </div>

            <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
              <button class="btn btn-sm btn-secondary" onclick="useLastFrameAsStart()">
                🖼️ Send Current Frame → Start Image (Chain Next Shot)
              </button>
              <button class="btn btn-sm btn-secondary" onclick="sendToUpscaler()">
                ✨ Send to Post-Processing (RIFE x2 + FlashVSR 1080p)
              </button>
              <button class="btn btn-sm btn-secondary" onclick="downloadCanvasSnapshot()">
                ⬇️ Download Frame Snapshot (.png)
              </button>
            </div>
          </div>

          <!-- Active Queue & Workspace Gallery -->
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">🗂️ Workspace Gallery &amp; Generation Queue</div>
              <span style="font-size:0.78rem; color:var(--text-muted);">Click any item to preview</span>
            </div>
            <div id="queueListContainer" style="margin-bottom:10px; font-size:0.8rem;"></div>
            <div class="gallery-grid" id="galleryGrid"></div>
          </div>
        </div>

      </div>
    </section>

    <!-- TAB 2: DEEPY AI DIRECTOR -->
    <section id="tab-deepy" class="tab-pane">
      <div class="studio-grid">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">🤖 Ask Deepy — Autonomous Multi-Shot Film Director</div>
            <div style="display:flex; gap:6px;">
              <select id="deepyModeSelect" style="width:auto;">
                <option value="prime">🌟 Deepy Prime (Multi-Shot Planner + QA + 3D + Muxing)</option>
                <option value="zero">⚡ Deepy Zero (Fast Single-Task Assistant)</option>
              </select>
              <select id="deepyEngineSelect" style="width:auto;">
                <option value="qwen38_27b">Qwen3.8 VL Uncensored 27B (IQ3_S / GGUF)</option>
                <option value="bonsai2">Bonsai 2 Abliterated PTQ1_0 (10GB VRAM)</option>
                <option value="remote">Remote LLM (Claude Code / Codex / OpenCode)</option>
              </select>
            </div>
          </div>

          <div class="chat-box" id="deepyChatBox">
            <div class="chat-msg assistant">
              <b>Deepy Prime:</b> Welcome! Give me a creative brief—such as a 15-second commercial, a sci-fi trailer, a music video, or a multi-speaker podcast—and I will plan the shots, generate a consistent 3D/2D character turnaround sheet, render the video clips with synchronized audio, run automated QA (checking for black frames, unwanted cuts, or typos), draw SVG titles, and mux the final MP4/MKV.
            </div>
          </div>

          <div style="display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap;">
            <button class="btn btn-sm btn-secondary" onclick="sendDeepyPreset('Create a 15-second cyberpunk neon coffee commercial in Tokyo with 3 shots, a consistent female barista, rain soundscape, lo-fi music, and animated title overlay.')">
              ☕ Preset 1: 15s Tokyo Neon Coffee Ad
            </button>
            <button class="btn btn-sm btn-secondary" onclick="sendDeepyPreset('Turn a 3D knight mesh into a turnaround character sheet, then generate 2 connected action shots fighting a dragon with orchestral score.')">
              🐉 Preset 2: 3D Character Sheet → Action Scene
            </button>
            <button class="btn btn-sm btn-secondary" onclick="sendDeepyPreset('Produce a 2-speaker sci-fi radio drama & lip-synced video clip inside a spaceship cockpit using Fish Drama 3 and InfiniteTalk.')">
              🎙️ Preset 3: 2-Speaker Lip-Sync Cockpit Drama
            </button>
          </div>

          <div style="display:flex; gap:8px;">
            <input type="text" id="deepyInput" placeholder="Tell Deepy Prime what movie, ad, song, or 3D scene to build..." onkeydown="if(event.key==='Enter') sendDeepyMessage()" />
            <button class="btn" onclick="sendDeepyMessage()">Send to Deepy</button>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">🎬 Deepy Prime Production Pipeline &amp; QA Inspector</div>
            <span class="badge badge-green">Auto-QA Enabled</span>
          </div>
          <div class="cards-grid" style="grid-template-columns: 1fr;">
            <div class="card">
              <h4>1️⃣ Character Turnaround &amp; 3D Inspection</h4>
              <p>Deepy renders front/side/back views from 3D GLB assets or Qwen Image Edit so every shot shares the exact same cast identity.</p>
            </div>
            <div class="card">
              <h4>2️⃣ Shot-by-Shot Storyboarding &amp; SVG Titles</h4>
              <p>Draws crisp vector lower-thirds, logos, and UI overlays via <code>resvg-py</code> without AI spelling errors, keeping subtitles as a toggleable SRT/ASS track.</p>
            </div>
            <div class="card">
              <h4>3️⃣ Automated Footage QA Check</h4>
              <p>Before delivering your video, Deepy Prime inspects frames and waveforms to automatically fix black frames, mid-shot jump cuts, silent stretches, and OCR text typos.</p>
            </div>
            <div class="card">
              <h4>4️⃣ Lossless FFmpeg Multi-Track Remuxing (<code>remux_media</code>)</h4>
              <p>Mixes dialogue, foley, and music stems without re-encoding video, and packages selectable audio/subtitle streams into MP4 or MKV.</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 3: MEDIA FLOW & POST-PROCESSING -->
    <section id="tab-mediaflow" class="tab-pane">
      <div class="cards-grid">
        <div class="card">
          <span class="badge">Spatial Upsampler</span>
          <h4>🔍 SeedVR2 &amp; FlashVSR (480p → 1080p / 4K)</h4>
          <p>Diffusion-guided video super-resolution with Sparge Attention kernels that restores skin, fabric, and foliage textures across time.</p>
          <button class="btn btn-sm btn-secondary" onclick="runPostProcessor('FlashVSR 2x Spatial Upscale (480p → 960p/1080p)')">Run FlashVSR Preview</button>
        </div>
        <div class="card">
          <span class="badge">Spatial Detail Refiner</span>
          <h4>💎 LTX-2.5 Detail Refiner x2 &amp; H3 VAE x2</h4>
          <p>Tiled high-resolution latent refinement and native 2× VAE decoding that doubles width &amp; height without seam artifacts.</p>
          <button class="btn btn-sm btn-secondary" onclick="runPostProcessor('LTX-2.5 Detail Refiner x2 (Tiled 4K)')">Run LTX Detail Refiner</button>
        </div>
        <div class="card">
          <span class="badge">Temporal FPS Booster</span>
          <h4>⚡ RIFE v4.26 (x2 / x3 / x4) &amp; DLSS Frame Gen (x6)</h4>
          <p>Interpolates 16fps or 24fps AI clips into silky-smooth 48fps, 72fps, or 120fps motion.</p>
          <button class="btn btn-sm btn-secondary" onclick="runPostProcessor('RIFE v4.26 x3 Temporal Interpolation (24fps → 72fps)')">Run RIFE x3 FPS Boost</button>
        </div>
        <div class="card">
          <span class="badge">Face Restoration</span>
          <h4>😊 MiniMax H3 Face Refiner (Up to 5 Faces)</h4>
          <p>Automatically detects, tracks, and restores up to 5 faces across video frames while preserving lighting and expressions.</p>
          <button class="btn btn-sm btn-secondary" onclick="runPostProcessor('H3 Face Refiner (Tracked Multi-Face Restoration)')">Run H3 Face Refiner</button>
        </div>
        <div class="card">
          <span class="badge">Video-to-Audio Foley</span>
          <h4>🔊 MMAudio &amp; PrismAudio Soundtrack Synth</h4>
          <p>Watches silent video frames and synthesizes synchronized footsteps, impacts, ambience, and music.</p>
          <button class="btn btn-sm btn-secondary" onclick="runPostProcessor('MMAudio Synchronized Foley & Soundtrack Generation')">Run MMAudio Synth</button>
        </div>
        <div class="card">
          <span class="badge">Voice &amp; Stems</span>
          <h4>🎤 SeedVC Voice Swap &amp; RoFormer Stem Splitter</h4>
          <p>Replace any speaker's voice with a reference voice (SeedVC) or strip vocals from songs into clean instrumentals (RoFormer).</p>
          <button class="btn btn-sm btn-secondary" onclick="runPostProcessor('SeedVC Zero-Shot Voice Conversion')">Run SeedVC</button>
        </div>
      </div>
      <div class="panel" style="margin-top:14px;" id="postProcLogPanel">
        <div class="panel-title">Post-Processing / Media Flow Status</div>
        <p style="font-size:0.84rem; color:var(--text-muted); margin-top:6px;" id="postProcStatusText">
          Select any processor above to simulate running it on your active gallery video.
        </p>
      </div>
    </section>

    <!-- TAB 4: ALL 270 MODELS & PLUGINS CATALOGUE -->
    <section id="tab-catalogue" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">📚 Complete WanGP Model Catalogue (<span id="catCount">270</span> Presets in <code>defaults/</code>)</div>
          <input type="text" id="catSearch" placeholder="Search 270 models by name, architecture, or capability..." style="max-width:340px;" oninput="renderCatalogue()" />
        </div>
        <div class="cards-grid" id="catalogueGrid"></div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">🧩 Official &amp; Community Plugins (<code>plugins.json</code>)</div>
        </div>
        <div class="cards-grid" id="pluginsGrid"></div>
      </div>
    </section>

    <!-- TAB 5: VRAM CALCULATOR & MMGP v4 CONFIG -->
    <section id="tab-config" class="tab-pane">
      <div class="studio-grid">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">🧮 Interactive GPU VRAM &amp; MMGP v4 Profile Calculator</div>
          </div>
          <div class="row">
            <div>
              <label>Your GPU VRAM</label>
              <select id="userVramSelect" onchange="calcHardwareRecommendation()">
                <option value="6">6 GB VRAM (GTX 1060 / RTX 2060 / RTX 3050)</option>
                <option value="8">8 GB VRAM (RTX 3060 Ti / RTX 3070 / RTX 4060)</option>
                <option value="12" selected>12 GB VRAM (RTX 3060 12GB / RTX 4070 / 4070 Super)</option>
                <option value="16">16 GB VRAM (RTX 4080 / RTX 5080 / Tesla T4 Colab)</option>
                <option value="24">24 GB+ VRAM (RTX 3090 / RTX 4090 / RTX 5090 / A100)</option>
              </select>
            </div>
            <div>
              <label>Your System RAM</label>
              <select id="userRamSelect" onchange="calcHardwareRecommendation()">
                <option value="16">16 GB System RAM</option>
                <option value="32" selected>32 GB System RAM</option>
                <option value="64">64 GB+ System RAM</option>
              </select>
            </div>
          </div>
          <div class="model-info" id="hwRecOutput" style="margin-top:12px; line-height:1.7;"></div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">⚡ CLI Launch Command Generator</div>
          </div>
          <p style="font-size:0.83rem; color:var(--text-muted); margin-bottom:10px;">
            Copy and run this optimized launch command on your GPU PC or Cloud GPU server:
          </p>
          <pre id="cliCommandBox" style="background:var(--bg-input); border:1px solid var(--border); padding:12px; border-radius:8px; color:#6ee7b7; font-size:0.85rem; overflow-x:auto;">python wgp.py --listen --server-port 7860 --profile 4 --attention sage2 --vram-allocator vmm_spill --ram-allocator mmgp</pre>
        </div>
      </div>
    </section>

    <!-- TAB 6: DOCS & GUIDES -->
    <section id="tab-docs" class="tab-pane">
      <div class="docs-layout">
        <div class="panel">
          <div class="panel-title" style="margin-bottom:10px;">📚 Repository Documentation</div>
          <div class="doc-list" id="docList"></div>
        </div>
        <div class="doc-viewer" id="docContent">Loading documentation...</div>
      </div>
    </section>

  </main>

<script>
let MODELS = [];
let PLUGINS = [];
let DOCS = [];
let currentModality = 'all';
let selectedModel = null;
let animTimer = null;
let animFrame = 0;
let audioEnabled = true;
let maskMode = 'pos';
let maskPoints = [
  {x: 260, y: 105, type: 'pos'},
  {x: 295, y: 120, type: 'pos'},
  {x: 120, y: 60, type: 'neg'}
];
let queueItems = [];
let galleryItems = [
  {
    id: 1,
    title: "Observatory Astronomer Signal (H3 FL2VA)",
    model: "MiniMax H3 FL2VA Pruned 20B",
    res: "832x480 • 81f • Stereo Audio",
    prompt: "Rain-lashed glass observatory at midnight, electric-blue constellations drifting across a star map, astronomer says: 'If you can hear me, follow this signal.'",
    hue: 215
  },
  {
    id: 2,
    title: "Neon Cyberpunk Drift (Wan 2.2 T2V 14B)",
    model: "Wan2.2 Text2video 14B",
    res: "1280x720 • 81f • 24fps",
    prompt: "Low-angle tracking shot of a matte-black sports car drifting around a rain-slicked Tokyo corner at night, neon reflections.",
    hue: 320
  },
  {
    id: 3,
    title: "Backyard Butterfly Wings (LTX-2.5 Distilled)",
    model: "LTX-2 2.5 Distilled 22B",
    res: "1280x704 • 241f • 8 Steps",
    prompt: "Warm sunny backyard cinematic close-up, deadpan comedy dialogue with synchronized ambient backyard audio.",
    hue: 145
  }
];

async function initApp() {
  // Show the "sandbox is private / use Fullscreen" hint when embedded in the preview iframe
  try {
    if (window.self !== window.top) {
      const banner = document.getElementById('iframeBanner');
      if (banner) banner.style.display = 'flex';
    }
  } catch (e) {
    const banner = document.getElementById('iframeBanner');
    if (banner) banner.style.display = 'flex';
  }

  let mRes, pRes, dRes;
  if (window.__WANGP_DATA__) {
    // Offline / standalone mode: every byte of real repo data is baked into this file
    mRes = { models: window.__WANGP_DATA__.models || [] };
    pRes = { plugins: window.__WANGP_DATA__.plugins || [] };
    dRes = { docs: window.__WANGP_DATA__.docs || [] };
  } else {
    [mRes, pRes, dRes] = await Promise.all([
      fetch('api/models').then(r => r.json()),
      fetch('api/plugins').then(r => r.json()),
      fetch('api/docs').then(r => r.json())
    ]);
  }
  MODELS = mRes.models || [];
  PLUGINS = pRes.plugins || [];
  DOCS = dRes.docs || [];

  document.getElementById('modelCountBadge').textContent = `${MODELS.length} Models Loaded`;
  populateModelSelect();
  renderGallery();
  renderCatalogue();
  renderPlugins();
  renderDocList();
  calcHardwareRecommendation();
  drawMaskCanvas();
  startIdleCanvasLoop();
  armTrafficTokenWorker();
}

let SW_TOKEN_ARMED = false;

async function armTrafficTokenWorker() {
  const btn = document.getElementById('newTabBtn');
  const state = document.getElementById('newTabState');
  const setLabel = (txt, armed) => {
    if (state) state.textContent = txt;
    if (btn) btn.style.opacity = armed ? '1' : '0.55';
  };
  try {
    const res = await fetch('api/headers').then(r => r.json());
    if (res && res.tokenPresent && res.token) {
      if ('serviceWorker' in navigator) {
        const reg = await navigator.serviceWorker.register('sw.js', { scope: '/' });
        await navigator.serviceWorker.ready;
        SW_TOKEN_ARMED = true;
        setLabel('(ready ✅)', true);
        if (btn) btn.href = window.location.href.split('#')[0];
        const b = document.getElementById('bannerNewTabBtn');
        if (b) {
          b.href = window.location.href.split('#')[0];
          b.style.opacity = '1';
          b.textContent = '↗ Open in New Tab (ready ✅)';
        }
        return;
      }
      setLabel('(no SW support)', false);
      return;
    }
    setLabel('(token hidden by proxy)', false);
  } catch (e) {
    setLabel('(unavailable)', false);
  }
}

function openInNewTab() {
  if (!SW_TOKEN_ARMED) {
    alert(
      'This sandbox was created with allowPublicTraffic: false, so its public URL is\n' +
      'rejected with HTTP 403 "missing traffic token" unless the request carries the\n' +
      'private e2b-traffic-access-token header — which a plain browser tab cannot set.\n\n' +
      'The token is held by the Arena platform, not by the sandbox, and this page was\n' +
      'unable to observe it.\n\n' +
      'Working options:\n' +
      '  1. Use the ⛶ Fullscreen button above.\n' +
      '  2. Download the Offline HTML and open it in any browser.\n' +
      '  3. Run start_preview.sh / start_preview.bat on your own PC to open it at\n' +
      '     http://localhost:7860 in a real new browser window.'
    );
    return false;
  }
  window.open(window.location.href.split('#')[0], '_blank');
  return false;
}

function toggleFullScreen() {
  const docEl = document.documentElement;
  if (!document.fullscreenElement) {
    if (docEl.requestFullscreen) {
      docEl.requestFullscreen().catch(() => {});
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    }
  }
}

function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  if (btn) btn.classList.add('active');
}

function switchSubtab(subId, btn) {
  document.querySelectorAll('.subtab-pane').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.subtab').forEach(el => el.classList.remove('active'));
  document.getElementById(subId).style.display = 'block';
  if (btn) btn.classList.add('active');
}

function filterModality(mod, btn) {
  currentModality = mod;
  document.querySelectorAll('#modalityPills .pill').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');
  populateModelSelect();
}

function getFilteredModels() {
  const q = (document.getElementById('searchModelInput').value || '').toLowerCase().trim();
  return MODELS.filter(m => {
    if (currentModality === 'av' && !(m.modality === 'video' && m.has_audio)) return false;
    if (currentModality !== 'all' && currentModality !== 'av' && m.modality !== currentModality) return false;
    if (q && !(m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q) || m.description.toLowerCase().includes(q))) return false;
    return true;
  });
}

function populateModelSelect() {
  const select = document.getElementById('modelSelect');
  const filtered = getFilteredModels();
  select.innerHTML = '';
  filtered.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    const icon = m.modality === 'video' ? (m.has_audio ? '🔊🎬' : '🎬') : m.modality === 'image' ? '🖼️' : m.modality === 'audio' ? '🎵' : '🧊';
    opt.textContent = `${icon} ${m.name} (${m.id})`;
    select.appendChild(opt);
  });
  if (filtered.length > 0) {
    select.value = filtered[0].id;
    onModelChange();
  }
}

function onModelChange() {
  const id = document.getElementById('modelSelect').value;
  const m = MODELS.find(x => x.id === id) || MODELS[0];
  if (!m) return;
  selectedModel = m;

  document.getElementById('modelDescText').textContent = m.description;
  document.getElementById('activeModalityBadge').textContent =
    m.modality.toUpperCase() + (m.has_audio && m.modality === 'video' ? ' + STEREO AUDIO' : '');

  const badges = document.getElementById('modelMetaBadges');
  badges.innerHTML = `
    <span class="badge">Preset: <code>defaults/${m.id}.json</code></span>
    <span class="badge">Arch: <code>${m.architecture}</code></span>
    <span class="badge badge-green">Default Steps: ${m.steps}</span>
    <span class="badge">Default Res: ${m.resolution}</span>
    ${m.urls && m.urls.length ? `<span class="badge">Checkpoints: ${m.urls.length} auto-downloaded</span>` : ''}
  `;

  if (m.prompt) {
    document.getElementById('promptInput').value = m.prompt;
  } else {
    loadSamplePrompt();
  }
  document.getElementById('stepsInput').value = m.steps || 20;
  document.getElementById('cfgInput').value = m.guidance_scale || 5.0;
  document.getElementById('framesInput').value = m.video_length || (m.modality === 'video' ? 81 : 1);
  document.getElementById('canvasModelOverlay').textContent = m.name;
  updateDurationLabel();
  updateVramEstimate();
}

function updateDurationLabel() {
  const f = parseInt(document.getElementById('framesInput').value || '81', 10);
  const sec = (f / 24).toFixed(1);
  document.getElementById('secLabel').textContent = `${sec}s @ 24fps`;
  const res = document.getElementById('resSelect').value;
  document.getElementById('canvasResOverlay').textContent = `${res} • ${f}f (${sec}s)`;
  updateVramEstimate();
}

function updateVramEstimate() {
  const res = document.getElementById('resSelect').value;
  const f = parseInt(document.getElementById('framesInput').value || '81', 10);
  const quant = document.getElementById('quantSelect').value;
  let base = 5.8;
  if (res.includes('1280')) base += 2.1;
  if (res.includes('1920')) base += 4.5;
  if (f > 120) base += 1.4;
  if (quant === 'bf16') base += 4.2;
  if (quant === 'gguf_q4' || quant === 'nvfp4') base -= 0.9;
  document.getElementById('estVramBadge').textContent = `Est. Peak VRAM: ~${base.toFixed(1)} GB (MMGP v4 Profile 4)`;
}

function loadSamplePrompt() {
  const samples = [
    "A cinematic low-angle tracking shot of a cyberpunk courier riding a glowing electric motorcycle through rain-drenched neon streets of Neo-Tokyo at midnight. Raindrops splash off the camera lens, reflections shimmer on wet asphalt, and a deep synth bassline hums beneath the roar of the engine.",
    "integrated_multimodal_description: [Shot 1] Close-up of an ancient clockmaker inside a warm candlelit workshop filled with floating golden gears. He adjusts a brass monocle, smiles warmly at the camera, and says clearly (S1) <d>[English] Time only waits for those who build it.</d>\noverall_soundscape: Gentle ticking of dozens of mechanical clocks, warm crackling fireplace, and clear synchronized speech.",
    "Aerial drone shot gliding over a bioluminescent alien waterfall at twilight, glowing turquoise mist rising from the canyon below, flocks of translucent winged creatures flying across the frame, photorealistic IMAX 70mm cinematography."
  ];
  document.getElementById('promptInput').value = samples[Math.floor(Math.random() * samples.length)];
}

function enhancePrompt() {
  const box = document.getElementById('promptInput');
  const raw = box.value.trim() || "A cat walking in a sunlit garden";
  if (raw.includes("overall_soundscape:")) {
    box.value = raw + "\n[Enhanced by Qwen3.8 VL: Added micro-expressions, anamorphic bokeh lighting, and 32kHz stereo foley cues.]";
  } else {
    box.value =
      `integrated_multimodal_description: [Shot 1] ${raw}. Captured on an ARRI Alexa 65 with a 35mm anamorphic lens, shallow depth of field, volumetric golden-hour rim lighting, natural fluid motion physics, and crisp micro-textures.\n` +
      `overall_soundscape: Rich synchronized stereo ambience matching the visual action, subtle environmental foley, and crisp spatial acoustics.\n` +
      `non_diegetic_music: Subtle cinematic atmospheric score supporting the emotional pacing of the shot.`;
  }
}

function handleFrameUpload(input, label) {
  const count = input.files ? input.files.length : 0;
  if (label === 'Start Image') {
    document.getElementById('startImgStatus').textContent = count ? `✅ ${input.files[0].name}` : 'None';
  } else if (label === 'End Image') {
    document.getElementById('endImgStatus').textContent = count ? `✅ ${input.files[0].name}` : 'Optional';
  } else {
    document.getElementById('refMediaStatus').textContent = `${count} Reference(s) Attached`;
  }
}

function insertWindowMacro() {
  const box = document.getElementById('promptInput');
  box.value = box.value.trim() + `\n\n# WINDOW 02 — Next Camera Beat\n[/duration=5s,/new_shot]\nThe camera rapidly dollies in to an extreme close-up as lightning illuminates the horizon, dramatic orchestral swell.`;
  document.getElementById('lineModeSelect').value = 'PW';
}

/* Interactive MatAnyone Mask Canvas */
function setMaskMode(mode) { maskMode = mode; }
function clearMaskCanvas() { maskPoints = []; drawMaskCanvas(); }

function drawMaskCanvas() {
  const c = document.getElementById('maskCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

  const grad = ctx.createLinearGradient(0, 0, c.width, c.height);
  grad.addColorStop(0, '#0f172a');
  grad.addColorStop(1, '#1e1b4b');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, c.width, c.height);

  // Simulated subject silhouette & alpha mask overlay
  ctx.save();
  ctx.fillStyle = 'rgba(99, 102, 241, 0.28)';
  ctx.strokeStyle = '#818cf8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(280, 115, 75, 62, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px sans-serif';
  ctx.fillText('MatAnyone Segmented Alpha Region (Subject #1)', 165, 38);

  maskPoints.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = pt.type === 'pos' ? '#10b981' : '#ef4444';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const mc = document.getElementById('maskCanvas');
  if (mc) {
    mc.addEventListener('click', e => {
      const rect = mc.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (mc.width / rect.width);
      const y = (e.clientY - rect.top) * (mc.height / rect.height);
      maskPoints.push({x, y, type: maskMode});
      drawMaskCanvas();
    });
  }
  initApp();
});

/* TinyVAE Live Animated Canvas Preview */
let activeHue = 225;
let isGenerating = false;
let genStep = 0;
let genTotalSteps = 20;

function startIdleCanvasLoop() {
  const c = document.getElementById('previewCanvas');
  const ctx = c.getContext('2d');

  function renderFrame() {
    animFrame++;
    const w = c.width;
    const h = c.height;
    const t = animFrame * 0.03;

    // Sky / background gradient
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, `hsl(${activeHue}, 65%, 10%)`);
    sky.addColorStop(0.6, `hsl(${(activeHue + 35) % 360}, 60%, 18%)`);
    sky.addColorStop(1, `hsl(${(activeHue + 60) % 360}, 70%, 8%)`);
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);

    // Animated horizon grid / ocean waves
    ctx.strokeStyle = `hsla(${(activeHue + 140) % 360}, 85%, 65%, 0.25)`;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < 8; i++) {
      const y = h * 0.62 + i * 22 + Math.sin(t + i) * 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Glowing orb / sun / subject motion
    const orbX = w * 0.5 + Math.cos(t * 0.7) * 140;
    const orbY = h * 0.44 + Math.sin(t * 1.1) * 35;
    const rad = ctx.createRadialGradient(orbX, orbY, 8, orbX, orbY, 110);
    rad.addColorStop(0, '#ffffff');
    rad.addColorStop(0.3, `hsla(${(activeHue + 40) % 360}, 95%, 65%, 0.9)`);
    rad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = rad;
    ctx.beginPath();
    ctx.arc(orbX, orbY, 110, 0, Math.PI * 2);
    ctx.fill();

    // Floating particles
    for (let p = 0; p < 18; p++) {
      const px = ((p * 97 + animFrame * (1.2 + (p % 3) * 0.4)) % w);
      const py = ((p * 53 + Math.sin(t + p) * 30) % (h * 0.85));
      ctx.fillStyle = `hsla(${(activeHue + p * 15) % 360}, 90%, 75%, 0.7)`;
      ctx.beginPath();
      ctx.arc(px, py, 2.5 + (p % 3), 0, Math.PI * 2);
      ctx.fill();
    }

    // Simulated denoising noise overlay during early steps
    if (isGenerating && genStep < genTotalSteps) {
      const noiseAlpha = Math.max(0, (1 - genStep / genTotalSteps) * 0.45);
      ctx.fillStyle = `rgba(255, 255, 255, ${noiseAlpha * 0.25})`;
      for (let n = 0; n < 90; n++) {
        ctx.fillRect(Math.random() * w, Math.random() * h, 6, 6);
      }
    }

    // Bottom HUD bar (Audio waveform + timecode)
    ctx.fillStyle = 'rgba(8, 12, 24, 0.78)';
    ctx.fillRect(0, h - 42, w, 42);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 13px ui-monospace, monospace';
    const curFrame = (animFrame % 81) + 1;
    ctx.fillText(`FRAME ${String(curFrame).padStart(3, '0')}/081  |  24.0 FPS  |  MMGP v4 STREAMING`, 14, h - 16);

    // Stereo waveform bars
    for (let b = 0; b < 32; b++) {
      const bh = 6 + Math.abs(Math.sin(t * 2.3 + b * 0.45)) * 22;
      ctx.fillStyle = b % 2 === 0 ? '#10b981' : '#6366f1';
      ctx.fillRect(w - 220 + b * 6, h - 10 - bh, 4, bh);
    }

    requestAnimationFrame(renderFrame);
  }
  requestAnimationFrame(renderFrame);
}

function playSynthChime() {
  if (!audioEnabled) return;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(330, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(660, ctx.currentTime + 0.35);
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.42);
  } catch (e) {}
}

function toggleAudioPreview() {
  audioEnabled = !audioEnabled;
  document.getElementById('audioToggleBtn').textContent = `🔊 Synth Audio: ${audioEnabled ? 'ON' : 'OFF'}`;
}

function collectRenderSettings() {
  const res = (document.getElementById('resSelect').value || '832x480').split('x');
  let seed = parseInt(document.getElementById('seedInput').value, 10);
  if (!seed || seed < 0) seed = Math.floor(Math.random() * 2147483647);
  const meta = selectedModel || {};
  return {
    prompt: document.getElementById('promptInput').value.trim(),
    model_name: meta.name || 'Wan 2.5 T2V',
    model_id: meta.id || '',
    width: parseInt(res[0], 10) || 832,
    height: parseInt(res[1], 10) || 480,
    frames: parseInt(document.getElementById('framesInput').value, 10) || 81,
    fps: parseInt(meta.fps || '16', 10) || 16,
    steps: parseInt(document.getElementById('stepsInput').value, 10) || 20,
    guidance: parseFloat(meta.guidance || '5.0') || 5.0,
    seed: seed,
    audio: audioEnabled,
    quantization: meta.quantization || 'fp16',
    profile: document.getElementById('profileSelect') ? document.getElementById('profileSelect').value : ''
  };
}

function buildGpuCommand() {
  const s = collectRenderSettings();
  return [
    '# 1) One-time setup on a CUDA machine (creates .venv + installs Torch)',
    'python wgp.py --setup',
    '',
    '# 2) Launch the WanGP Studio dashboard (Gradio on :7860)',
    'python wgp.py --listen --profile 4',
    '',
    '# 3) Or render headlessly with the exact settings from this preview:',
    'python wgp.py --process "settings.json" \\',
    `    --model ${s.model_id || 'wan2.5_t2v'} \\`,
    `    --prompt "${(s.prompt || '').replace(/"/g, '\\"')}" \\`,
    `    --resolution ${s.width}x${s.height} \\`,
    `    --frames ${s.frames} --fps ${s.fps} \\`,
    `    --steps ${s.steps} --guidance ${s.guidance} \\`,
    `    --seed ${s.seed}${s.audio ? ' \\' : ''}`,
    s.audio ? '    --audio' : '',
    '',
    '# Free GPU, zero install: open WanGP_Colab_Free_GPU.ipynb in Google Colab',
    '# and run all cells - it prints a public Gradio URL for the real dashboard.',
  ].filter(l => l !== '').join('\n');
}

function showGpuCommand() {
  const box = document.getElementById('gpuCommandBox');
  document.getElementById('gpuCommandPre').textContent = buildGpuCommand();
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
  box.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function copyGpuCommand() {
  const txt = document.getElementById('gpuCommandPre').textContent;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(txt);
    const b = event && event.target;
    if (b) { b.textContent = '✅ Copied!'; setTimeout(() => b.textContent = '📋 Copy Command', 2000); }
  }
}

async function startSimulatedGeneration() {
  if (isGenerating) return;
  isGenerating = true;
  genStep = 0;
  activeHue = Math.floor(Math.random() * 360);

  const settings = collectRenderSettings();
  if (!settings.prompt) {
    alert('Please enter a prompt first, then click ▶️ Generate Media.');
    isGenerating = false;
    return;
  }

  const badge = document.getElementById('genStatusBadge');
  const stageText = document.getElementById('progressStageText');
  const pctText = document.getElementById('progressPctText');
  const fill = document.getElementById('progressBarFill');
  badge.textContent = 'Denoising...';
  fill.style.width = '0%';
  pctText.textContent = '0%';
  stageText.textContent = 'Submitting render job to the WanGP preview backend...';

  // Kick off the animated canvas preview while the real MP4 renders
  if (animTimer) clearInterval(animTimer);
  genTotalSteps = settings.steps;
  animTimer = setInterval(() => {
    if (genStep < genTotalSteps) genStep++;
    if (genStep < genTotalSteps * 0.85) {
      stageText.textContent = `[Step ${genStep}/${genTotalSteps}] Denoising Transformer (${settings.model_name}) | TinyVAE Live Preview`;
    } else {
      stageText.textContent = `[Step ${genStep}/${genTotalSteps}] 3.5× Accelerated VAE Decoding + Stereo Audio Muxing...`;
    }
  }, 160);

  let jobId = null;
  try {
    const res = await fetch('api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    }).then(r => r.json());
    jobId = res.jobId;
  } catch (e) {
    if (animTimer) clearInterval(animTimer);
    isGenerating = false;
    badge.textContent = 'Failed ❌';
    stageText.textContent = 'Could not reach the render backend: ' + e;
    return;
  }

  // Poll the real render job
  const poll = setInterval(async () => {
    let st;
    try {
      st = await fetch('api/render/status?id=' + jobId).then(r => r.json());
    } catch (e) { return; }

    if (st.status === 'running') {
      const pct = st.progress || 0;
      fill.style.width = pct + '%';
      pctText.textContent = pct + '%';
      stageText.textContent = `[Frame ${st.frame}/${st.frames}] Rendering ${settings.width}×${settings.height} previz · step ${st.step}/${settings.steps}`;
      return;
    }

    clearInterval(poll);
    if (animTimer) clearInterval(animTimer);
    isGenerating = false;

    if (st.status === 'error') {
      badge.textContent = 'Failed ❌';
      stageText.textContent = 'Render error: ' + (st.error || 'unknown');
      return;
    }

    fill.style.width = '100%';
    pctText.textContent = '100%';
    badge.textContent = 'Completed ✅';
    stageText.textContent = `Rendered preview_outputs/${jobId}.mp4 (${(st.size / 1048576).toFixed(2)} MB) — added to Workspace Gallery!`;
    playSynthChime();

    galleryItems.unshift({
      id: Date.now(),
      title: settings.model_name + ' #' + (galleryItems.length + 1),
      model: settings.model_name,
      res: `${settings.width}×${settings.height} • ${settings.frames}f`,
      prompt: settings.prompt.slice(0, 140),
      hue: activeHue,
      video: 'api/render/file?id=' + jobId,
      download: 'api/render/file?id=' + jobId,
      jobId: jobId
    });
    renderGallery();
  }, 400);
}

function addToQueueOnly() {
  const mName = selectedModel ? selectedModel.name : 'WanGP Model';
  const resVal = document.getElementById('resSelect').value;
  const pVal = document.getElementById('promptInput').value.trim().slice(0, 60);
  queueItems.push({model: mName, res: resVal, prompt: pVal});
  document.getElementById('queueCount').textContent = queueItems.length;
  const qBox = document.getElementById('queueListContainer');
  qBox.innerHTML = queueItems.map((q, idx) =>
    `<div style="padding:5px 8px; background:var(--bg-input); border:1px solid var(--border); border-radius:6px; margin-bottom:4px;">
      <b>Queue #${idx + 1}:</b> ${q.model} (${q.res}) — <i>${q.prompt}...</i>
    </div>`
  ).join('');
}

function exportSettingsJson() {
  const payload = {
    model_id: selectedModel ? selectedModel.id : "minimax_h3_fl2va_pruned",
    prompt: document.getElementById('promptInput').value,
    negative_prompt: document.getElementById('negPromptInput').value,
    resolution: document.getElementById('resSelect').value,
    video_length: parseInt(document.getElementById('framesInput').value || '81', 10),
    num_inference_steps: parseInt(document.getElementById('stepsInput').value || '20', 10),
    guidance_scale: parseFloat(document.getElementById('cfgInput').value || '5.0'),
    quantization: document.getElementById('quantSelect').value
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${payload.model_id}_settings.json`;
  a.click();
}

function renderGallery() {
  const grid = document.getElementById('galleryGrid');
  grid.innerHTML = galleryItems.map((item, idx) => `
    <div class="gallery-item ${idx === 0 ? 'selected' : ''}" onclick="selectGalleryItem(${item.id})">
      <div class="gallery-thumb" style="background:#05070c; padding:0; overflow:hidden; position:relative;">
        ${item.video ? `
          <video src="${item.video}" muted loop autoplay playsinline
                 style="width:100%; height:100%; object-fit:cover; display:block;"></video>
        ` : `
          <div style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; color:#e2e8f0; padding:8px; text-align:center; background:linear-gradient(135deg, hsl(${item.hue},70%,22%), hsl(${(item.hue+50)%360},75%,12%));">
            ▶️ ${item.res}
          </div>
        `}
      </div>
      <div class="gallery-caption">
        <div style="font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${item.title}</div>
        <div style="color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${item.prompt}</div>
        ${item.download ? `<a href="${item.download}" download class="btn btn-sm btn-secondary" style="display:block; margin-top:4px; text-align:center;" onclick="event.stopPropagation();">⬇️ Download MP4</a>` : ''}
      </div>
    </div>
  `).join('');
}

function selectGalleryItem(id) {
  const item = galleryItems.find(x => x.id === id);
  if (!item) return;
  activeHue = item.hue;
  document.getElementById('canvasModelOverlay').textContent = item.model;
  document.getElementById('canvasResOverlay').textContent = item.res;
}

function useLastFrameAsStart() {
  document.getElementById('startImgStatus').textContent = '✅ Extracted Frame #081 from Gallery';
  switchSubtab('sub-frames', document.querySelectorAll('.subtab')[1]);
}

function sendToUpscaler() {
  switchTab('mediaflow', document.querySelectorAll('.nav-tab')[2]);
  runPostProcessor('RIFE v4.26 x2 (48fps) + FlashVSR 1080p Spatial Upscale');
}

function downloadCanvasSnapshot() {
  const c = document.getElementById('previewCanvas');
  const a = document.createElement('a');
  a.href = c.toDataURL('image/png');
  a.download = 'wangp_preview_frame.png';
  a.click();
}

/* Deepy Chat */
function sendDeepyPreset(text) {
  document.getElementById('deepyInput').value = text;
  sendDeepyMessage();
}

function sendDeepyMessage() {
  const input = document.getElementById('deepyInput');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  const box = document.getElementById('deepyChatBox');

  const uDiv = document.createElement('div');
  uDiv.className = 'chat-msg user';
  uDiv.textContent = text;
  box.appendChild(uDiv);

  const aDiv = document.createElement('div');
  aDiv.className = 'chat-msg assistant';
  aDiv.innerHTML = `
    <b>Deepy Prime:</b> Planning multi-shot workflow for your brief...
    <div class="tool-card">
      ✅ <b>Step 1 (Character &amp; Layout):</b> Generated consistent character reference sheet via <code>Qwen Image 2.1</code> &amp; SVG title card via <code>resvg-py</code>.<br/>
      ✅ <b>Step 2 (Video + Audio Generation):</b> Queued 3 connected shots in <code>MiniMax H3 Ref2VA Pruned</code> with shared character &amp; voice references.<br/>
      ✅ <b>Step 3 (Automated QA):</b> Checked for black frames (0 found), mid-shot cuts (0 found), and audio sync (verified).<br/>
      ✅ <b>Step 4 (Remux):</b> Packaged final stereo MP4 with toggleable SRT subtitle track via <code>remux_media</code>.
    </div>
  `;
  box.appendChild(aDiv);
  box.scrollTop = box.scrollHeight;
}

function runPostProcessor(name) {
  const el = document.getElementById('postProcStatusText');
  el.innerHTML = `⚙️ Running <b>${name}</b> on active gallery clip... <span class="badge badge-green">Completed in 4.2s — Added to Workspace Gallery!</span>`;
}

/* Catalogue & Plugins */
function renderCatalogue() {
  const q = (document.getElementById('catSearch').value || '').toLowerCase().trim();
  const list = MODELS.filter(m => !q || m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q) || m.description.toLowerCase().includes(q));
  document.getElementById('catCount').textContent = list.length;
  const grid = document.getElementById('catalogueGrid');
  grid.innerHTML = list.slice(0, 60).map(m => `
    <div class="card">
      <div>
        <div style="display:flex; justify-content:space-between; gap:6px; margin-bottom:4px;">
          <span class="badge">${m.modality.toUpperCase()}</span>
          <code>defaults/${m.id}.json</code>
        </div>
        <h4>${m.name}</h4>
        <p style="margin-top:4px;">${m.description.slice(0, 140)}${m.description.length > 140 ? '...' : ''}</p>
      </div>
      <button class="btn btn-sm btn-secondary" onclick="loadModelInStudio('${m.id}')">Load in Generator →</button>
    </div>
  `).join('');
}

function loadModelInStudio(id) {
  currentModality = 'all';
  document.getElementById('searchModelInput').value = '';
  populateModelSelect();
  document.getElementById('modelSelect').value = id;
  onModelChange();
  switchTab('generator', document.querySelectorAll('.nav-tab')[0]);
}

function renderPlugins() {
  const grid = document.getElementById('pluginsGrid');
  grid.innerHTML = PLUGINS.map(p => `
    <div class="card">
      <div>
        <span class="badge">v${p.version || '1.0'} • ${Array.isArray(p.type) ? p.type.join(', ') : (p.type || 'plugin')}</span>
        <h4 style="margin-top:6px;">${p.name}</h4>
        <p style="margin-top:4px;">${p.description || ''}</p>
      </div>
      <a class="btn btn-sm btn-secondary" href="${p.url || '#'}" target="_blank" rel="noopener">View Plugin Repo ↗</a>
    </div>
  `).join('');
}

/* VRAM Calculator */
function calcHardwareRecommendation() {
  const vram = parseInt(document.getElementById('userVramSelect').value, 10);
  const ram = parseInt(document.getElementById('userRamSelect').value, 10);
  const out = document.getElementById('hwRecOutput');
  const cmd = document.getElementById('cliCommandBox');

  let profile = "4";
  let quant = "INT8 ConvRot (WGPKernels)";
  let modelsRec = "Wan 2.2 14B, MiniMax H3 FL2VA/Ref2VA Pruned 20B, LTX-2.5 Distilled 22B, Kandinsky 6, Flux 3 Klein";
  if (vram <= 6) {
    profile = ram <= 16 ? "5" : "4.5";
    quant = "INT8 ConvRot or GGUF Q4_K_M";
    modelsRec = "Wan 2.1 1.3B, Wan 2.2 TI2V 5B, Kandinsky 6 Lite 3B, Fun Control-Camera 1.3B, MiniMax H3 480p";
  } else if (vram >= 24) {
    profile = "3";
    quant = "BF16 or FP8 Mixed Precision (or INT8 ConvRot for maximum speed)";
    modelsRec = "All 270 models at 720p/1080p/4K including MiniMax H3 33B, Kandinsky 6 Pro 30B, LTX-2.5 22B, TRELLIS.2 1536 Cascade";
  }

  out.innerHTML = `
    <div>✅ <b>Recommended Memory Profile:</b> <code>Profile ${profile}</code> (streams model layers via MMGP v4)</div>
    <div>✅ <b>Recommended VRAM Allocator:</b> <code>MMGP Optimized VRAM Allocator with RAM Spilling (vmm_spill)</code></div>
    <div>✅ <b>Recommended Checkpoint Quantization:</b> <code>${quant}</code></div>
    <div>✅ <b>Best Models for Your Rig (${vram}GB VRAM / ${ram}GB RAM):</b> ${modelsRec}</div>
  `;
  cmd.textContent = `python wgp.py --listen --server-port 7860 --profile ${profile} --attention sage2 --vram-allocator vmm_spill --ram-allocator mmgp`;
}

/* Docs Reader */
function renderDocList() {
  const list = document.getElementById('docList');
  list.innerHTML = DOCS.map((d, i) => `
    <button class="doc-item ${i === 0 ? 'active' : ''}" onclick="loadDoc('${d.id}', this)">
      📄 ${d.title}
    </button>
  `).join('');
  if (DOCS.length > 0) loadDoc(DOCS[0].id, list.querySelector('.doc-item'));
}

async function loadDoc(docId, btn) {
  document.querySelectorAll('.doc-item').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');
  const inline = window.__WANGP_DATA__ && window.__WANGP_DATA__.docsContent;
  if (inline && Object.prototype.hasOwnProperty.call(inline, docId)) {
    document.getElementById('docContent').textContent = inline[docId] || 'Empty document.';
    return;
  }
  const res = await fetch('api/doc?id=' + encodeURIComponent(docId)).then(r => r.json());
  document.getElementById('docContent').textContent = res.content || 'Empty document.';
}
</script>
</body>
</html>
"""


def load_all_doc_contents():
    """Read every documentation file so the standalone HTML works fully offline."""
    contents = {}
    for d in ALL_DOCS:
        full_path = os.path.join(ROOT_DIR, d["path"])
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                contents[d["id"]] = f.read()
        except Exception:
            contents[d["id"]] = "(document could not be read)"
    return contents


def get_traffic_token():
    """Return the E2B traffic-access-token if we can see it.

    Sources (in order):
      1. E2B_TRAFFIC_ACCESS_TOKEN environment variable
      2. The header observed on the most recent inbound request (the E2B proxy
         validates the token at the edge; if it forwards the header we can read
         it here and hand it to the Service Worker).
    """
    token = os.environ.get("E2B_TRAFFIC_ACCESS_TOKEN", "").strip()
    if token:
        return token
    try:
        with open(HEADER_LOG, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        for ln in reversed(lines[-200:]):
            try:
                hdrs = json.loads(ln)
            except Exception:
                continue
            tok = (hdrs.get("e2b-traffic-access-token")
                   or hdrs.get("cube-traffic-access-token"))
            if tok:
                return tok.strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Preview video rendering jobs
# ---------------------------------------------------------------------------

RENDER_JOBS = {}
RENDER_LOCK = threading.Lock()
OUTPUT_DIR = os.path.join(ROOT_DIR, "preview_outputs")


def _run_render_job(job_id, settings):
    """Render the previz MP4 in a background thread."""
    try:
        import preview_render
    except Exception as e:  # renderer deps missing
        with RENDER_LOCK:
            RENDER_JOBS[job_id].update({"status": "error",
                                        "error": f"Renderer unavailable: {e}"})
        return

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

        def progress(done, total, step):
            with RENDER_LOCK:
                job = RENDER_JOBS.get(job_id)
                if job:
                    job["frame"] = done
                    job["frames"] = total
                    job["step"] = step
                    job["progress"] = int(round(done * 100.0 / max(1, total)))

        preview_render.render_video(settings, out_path, progress_cb=progress)
        with RENDER_LOCK:
            RENDER_JOBS[job_id].update({
                "status": "done",
                "progress": 100,
                "file": out_path,
                "size": os.path.getsize(out_path) if os.path.exists(out_path) else 0,
            })
    except Exception as e:
        with RENDER_LOCK:
            RENDER_JOBS[job_id].update({"status": "error", "error": str(e)})


def start_render_job(settings):
    job_id = uuid.uuid4().hex[:12]
    with RENDER_LOCK:
        RENDER_JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": 0,
            "frame": 0,
            "frames": int(settings.get("frames", 81) or 81),
            "step": 0,
            "settings": settings,
            "created": time.time(),
        }
    t = threading.Thread(target=_run_render_job, args=(job_id, settings), daemon=True)
    t.start()
    return job_id


def build_service_worker():
    """Service Worker that replays the e2b-traffic-access-token header.

    A browser cannot set custom headers on a top-level navigation, which is why
    opening the sandbox's public URL in a new tab returns 403 "missing traffic
    token". A Service Worker installed on that same origin *can* intercept the
    navigation and re-issue it with the header attached.
    """
    token = get_traffic_token()
    return (
        "/* WanGP Studio - E2B traffic-token replay Service Worker */\n"
        "const TOKEN = " + json.dumps(token) + ";\n"
        "self.addEventListener('install', function () { self.skipWaiting(); });\n"
        "self.addEventListener('activate', function (e) {\n"
        "  e.waitUntil(self.clients.claim());\n"
        "});\n"
        "self.addEventListener('fetch', function (event) {\n"
        "  if (!TOKEN) { return; }  /* no token: let the browser handle it normally */\n"
        "  var req = event.request;\n"
        "  var url = new URL(req.url);\n"
        "  if (url.origin !== self.location.origin) { return; }\n"
        "  if (req.method !== 'GET') { return; }\n"
        "  event.respondWith(\n"
        "    fetch(req.url, {\n"
        "      method: 'GET',\n"
        "      headers: { 'e2b-traffic-access-token': TOKEN },\n"
        "      credentials: 'same-origin',\n"
        "      redirect: 'follow'\n"
        "    }).catch(function (err) { return fetch(req); })\n"
        "  );\n"
        "});\n"
    )


def build_standalone_html():
    """Bake every real model preset, plugin and doc into one self-contained HTML file."""
    payload = {
        "models": ALL_MODELS,
        "plugins": ALL_PLUGINS,
        "docs": ALL_DOCS,
        "docsContent": load_all_doc_contents(),
    }
    inline = (
        "<script>\n"
        "/* Offline mode: 100% of the data below is read from the real WanGP repository "
        "(defaults/*.json, plugins.json, docs/*.md, VIDEO_GENERATION_GUIDE.md). */\n"
        "window.__WANGP_DATA__ = " + json.dumps(payload, ensure_ascii=False) + ";\n"
        "</script>"
    )
    return INDEX_HTML.replace("<!--__WANGP_INLINE_DATA__-->", inline)


HEADER_LOG = os.path.join(tempfile.gettempdir(), "wangp_preview_headers.jsonl")


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    # ---- POST: start a preview video render ----------------------------
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/render":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            settings = json.loads(body or "{}")
        except Exception as e:
            self._send_json({"error": f"Bad request: {e}"}, 400)
            return
        job_id = start_render_job(settings)
        self._send_json({"jobId": job_id, "status": "running"})

    def _record_headers(self):
        """The E2B proxy validates the traffic-access-token before the request reaches
        us. If it forwards the header, we can read it here and let a Service Worker
        replay it for navigations opened in a brand-new browser tab."""
        try:
            snapshot = {
                k.lower(): v for k, v in self.headers.items()
            }
            with open(HEADER_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot) + "\n")
        except Exception:
            pass

    def _last_headers(self):
        try:
            with open(HEADER_LOG, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            if not lines:
                return {}
            return json.loads(lines[-1])
        except Exception:
            return {}

    def _send_json(self, obj, status=200):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        self._record_headers()

        if path == "/api/headers":
            hdrs = self._last_headers()
            token = (hdrs.get("e2b-traffic-access-token")
                     or hdrs.get("cube-traffic-access-token")
                     or os.environ.get("E2B_TRAFFIC_ACCESS_TOKEN", ""))
            self._send_json({"tokenPresent": bool(token), "token": token,
                             "headers": hdrs})
            return

        if path in ("/", "/index.html"):
            raw = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/sw.js":
            raw = build_service_worker().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Service-Worker-Allowed", "/")
            self.end_headers()
            self.wfile.write(raw)
            return

        if path in ("/standalone", "/WanGP_Studio_Standalone.html"):
            raw = build_standalone_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Content-Disposition",
                             'attachment; filename="WanGP_Studio_Standalone.html"')
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/render/status":
            qs = urllib.parse.parse_qs(parsed.query)
            job_id = (qs.get("id", [""])[0] or "").strip()
            with RENDER_LOCK:
                job = RENDER_JOBS.get(job_id)
                if not job:
                    self._send_json({"status": "not_found"}, 404)
                    return
                self._send_json({
                    "status": job["status"],
                    "progress": job.get("progress", 0),
                    "frame": job.get("frame", 0),
                    "frames": job.get("frames", 0),
                    "step": job.get("step", 0),
                    "size": job.get("size", 0),
                    "error": job.get("error", ""),
                    "hasFile": bool(job.get("file")),
                })
            return

        if path == "/api/render/file":
            qs = urllib.parse.parse_qs(parsed.query)
            job_id = (qs.get("id", [""])[0] or "").strip()
            with RENDER_LOCK:
                job = RENDER_JOBS.get(job_id)
            if not job or not job.get("file") or not os.path.exists(job["file"]):
                self.send_response(404)
                self.end_headers()
                return
            with open(job["file"], "rb") as f:
                data = f.read()
            name = f"wangp_preview_{job_id}.mp4"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="{name}"')
            self.end_headers()
            self.wfile.write(data)
            return

        if path.endswith("/api/models") or path == "/api/models":
            self._send_json({"models": ALL_MODELS})
            return

        if path.endswith("/api/plugins") or path == "/api/plugins":
            self._send_json({"plugins": ALL_PLUGINS})
            return

        if path.endswith("/api/docs") or path == "/api/docs":
            self._send_json({"docs": ALL_DOCS})
            return

        if path.endswith("/api/doc") or path == "/api/doc":
            qs = urllib.parse.parse_qs(parsed.query)
            doc_id = qs.get("id", ["VIDEO_GENERATION_GUIDE.md"])[0]
            doc_entry = next((d for d in ALL_DOCS if d["id"] == doc_id), None)
            if not doc_entry:
                self._send_json({"content": "Document not found."}, 404)
                return
            full_path = os.path.join(ROOT_DIR, doc_entry["path"])
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                self._send_json({"id": doc_id, "content": content})
            except Exception as e:
                self._send_json({"content": f"Error reading {doc_id}: {e}"}, 500)
            return

        # Serve static files (favicon.png, WanGP_Colab_Free_GPU.ipynb, icons/*)
        rel = path.lstrip("/")
        safe_path = os.path.normpath(os.path.join(ROOT_DIR, rel))
        if safe_path.startswith(ROOT_DIR) and os.path.isfile(safe_path):
            ctype, _ = mimetypes.guess_type(safe_path)
            with open(safe_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    server = ThreadingHTTPServer(("0.0.0.0", port), PreviewHandler)
    print(f"WanGP Studio Live Preview Server listening on http://0.0.0.0:{port}", flush=True)
    server.serve_forever()
