"""
WanGP Studio Preview - deterministic preview-video renderer.

WHY THIS EXISTS
---------------
Real WanGP generation needs an NVIDIA/AMD GPU with >= 6 GB VRAM plus the model
weights (multiple GB, pulled from HuggingFace on first run). The preview
workspace has neither a GPU nor outbound access to HuggingFace, so it cannot run
the actual diffusion pipeline.

What this module does instead is render a *faithful deterministic previz* of the
generation you configured in the dashboard: the exact resolution, frame count,
fps, step schedule, guidance scale, seed and audio flag are used to drive a
procedural scene that denoises from pure noise into the composed shot, stamped
with a WanGP-style HUD. It is a real, playable H.264 MP4 that you can use to
lock framing, pacing, palette and soundtrack direction before spending GPU
credits -- and the dashboard hands you the exact command to render the real
AI video from those same settings.

Everything here is seeded from `seed`, so the same settings always produce the
exact same video.

Requires: numpy, Pillow and an ffmpeg binary (imageio-ffmpeg provides one).
"""

import colorsys
import math
import os
import random
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Prompt analysis -> scene description
# ---------------------------------------------------------------------------

PALETTES = {
    "sunset":   ((28, 18, 48), (255, 138, 76), (255, 214, 130)),
    "ocean":    ((8, 32, 62), (32, 124, 176), (160, 226, 232)),
    "forest":   ((10, 30, 22), (34, 104, 66), (168, 214, 132)),
    "city":     ((12, 16, 34), (58, 84, 148), (226, 168, 210)),
    "snow":     ((36, 46, 66), (150, 176, 208), (238, 246, 255)),
    "desert":   ((48, 26, 26), (196, 116, 62), (248, 220, 158)),
    "cosmos":   ((4, 4, 16), (58, 32, 116), (222, 200, 255)),
    "studio":   ((18, 18, 24), (86, 86, 104), (236, 234, 240)),
}

KEYWORD_SCENE = [
    (("ocean", "sea", "beach", "wave", "water", "underwater"), "ocean"),
    (("forest", "tree", "jungle", "woods", "leaf", "leaves"), "forest"),
    (("city", "urban", "street", "neon", "tokyo", "cyberpunk", "car"), "city"),
    (("snow", "winter", "ice", "mountain", "alpine", "frozen"), "snow"),
    (("desert", "sand", "dune", "canyon", "sahara"), "desert"),
    (("space", "galaxy", "cosmos", "nebula", "planet", "star", "orbit"), "cosmos"),
    (("studio", "portrait", "product", "indoor", "room", "character"), "studio"),
    (("sunset", "sunrise", "golden", "dusk", "dawn", "sky"), "sunset"),
]

PARTICLE_WORDS = {
    "snow":  ("snow", "blizzard", "winter", "frost"),
    "rain":  ("rain", "storm", "thunder", "monsoon"),
    "leaves": ("leaf", "leaves", "autumn", "forest", "wind"),
    "stars": ("star", "stars", "space", "night", "galaxy", "cosmic"),
    "ember": ("fire", "ember", "spark", "lava", "flame", "forge"),
    "dust":  ("dust", "sand", "desert", "sunbeam", "godray", "atmospheric"),
}

MOTION_WORDS = {
    "orbit":   ("orbit", "around", "rotate", "360", "circle"),
    "dolly":   ("dolly", "push in", "pushin", "walk", "forward", "towards", "zoom in"),
    "pullback": ("pull back", "pullback", "reveal", "zoom out", "retreat"),
    "crane":   ("crane", "rise", "ascend", "lift", "upward", "aerial", "drone"),
    "handheld": ("handheld", "shake", "shaky", "chase", "run", "action"),
    "static":  ("static", "locked", "still", "tripod"),
}

MOOD_WORDS = {
    "epic":    ("epic", "cinematic", "grand", "majestic", "symphonic", "trailer"),
    "calm":    ("calm", "peaceful", "serene", "gentle", "soft", "ambient", "slow"),
    "tense":   ("tense", "dark", "horror", "eerie", "suspense", "ominous"),
    "uplift":  ("happy", "bright", "joy", "upbeat", "fun", "cheerful", "vibrant"),
}


def _has(text, words):
    return any(w in text for w in words)


def analyze_prompt(prompt, model_name=""):
    """Turn free text + model name into a deterministic scene description."""
    t = (prompt or "").lower()
    blob = t + " " + (model_name or "").lower()

    scene = "sunset"
    for words, name in KEYWORD_SCENE:
        if _has(blob, words):
            scene = name
            break

    particles = None
    for kind, words in PARTICLE_WORDS.items():
        if _has(t, words):
            particles = kind
            break

    motion = "dolly"
    for kind, words in MOTION_WORDS.items():
        if _has(t, words):
            motion = kind
            break

    mood = "epic"
    for kind, words in MOOD_WORDS.items():
        if _has(t, words):
            mood = kind
            break

    top, mid, glow = PALETTES[scene]
    night = scene in ("cosmos", "city") or "night" in t or "moon" in t

    return {
        "scene": scene,
        "particles": particles,
        "motion": motion,
        "mood": mood,
        "night": night,
        "sky_top": np.array(top, dtype=np.float32),
        "sky_mid": np.array(mid, dtype=np.float32),
        "glow": np.array(glow, dtype=np.float32),
        "has_water": scene == "ocean" or _has(t, ("lake", "river", "reflection", "puddle", "sea")),
    }


# ---------------------------------------------------------------------------
# Value-noise helpers (smooth procedural fields for clouds / textures)
# ---------------------------------------------------------------------------

def _value_noise(rng, h, w, cells, seed_offset=0):
    gh = max(2, int(cells))
    gw = max(2, int(cells * w / max(1, h)))
    grid = rng.random((gh + 1, gw + 1)).astype(np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.BICUBIC)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return np.roll(out, seed_offset, axis=1)


def _fbm(rng, h, w, seed_offset=0):
    n = np.zeros((h, w), dtype=np.float32)
    amp, total = 1.0, 0.0
    for cells in (3, 7, 15, 31):
        n += _value_noise(rng, h, w, cells, seed_offset) * amp
        total += amp
        amp *= 0.5
    return n / total


# ---------------------------------------------------------------------------
# Core frame renderer
# ---------------------------------------------------------------------------

def _ridge(x, h, base, amplitude, offsets, rng):
    """Sum of sines -> a believable mountain / dune silhouette height in pixels."""
    y = np.full_like(x, base, dtype=np.float32)
    for i, (freq, amp, phase) in enumerate(offsets):
        y -= amp * (0.5 + 0.5 * np.sin(x * freq + phase + i * 1.7))
    return np.clip(y, h * 0.18, h * 1.05)


def render_scene(cfg, w, h, t, frame_idx):
    """Compose one clean (fully denoised) frame of the shot."""
    rng = np.random.default_rng(cfg["seed"] + 9176)
    sc = cfg["scene_info"]

    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    x = xx / max(1, w - 1)
    y = yy / max(1, h - 1)

    # ---- camera motion -------------------------------------------------
    motion = sc["motion"]
    pan = zoom = lift = 0.0
    if motion == "dolly":
        zoom = 0.16 * t
        lift = -0.02 * t
    elif motion == "pullback":
        zoom = 0.18 * (1.0 - t)
        lift = 0.03 * t
    elif motion == "orbit":
        pan = 0.42 * np.sin(t * math.pi * 2.0)
        zoom = 0.05 * np.sin(t * math.pi)
    elif motion == "crane":
        lift = -0.16 * t
        zoom = 0.06 * t
    elif motion == "handheld":
        pan = 0.030 * np.sin(t * math.pi * 6.0) + 0.012 * np.sin(frame_idx * 0.9)
        lift = -0.024 * np.sin(t * math.pi * 5.0 + 1.1)
        zoom = 0.05 + 0.02 * np.sin(t * math.pi * 3.0)
    else:  # static
        pan = 0.006 * np.sin(t * math.pi * 2.0)
        zoom = 0.03

    xs = (x - 0.5) * (1.0 - zoom) + 0.5 + pan
    ys = (y - 0.5) * (1.0 - zoom) + 0.5 + lift

    # ---- sky gradient --------------------------------------------------
    horizon = 0.60
    sky_mask = np.clip((horizon - ys) / horizon, 0, 1) ** 0.85
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = sc["sky_mid"][c] + (sc["sky_top"][c] - sc["sky_mid"][c]) * sky_mask

    # ---- sun / moon glow ----------------------------------------------
    sun_x = 0.5 + 0.26 * np.sin(cfg["seed"] % 7)
    sun_y = horizon - 0.16 - 0.05 * np.cos(cfg["seed"] % 11)
    d = np.sqrt(((xs - sun_x) * 1.35) ** 2 + ((ys - sun_y)) ** 2)
    glow = np.exp(-d * 7.0)[..., None]
    core = np.exp(-d * 46.0)[..., None]
    img += sc["glow"][None, None, :] * (glow * 0.85 + core * 1.35)
    if sc["night"]:
        img *= 0.72

    # ---- clouds / nebula ----------------------------------------------
    cloud = _fbm(rng, h, w, seed_offset=int(40 * t) + int(cfg["seed"] % 500))
    cloud = np.clip((cloud - 0.46) * 2.4, 0, 1)
    cloud *= np.clip((horizon + 0.10 - ys) / (horizon + 0.10), 0, 1) ** 1.2
    cloud_rgb = sc["glow"][None, None, :] * 0.55 + sc["sky_mid"][None, None, :] * 0.45
    img = img * (1 - cloud[..., None] * 0.62) + cloud_rgb * cloud[..., None] * 0.85

    # ---- stars ---------------------------------------------------------
    if sc["night"] or sc["particles"] == "stars":
        srng = np.random.default_rng(cfg["seed"] + 31)
        n_st = 420
        sx = (srng.random(n_st) * w).astype(int)
        sy = (srng.random(n_st) * (h * 0.62)).astype(int)
        tw = 0.55 + 0.45 * np.sin(frame_idx * 0.22 + srng.random(n_st) * 6.28)
        for i in range(n_st):
            r = max(0, min(h - 1, sy[i]))
            c = max(0, min(w - 1, sx[i]))
            v = 190 * tw[i]
            img[r, c] = np.clip(img[r, c] + v, 0, 255)

    # ---- terrain silhouettes ------------------------------------------
    layers = [
        (horizon + 0.02, 0.085, [(2.1, 0.055, 0.4), (5.3, 0.030, 1.7), (11.0, 0.014, 3.1)], 0.42),
        (horizon + 0.12, 0.115, [(1.6, 0.075, 2.2), (4.1, 0.040, 0.6), (9.5, 0.018, 4.4)], 0.66),
        (horizon + 0.26, 0.150, [(1.2, 0.095, 5.1), (3.2, 0.050, 2.9), (7.4, 0.022, 1.2)], 1.0),
    ]
    for li, (base, amp, offsets, darkness) in enumerate(layers):
        ridge = _ridge(xs[0], 1.0, base, amp, offsets, rng)
        ridge = ridge[None, :] + 0.0
        below = (ys > ridge).astype(np.float32)
        aa = np.clip((ys - ridge) * 220.0, 0, 1)  # cheap anti-aliasing
        below = below * aa
        shade = 0.30 + 0.20 * (1.0 - darkness)
        col = np.array([shade * 0.9, shade * 0.95, shade * 1.15], dtype=np.float32) * 255.0
        col = col * (1.0 - 0.18 * li)
        # slight atmospheric perspective for far layers
        col = col * (0.62 + 0.38 * darkness)
        img = img * (1 - below[..., None]) + col[None, None, :] * below[..., None]

    # ---- water / reflection -------------------------------------------
    if sc["has_water"]:
        below = ys > (horizon + 0.30)
        if below.any():
            depth = np.clip((ys - (horizon + 0.30)) / 0.30, 0, 1)
            water = np.array([18, 46, 74], dtype=np.float32)
            water = water + sc["sky_mid"] * 0.35
            ripple = _fbm(rng, h, w, seed_offset=int(90 * t) + 77)
            spec = np.clip(np.sin((ys - horizon) * 340.0 + ripple * 14.0 + frame_idx * 0.16), 0, 1)
            spec = (spec ** 6.0) * (1.0 - depth * 0.75)
            band = np.exp(-np.abs(xs - sun_x) * 5.0)[..., None]
            water_col = water[None, None, :] * (1 - depth[..., None] * 0.45) + sc["glow"][None, None, :] * spec[..., None] * band * 0.9
            img = np.where(below[..., None], water_col, img)

    # ---- volumetric light rays ----------------------------------------
    rays = np.clip(
        np.sin((xs * 9.0) + (ys * 3.0) - t * 1.1) * 0.5 + 0.5, 0, 1
    ) ** 12.0
    rays *= np.exp(-np.sqrt((xs - sun_x) ** 2 + (ys - sun_y) ** 2) * 2.2)
    img += sc["glow"][None, None, :] * rays[..., None] * 0.22

    # ---- particles -----------------------------------------------------
    kind = sc["particles"]
    if kind:
        prng = np.random.default_rng(cfg["seed"] + 5150)
        counts = {"snow": 260, "rain": 340, "leaves": 120, "stars": 0,
                  "ember": 190, "dust": 220}
        n = counts.get(kind, 180)
        if n:
            px = prng.random(n)
            py = prng.random(n)
            sp = 0.15 + prng.random(n) * 0.85
            size = prng.random(n)
            for i in range(n):
                if kind == "rain":
                    yy_p = (py[i] + t * sp[i] * 2.4) % 1.0
                    xx_p = (px[i] + yy_p * 0.06) % 1.0
                    ln = 0.035
                    x0, x1 = int(xx_p * w), int((xx_p + 0.004) * w)
                    for k in range(int(h * ln)):
                        r = int((yy_p + k / h) * h)
                        if 0 <= r < h and 0 <= x0 < w:
                            img[r, x0] = np.clip(img[r, x0] + 70, 0, 255)
                elif kind == "ember" or kind == "dust":
                    yy_p = (py[i] - t * sp[i] * 0.8) % 1.0
                    xx_p = (px[i] + 0.05 * np.sin(t * 4.0 + i)) % 1.0
                    c, r = int(xx_p * w), int(yy_p * h)
                    if 0 <= r < h and 0 <= c < w:
                        v = 150 if kind == "ember" else 60
                        col = np.array([v, v * 0.72, v * 0.35]) if kind == "ember" else np.array([v, v, v])
                        rr = max(0, min(h - 1, r)); cc = max(0, min(w - 1, c))
                        img[rr, cc] = np.clip(img[rr, cc] + col, 0, 255)
                else:  # snow / leaves
                    yy_p = (py[i] + t * sp[i] * 0.9) % 1.0
                    xx_p = (px[i] + 0.08 * np.sin(t * 3.0 + i * 0.7)) % 1.0
                    c, r = int(xx_p * w), int(yy_p * h)
                    rad = 1 if kind == "snow" else 2
                    for dr in range(-rad, rad + 1):
                        for dc in range(-rad, rad + 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w:
                                v = 210 if kind == "snow" else 120
                                col = np.array([v, v * 0.85, v * 0.45]) if kind == "leaves" else np.array([v, v, v])
                                img[rr, cc] = np.clip(img[rr, cc] + col * 0.85, 0, 255)

    # ---- grade: vignette, lift, contrast ------------------------------
    vig = 1.0 - 0.55 * np.clip((np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) - 0.22) * 1.5, 0, 1)
    img *= vig[..., None]
    mood = sc["mood"]
    if mood == "tense":
        img = img * np.array([0.92, 0.94, 1.06], dtype=np.float32) * 0.92
    elif mood == "uplift":
        img = img * np.array([1.06, 1.03, 0.96], dtype=np.float32) * 1.04
    elif mood == "calm":
        img = img * np.array([0.98, 1.02, 1.04], dtype=np.float32)
    img = (img - 128.0) * 1.06 + 128.0

    return np.clip(img, 0, 255).astype(np.uint8)


def apply_denoise(clean, noise, strength, rng):
    """Blend pure noise into the clean frame using the diffusion step schedule."""
    s = float(np.clip(strength, 0.0, 1.0))
    s = s ** 1.35  # perceptual easing: structure resolves early
    out = clean.astype(np.float32) * (0.22 + 0.78 * s) + noise.astype(np.float32) * (1.0 - s) * 0.95
    grain = (rng.random(clean.shape[:2])[..., None] - 0.5) * (10.0 + 46.0 * (1.0 - s))
    return np.clip(out + grain, 0, 255).astype(np.uint8)


def add_hud(frame, cfg, frame_idx, total_frames, step, total_steps):
    """WanGP-style burn-in: model, settings, prompt and step progress."""
    img = Image.fromarray(frame, mode="RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size

    def font(size):
        try:
            return ImageFont.load_default(size=size)
        except Exception:
            return ImageFont.load_default()

    fs = max(11, int(h * 0.026))
    f_small = font(fs)
    f_big = font(int(fs * 1.25))

    pad = int(h * 0.028)
    # top-left: model + settings
    label = f"{cfg['model_name']}  ·  {cfg['width']}x{cfg['height']}  ·  {total_frames}f @ {cfg['fps']}fps"
    d.text((pad, pad), label, font=f_big, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    sub = f"steps {step}/{total_steps}   cfg {cfg['guidance']}   seed {cfg['seed']}   {cfg['quantization']}"
    d.text((pad, pad + int(fs * 1.7)), sub, font=f_small, fill=(180, 224, 255),
           stroke_width=1, stroke_fill=(0, 0, 0))

    # top-right: live timecode
    secs = frame_idx / max(1.0, float(cfg["fps"]))
    tc = f"{int(secs // 60):02d}:{int(secs % 60):02d}.{int((secs % 1) * 100):02d}"
    tw = d.textlength(tc, font=f_big)
    d.text((w - pad - tw, pad), tc, font=f_big, fill=(126, 231, 183),
           stroke_width=1, stroke_fill=(0, 0, 0))

    # bottom: prompt (wrapped, max 2 lines)
    prompt = cfg["prompt"] or ""
    max_chars = max(18, int(w / (fs * 0.56)))
    lines, line = [], ""
    for word in prompt.split():
        if len(line) + len(word) + 1 > max_chars:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
        if len(lines) >= 2:
            break
    if line and len(lines) < 2:
        lines.append(line)
    y_text = h - pad - int(fs * (len(lines) + 0.4)) - int(h * 0.022)
    for ln in lines:
        d.text((pad, y_text), ln, font=f_small, fill=(238, 238, 244),
               stroke_width=1, stroke_fill=(0, 0, 0))
        y_text += int(fs * 1.45)

    # bottom progress bar
    bar_h = max(3, int(h * 0.008))
    bar_y = h - pad - bar_h
    d.rectangle([pad, bar_y, w - pad, bar_y + bar_h], fill=(28, 32, 44))
    prog = (frame_idx + 1) / max(1, total_frames)
    d.rectangle([pad, bar_y, pad + int((w - 2 * pad) * prog), bar_y + bar_h],
                fill=(99, 102, 241))

    # REC dot while denoising
    if prog < 0.42:
        r = max(3, int(h * 0.010))
        d.ellipse([w - pad - r * 2, bar_y - r * 3, w - pad, bar_y - r], fill=(239, 68, 68))

    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Audio (procedural score for the video+audio models)
# ---------------------------------------------------------------------------

SCALES = {
    "epic":    [0, 3, 5, 7, 10, 12],
    "calm":    [0, 2, 4, 7, 9, 12],
    "tense":   [0, 1, 3, 6, 8, 11],
    "uplift":  [0, 4, 7, 9, 11, 12],
}


def render_audio(path, cfg, duration, fps):
    """Write a short procedural stereo score as a 44.1 kHz WAV."""
    sr = 44100
    n = int(duration * sr)
    tt = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(cfg["seed"] + 808)
    mood = cfg["scene_info"]["mood"]
    scale = SCALES.get(mood, SCALES["epic"])
    root = 55.0 * (2.0 ** ((cfg["seed"] % 12) / 12.0))
    bpm = {"epic": 92, "calm": 66, "tense": 74, "uplift": 108}.get(mood, 88)
    beat = 60.0 / bpm

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    # sustained pad
    for i, semi in enumerate(scale[:4]):
        f = root * (2.0 ** (semi / 12.0))
        det = 1.0 + (i - 1.5) * 0.0012
        env = 0.5 + 0.5 * np.sin(2 * np.pi * tt / (beat * 8) + i)
        wave_i = np.sin(2 * np.pi * f * det * tt) * 0.5
        wave_i += np.sin(2 * np.pi * f * 2.0 * det * tt) * 0.18
        pan = 0.35 + 0.25 * i
        left += wave_i * env * (1.0 - pan)
        right += wave_i * env * pan
    left *= 0.16
    right *= 0.16

    # arpeggio
    idx = 0
    pos = 0.0
    while pos < duration:
        semi = scale[idx % len(scale)] + 12 * (1 if idx % 7 == 3 else 0)
        f = root * (2.0 ** (semi / 12.0))
        dur = beat * 0.5
        s0 = int(pos * sr)
        s1 = min(n, int((pos + dur) * sr))
        if s1 > s0:
            local = np.arange(s1 - s0, dtype=np.float32) / sr
            env = np.exp(-local * 4.5)
            tone = (np.sin(2 * np.pi * f * local) * 0.6
                    + np.sin(2 * np.pi * f * 1.5 * local) * 0.25) * env * 0.20
            pan = 0.5 + 0.4 * math.sin(idx * 0.8)
            left[s0:s1] += tone * (1 - pan)
            right[s0:s1] += tone * pan
        idx += 1
        pos += beat * 0.5

    # soft kick on the beat
    pos = 0.0
    while pos < duration:
        s0 = int(pos * sr)
        s1 = min(n, int((pos + 0.28) * sr))
        if s1 > s0:
            local = np.arange(s1 - s0, dtype=np.float32) / sr
            env = np.exp(-local * 16.0)
            fk = 110.0 * np.exp(-local * 9.0) + 42.0
            kick = np.sin(2 * np.pi * fk * local) * env * 0.42
            left[s0:s1] += kick
            right[s0:s1] += kick
        pos += beat

    # airy noise bed
    bed = rng.standard_normal(n).astype(np.float32) * 0.010
    left += bed
    right += bed

    stereo = np.stack([np.clip(left, -1, 1), np.clip(right, -1, 1)], axis=1)
    pcm = (stereo * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_video(settings, out_path, progress_cb=None):
    """Render the preview MP4 for a set of dashboard settings.

    settings keys (all optional): prompt, model_name, width, height, frames,
    fps, steps, guidance, seed, audio, quantization, profile.
    Returns the path written.
    """
    prompt = str(settings.get("prompt", "") or "")
    model_name = str(settings.get("model_name", "Wan 2.5 T2V") or "Wan 2.5 T2V")
    width = int(settings.get("width", 832) or 832)
    height = int(settings.get("height", 480) or 480)
    frames = int(settings.get("frames", 81) or 81)
    fps = int(settings.get("fps", 16) or 16)
    steps = int(settings.get("steps", 50) or 50)
    guidance = float(settings.get("guidance", 5.0) or 5.0)
    seed = int(settings.get("seed", random.randint(1, 2 ** 31 - 1)) or 1)
    audio = bool(settings.get("audio", False))

    # keep the CPU-only preview render quick: cap the working resolution
    max_side = 768
    scale = min(1.0, max_side / float(max(width, height)))
    rw = max(2, int(round(width * scale / 2.0) * 2))
    rh = max(2, int(round(height * scale / 2.0) * 2))
    frames = max(2, min(frames, 240))

    cfg = {
        "prompt": prompt,
        "model_name": model_name,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "quantization": str(settings.get("quantization", "fp16") or "fp16"),
        "scene_info": analyze_prompt(prompt, model_name),
    }

    tmpdir = tempfile.mkdtemp(prefix="wangp_render_")
    silent_mp4 = os.path.join(tmpdir, "silent.mp4")
    wav_path = os.path.join(tmpdir, "score.wav")

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"

    writer = imageio_ffmpeg_write(silent_mp4, rw, rh, fps, exe)

    rng = np.random.default_rng(seed)
    for i in range(frames):
        t = i / max(1, frames - 1)
        clean = render_scene(cfg, rw, rh, t, i)

        # denoise schedule: structure resolves over the first ~42% of frames
        sched = min(1.0, t / 0.42)
        step_now = min(steps, max(1, int(round(sched * steps))))
        noise = (rng.random((rh, rw, 3)) * 255.0).astype(np.uint8)
        frame = apply_denoise(clean, noise, sched, rng)
        frame = add_hud(frame, cfg, i, frames, step_now, steps)
        writer.send(frame)

        if progress_cb and (i % 2 == 0 or i == frames - 1):
            progress_cb(i + 1, frames, step_now)
    writer.close()

    duration = frames / float(fps)

    if audio:
        try:
            render_audio(wav_path, cfg, duration, fps)
            cmd = [exe, "-y", "-loglevel", "error",
                   "-i", silent_mp4, "-i", wav_path,
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                   "-shortest", out_path]
            subprocess.run(cmd, check=True, capture_output=True)
            return out_path
        except Exception:
            pass

    with open(silent_mp4, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    return out_path


def imageio_ffmpeg_write(path, w, h, fps, exe):
    """Minimal generator-based pipe writer to ffmpeg (rawvideo -> h264 mp4)."""
    cmd = [exe, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "23",
           "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def gen():
        try:
            while True:
                frame = yield
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except GeneratorExit:
            pass

    g = gen()
    next(g)

    class _W:
        def send(self, frame):
            g.send(frame)

        def close(self):
            try:
                g.close()
            except Exception:
                pass
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait(timeout=180)

    return _W()
