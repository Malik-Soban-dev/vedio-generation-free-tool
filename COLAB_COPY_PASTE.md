# Colab — Copy & Paste Version (use this if the links didn't work)

## ✅ The one-click link works now (use this first)

Your repo was made **public** and the notebook was merged to `main`, so this link
is now verified working:

**https://colab.research.google.com/github/Malik-Soban-dev/vedio-generation-free-tool/blob/main/WanGP_Colab_Free_GPU.ipynb**

Open it, then: **Runtime → Change runtime type → T4 GPU → Save**, then **Runtime → Run all**.

---

## Only use the copy-paste method below if that link ever breaks

It exists because your repository used to be **private**, and Google Colab cannot
read files from a private repo — so every `colab.research.google.com/github/...`
link failed at the time.

**Good news:** WanGP itself is a **public** project (`deepbeepmeep/WanGP`), so Colab
can download it directly either way.

---

## What to do

You already have a blank notebook open (`Untitled0.ipynb`). Do this:

### Step 0 — Turn on the free GPU (do this first!)

**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**

---

### Step 1 — Paste this into the first cell, then press the ▶ button

*(or click the cell and press `Ctrl` + `Enter`)*

```python
# 1. Check the free GPU is attached  (you want to see "Tesla T4")
!nvidia-smi

# 2. Download WanGP (~2 minutes)
!git clone --depth 1 https://github.com/deepbeepmeep/WanGP.git /content/WanGP
%cd /content/WanGP

# 3. Install everything (~5 minutes - lots of scrolling text is completely normal)
!pip install -q -r requirements.txt

# 4. Safety net: reinstall the important parts if anything above failed
import importlib, subprocess, sys
for m in ["torch", "gradio", "diffusers", "transformers", "accelerate",
          "safetensors", "einops", "numpy", "imageio", "imageio_ffmpeg"]:
    try:
        importlib.import_module(m)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", m])

import torch
print("PyTorch     :", torch.__version__)
print("GPU found   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Graphics    :", torch.cuda.get_device_name(0))
    print("Video memory:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), "GB")
```

⏱️ **Wait for this to finish before doing Step 2.** You'll know it's done when the
▶ button stops spinning and `GPU found : True` is printed.

If `GPU found : False` → **Runtime → Change runtime type → T4 GPU → Save**, then run the cell again.

---

### Step 2 — Add a new cell (`+ Code`) and paste this, then press ▶

```python
import torch
%cd /content/WanGP

vram = torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0
profile = "5" if vram >= 22 else "4" if vram >= 14 else "3" if vram >= 10 else "2"
print(f"{vram:.1f} GB detected -> using Profile {profile}")
print("WAIT ~2 MINUTES, then click the  https://....gradio.live  link below:\n")

!python wgp.py --share --listen --profile {profile}
```

---

### Step 3 — Open your studio

After ~2 minutes you'll see:

```
Running on public URL: https://ab12cd34ef.gradio.live
```

👉 **Click that link.** That's the real WanGP studio.

Then:
1. Type your idea in the **Prompt** box, e.g.
   *a golden retriever surfing a wave at sunset, cinematic slow motion*
2. Leave everything else on the defaults
3. Click **Generate**
4. Wait a few minutes → your video appears → **download it**

🎉 Free and unlimited. No credits, no limits.

---

## Important

- **Leave the Colab tab open** while you use the studio. Closing it shuts the studio down.
- If Colab disconnects (it sleeps when idle), Colab wipes `/content`.
  Just run **Step 1** then **Step 2** again.

## If something errors

| What you see | Fix |
|---|---|
| `No devices were found` in `nvidia-smi` | Runtime → Change runtime type → **T4 GPU** → Save, re-run |
| `CUDA out of memory` | Re-run Step 2 — it auto-picks a lower profile. Or use Profile 3. |
| Red error while installing | Ignore it if the Safety net prints `GPU found : True`. |
| `gradio.live` link never appears | Wait a full 3 minutes, then scroll to the very bottom of the output. |

---

## Optional: make this permanent

If you'd like the normal GitHub links to work in future, **make the repo public**:
GitHub → your repo → **Settings** → **Danger Zone** → **Change visibility** → **Public**.

WanGP is open-source so there's nothing secret in it. Say the word and I'll
tidy the repo for public viewing.
