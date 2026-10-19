# START HERE — Make your first AI video (no tech skills needed)

## The 30-second version

You have **WanGP**: a free AI video generator. You type a sentence, it makes a video.
It is **free forever** — no limits, no credits, no subscription. You just need a
powerful graphics card (GPU), which you can borrow **for free from Google**.

---

## ⚡ Easiest path — free Google GPU (recommended)

**Step 1.** Click this link:

https://colab.research.google.com/github/Malik-Soban-dev/vedio-generation-free-tool/blob/arena/f01df44c-vedio-generation-free-tool/WanGP_Colab_Free_GPU.ipynb

**Step 2.** In the Colab menu click **Runtime → Run all**.
(If it asks "Run anyway?", click it.)

**Step 3.** Wait about **10 minutes**. You will see lots of text scrolling — that is
normal, it is downloading the AI model.

**Step 4.** At the bottom you will see a link that looks like:

```
Running on public URL: https://ab12cd34ef.gradio.live
```

**Click that link.** The WanGP studio opens in your browser.

**Step 5.** In the studio:
- Type your idea in the **Prompt** box (e.g. *"a golden retriever surfing at sunset, cinematic"*)
- Pick a model from the list if you want (default is fine)
- Click **Generate**
- Wait a few minutes → **your video appears. Right-click it → Save.**

That's it. You can make as many as you want. It's free.

> ⚠️ Colab disconnects after ~12 hours of idling. Just re-run the notebook to start again.

---

## 🖥️ Second path — your own computer (only if you have an NVIDIA graphics card)

You need an **NVIDIA GPU with 6 GB or more** of memory (GTX 1060 and up, or any RTX).
To check on Windows: `Ctrl+Shift+Esc` → Performance → GPU (top-right will say NVIDIA).

```bash
git clone https://github.com/Malik-Soban-dev/vedio-generation-free-tool.git
cd vedio-generation-free-tool
python wgp.py --setup      # one time only, ~10 min
python wgp.py              # every time after
```

Then open **http://localhost:7860** in your browser. Same studio as above.

---

## ❓ Wait — what is the "Preview" dashboard then?

The preview in this workspace (`preview_server.py`) is a **practice board**.
This workspace has **no graphics card**, so it cannot run the real AI.

What it *can* do, and does well:
- Browse all **270 AI models** and see what each one does
- Turn your idea into a **moving storyboard video** (real MP4 file) with the right
  colors, camera movement, pacing and music mood — so you can see your shot first
- Tell you the **exact command** to render the real video later
- Read the full **VIDEO_GENERATION_GUIDE.md**

Think of it as: **plan your shot here for free → render the real thing on Google's GPU.**

---

## 🆘 Things that might confuse you

| You see | What it means |
|---|---|
| **"missing traffic token" / 403** | The preview's public link is locked by the sandbox. Just use the **⛶ Fullscreen** button, or download the offline HTML. |
| **"CUDA out of memory"** | Your chosen model is too big for the GPU. In the studio, lower **Profile** from 5 to 3 or 2, or pick a smaller model. |
| **Very slow first generation** | The model is downloading (several GB). Only happens once. |
| **Colab says "disconnected"** | You idled too long. Re-run the notebook. |

---

## 📄 Full detail

- `VIDEO_GENERATION_GUIDE.md` — everything: install, every model, every setting
- `WanGP_Colab_Free_GPU.ipynb` — the one-click free GPU notebook
- `docs/` — the official WanGP documentation (26 files)
