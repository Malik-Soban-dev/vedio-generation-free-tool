# Complete Guide: Installing WanGP & Generating AI Videos

This repository contains **WanGP (`deepbeepmeep/WanGP`)**, a unified open-source generative AI studio designed to run state-of-the-art **Video, Audio, Image, and 3D models** on consumer GPUs with as little as **6 GB of VRAM**.

---

## 1. System Prerequisites

Before generating videos, make sure the machine where you run WanGP meets these requirements:

| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GTX 10XX / RTX 20XX (6 GB VRAM) or AMD RDNA 2/3 | NVIDIA RTX 3060 12GB / RTX 4070+ / RTX 50XX |
| **System RAM** | 16 GB RAM | 32 GB+ RAM (helps MMGP v4 offload large models) |
| **Disk Space** | 25 GB free | 100 GB+ SSD/NVMe (checkpoints download on demand to `ckpts/`) |
| **Python** | Python 3.11.x (or 3.10.9 for GTX 10XX) | Python 3.11.14 |
| **CUDA / Driver** | CUDA 12.8 (GTX 10XX) / CUDA 13.0+ (RTX 20XX–50XX) | Latest NVIDIA R580+ driver + CUDA 13.0/13.1 |

> **Note on Cloud / CPU-Only Containers:**  
> WanGP requires a CUDA, ROCm, or Apple Silicon (MPS) GPU to run diffusion inference. If your current cloud workspace is CPU-only, clone this repository onto your local GPU PC or a GPU cloud instance (RunPod, Vast.ai, Lightning AI, Google Colab, or Kaggle) and follow the steps below.

---

## 2. Installation Instructions

### Option A: 1-Click Automated Installer (Recommended)

WanGP includes interactive scripts that automatically detect your GPU, create an isolated Python environment (`venv`, `uv`, or `conda`), and install matching PyTorch + acceleration kernels (**Triton**, **SageAttention**, **WGPKernels**, **GGUF**, **Nunchaku**).

#### On Linux / macOS:
```bash
chmod +x scripts/*.sh
./scripts/install.sh
```
1. Choose **`1. Automatic Install (1-Click, Venv, Auto-Detect GPU)`** (or `2. Custom/Manual Install` to pick `uv` or `conda`).
2. Once finished, start WanGP anytime with:
   ```bash
   ./scripts/run.sh
   ```

#### On Windows:
1. Double-click **`scripts\install.bat`**.
2. Select **`1. Automatic Install`**.
3. Once finished, double-click **`scripts\run.bat`** to start the app.

---

### Option B: Manual Installation (Conda or Venv)

#### For NVIDIA RTX 20XX, 30XX, 40XX, and 50XX:
```bash
# 1. Create and activate Python 3.11 environment
conda create -n wan2gp python=3.11.14 -y
conda activate wan2gp

# 2. Install PyTorch 2.10 with CUDA 13.0
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# 3. Install WanGP core dependencies
pip install -r requirements.txt

# 4. (Recommended for RTX 30XX-50XX on Linux) Install SageAttention 2 & WGPKernels for ~40% faster video generation
pip install https://github.com/deepbeepmeep/SageAttention/releases/download/v2.2.0-linux.cg1/sageattention-2.2.0+cu130torch2.10.0.cg1-cp311-cp311-linux_x86_64.whl
pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.1.2/wgpkernels-1.1.2%2Bcu130-py3-none-linux_x86_64.whl
```

#### For Older NVIDIA GTX 10XX GPUs:
```bash
conda create -n wan2gp python=3.10.9 -y
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

---

### Option C: Docker (Ubuntu / Debian Linux)
```bash
chmod +x run-docker-cuda-deb.sh
./run-docker-cuda-deb.sh
```

---

## 3. Launching WanGP

Run the main server from the repository root:

```bash
# Standard local launch (opens on http://localhost:7860)
python wgp.py

# Launch accessible on your local network or cloud port forwarding (binds to 0.0.0.0:7860)
python wgp.py --listen --server-port 7860

# Launch with a public Gradio share link (great for Colab / remote GPU servers)
python wgp.py --share
```

Then open **`http://localhost:7860`** in your browser.

---

## 4. Which Video Model Should You Choose?

In the top **Model Selection** dropdown of the Web UI, pick the model that matches your hardware and creative goal. **Models download automatically** from Hugging Face the first time you click **Generate**.

| Goal | Best Model(s) to Select | Min VRAM | Key Strengths |
| :--- | :--- | :--- | :--- |
| **Fastest / Low-VRAM Text-to-Video** | `Wan 2.1 T2V 1.3B` or `Wan 2.2 TI2V 5B` | **6 GB** | Great starter model; generates 480p clips quickly on entry-level GPUs. |
| **High-Quality Pure Video (T2V / I2V)** | `Wan 2.2 T2V 14B` / `Wan 2.2 I2V 14B` | **8–12 GB** | Strong motion physics, cinematic quality, and broad LoRA support. |
| **Video + Synchronized Stereo Audio** | `MiniMax H3 FL2VA Pruned (20B)` | **6–11 GB** | Creates realistic video **plus speech, SFX, and music** in one pass. |
| **Fast Cinematic Video + Native Audio** | `LTX 2.5 Distilled (22B)` | **8–12 GB** | 8-step distilled generator for 720p/1080p video with synchronized audio. |
| **Lightweight Video + Audio (3B)** | `Kandinsky 6 Lite Distilled (3B)` | **6 GB** | 10-step 3B model that outputs 5-second clips with speech, music & SFX. |
| **Reference-Guided Video + Audio** | `MiniMax H3 Ref2VA Pruned (20B)` or `LTX 2.5 MSR` | **8–12 GB** | Feed up to 3 reference images/videos/voices to keep characters & voices consistent. |
| **Camera Path Control from 1 Image** | `Fun Control-Camera 1.3B` or `Uni3C 14B` | **6–10 GB** | Upload a start image and fly the camera along a preset or custom trajectory. |
| **Character Animation / Motion Transfer** | `Wan 2.2 Animate 2` or `Viggle Animate` | **8–12 GB** | Transfer body motion and expressions from a driving video onto a target character. |
| **Masked Video Editing & Outpainting** | `Wan 2.1 VACE 14B` (or `Vace Fusionix`) | **10–12 GB** | Inpaint objects, swap faces, or expand video borders (spatial/temporal outpainting). |
| **Talking Avatar / Lip-Sync** | `InfiniteTalk` / `MultiTalk` / `LongCat Avatar 1.5` | **8–12 GB** | Animate a portrait image with 1–2 voice tracks for long lip-synced dialogue. |

---

## 5. Step-by-Step Workflows to Make Videos

### Workflow 1: Your First Text-to-Video (T2V)

1. **Select a Model**: In the top dropdown, choose **`Wan 2.1 Text2Video 1.3B`** (for a fast test) or **`MiniMax H3 FL2VA Pruned`** (for video with sound).
2. **Write Your Prompt**: Use the structure:
   ```text
   [Subject] + [Action / Motion] + [Setting & Atmosphere] + [Camera Movement & Lighting]
   ```
   **Example Prompt:**
   > *A golden retriever puppy running joyfully through shallow ocean waves at golden hour sunset, water splashing in slow motion, low-angle tracking camera shot, warm cinematic rim lighting, photorealistic, 4k detail.*
3. **(Optional) Use Prompt Enhancer**: Expand the prompt tools or enable the **Prompt Enhancer** to let WanGP automatically enrich your prompt with model-specific visual and audio descriptors.
4. **Configure Basic Settings**:
   - **Resolution**: Start with `832x480` (16:9 widescreen) or `480x832` (9:16 vertical/Reels/Shorts). Move up to `1280x720` once tested.
   - **Video Length (Frames)**:
     - `49 frames` ≈ 2 seconds (at 24 fps)
     - `81 frames` ≈ 3.3 seconds
     - `121 frames` ≈ 5 seconds
   - **Steps**:
     - Standard models (`Wan 2.1/2.2`, `H3`): `20–30 steps`
     - Distilled / Accelerator models (`LTX 2.5 Distilled`, `Kandinsky 6 Distilled`, `FusioniX`, `PDD`): `4–10 steps` (use the model's default!)
5. **Click `Generate`**:
   - Watch live frame/video previews in the status panel as denoising progresses.
   - Completed videos are saved automatically to the **`outputs/`** folder and appear in the **Video Gallery**.

---

### Workflow 2: Image-to-Video (I2V) & First/Last Frame Storyboarding

Animating a still image gives you total control over character appearance and composition:

1. **Generate or Upload a Still Image**:
   - You can generate a keyframe right inside WanGP using **`Z-Image Turbo`**, **`Krea 2 Turbo`**, or **`Qwen Image 2.1`**, then send it from the Gallery to your video workflow—or upload your own image.
2. **Switch to an I2V Model**:
   - Select **`Wan 2.2 I2V 14B`**, **`MiniMax H3 FL2VA Pruned`**, or **`LTX 2.5 Distilled`**.
3. **Upload `Start Image` (and optional `End Image`)**:
   - Drop your image into the **Start Image** slot.
   - *(Optional)* Drop a second keyframe into the **End Image** slot to make WanGP generate a smooth motion bridge between the two frames.
4. **Write a Motion-Focused Prompt**:
   - Do **not** waste words re-describing every static detail already in the image. Focus on **what moves**:
     ```text
     The woman smiles softly, turns her head toward the rain-streaked window, and lifts her coffee cup as the camera slowly pushes in.
     ```
5. **Chain Shots into a Full Scene**:
   - In the Video Gallery, click to extract the **last frame** of your generated clip into the Image Gallery, then use it as the **Start Image** of your next generation!

---

### Workflow 3: Generating Video with Synchronized Speech, Music & SFX

With **`MiniMax H3 FL2VA Pruned`**, **`LTX 2.5 Distilled`**, or **`Kandinsky 6 Distilled`**, WanGP generates video and audio simultaneously:

1. Select **`MiniMax H3 FL2VA Pruned`** (or **`LTX 2.5 Distilled`**).
2. In your prompt, describe **both the visual action and the sound/dialogue**:
   ```text
   Medium close-up of a weathered sea captain standing on a wooden ship deck during a thunderstorm. Lightning flashes across the dark sky. He looks directly at the camera and says in a deep, gravelly voice: "Hold the helm steady, we are sailing straight through the eye!" Loud thunder rumbles, rain pelts his oilskin coat, and ocean waves crash against the hull.
   ```
3. To clone specific voices or keep an existing song/voice track, switch to **`MiniMax H3 Ref2VA Pruned`** and upload up to **3 Audio References** (or select an audio mode marked **Soundtrack Kept**).

---

### Workflow 4: Making Longer Multi-Scene Videos (Sliding Windows)

To generate videos longer than a single 5-second pass without running out of VRAM:

1. Enable **Advanced Mode** in the UI and open the **Sliding Window / Continuation** settings.
2. Set the total duration/frame count higher (e.g., 10–30 seconds). WanGP will generate the video in overlapping windows, using the last frames of each window to guide the next.
3. **Multi-Window Prompting**: Set *How to Process Each Line of the Text Prompt* to **Each Line (or Paragraph) Will be used for a new Sliding Window of the same Video Generation**, then write beat-by-beat actions:
   ```text
   A sleek silver spaceship approaches a glowing rings planet in deep space, slow cinematic pan
   The spaceship fires its blue thrusters and dives through the icy rings, debris flying past the camera
   The spaceship emerges above a futuristic bioluminescent city on the planet's surface at night
   ```

---

### Workflow 5: Autonomous Video Creation with Deepy (AI Director Agent)

WanGP includes **Deepy** (`Deepy Zero` / `Deepy Prime`), an offline low-VRAM AI agent that can plan shots, generate consistent character sheets, create video clips, add voiceovers/music, and assemble a complete edited video:

- **In the Web UI**: Open the **Deepy** tab (or click **Web app →** in Deepy's settings for a mobile-friendly chat interface).
- **From the Terminal**:
  ```bash
  python wgp.py --ask-deepy
  ```
- Give Deepy a brief such as: *"Create a 15-second cinematic commercial for a futuristic neon coffee shop in Tokyo with 3 shots, consistent barista character, rain ambience, and upbeat lo-fi music."*

---

### Workflow 6: Upscaling, Frame Interpolation & Audio Post-Processing

Once you have a base video (e.g., 480p at 24 fps), polish it inside WanGP's **Post-Processing / Media Flow** tabs:

1. **Smooth Frame Rate (Temporal Upsampling)**:
   - Use **RIFE (x2 / x3 / x4)** or **DLSS Frame Generation** to turn 24 fps clips into buttery-smooth 48 fps or 72 fps video.
2. **Increase Resolution (Spatial Upsampling)**:
   - Use **FlashVSR**, **LTX 2.5 Detail Refiner x2**, or **H3 VAE Upsampler x2** to upscale 480p/540p renders to crisp 1080p or 4K.
3. **Add or Replace Soundtracks**:
   - Use **MMAudio** to automatically synthesize matching sound effects for silent videos, or **SeedVC** to swap a speaker's voice.

---

## 6. Best VRAM & Performance Settings ("GPU Poor" Tuning)

If you have **6 GB – 12 GB VRAM**, configure these settings under the **Configuration** tab for maximum speed and zero Out-Of-Memory (OOM) errors:

1. **Memory Profile (`Configuration → RAM/VRAM Management`)**:
   - **Profile 4 (Default & Recommended)**: Keeps only the active model in pinned RAM and streams layers to the GPU on the fly. Lets an 8 GB GPU run 14B–33B models!
   - **Profile 4+ (`4.5`)**: Streams one part at a time—saves an extra ~1 GB VRAM.
   - **Profile 5 (Fail-Safe)**: Uses almost zero reserved RAM and minimal VRAM. Use if your PC has both low RAM (16 GB) and low VRAM (6–8 GB).
   - **Profile 1 or 3**: Use only if you have **24 GB+ VRAM** (RTX 3090 / 4090 / 5090) and want the entire model resident in VRAM.
2. **VRAM Allocator**:
   - Set to **`MMGP Optimized VRAM Allocator with RAM Spilling`** (`--vram-allocator vmm_spill`). It cuts peak VRAM by up to 50% on long videos and spills gracefully to system RAM instead of crashing if VRAM fills up.
3. **Smart Memory Pinning & MMGP RAM Allocator**:
   - Keep **Smart Memory Pinning** enabled and **RAM Allocator** set to **MMGP Optimized RAM Allocator**.
4. **Attention Mode (`Configuration → Performance`)**:
   - Select **`Sage2`** on RTX 30XX/40XX/50XX (up to 40% faster than SDPA).
   - Select **`Sage`** on RTX 20XX, or **`SDPA`** on GTX 10XX / AMD.
5. **Attention Head Split**:
   - Set *Attention Head Split* to **`Medium (good balance)`** under *Configuration → Performance* for an extra **20% VRAM reduction** on higher-resolution videos.
6. **Batch / Overnight Queue Processing (Headless CLI)**:
   - Queue up multiple video jobs in the Web UI, click **Save Queue** (`my_queue.zip`), and run them overnight without the browser:
     ```bash
     python wgp.py --process my_queue.zip --output-dir ./outputs
     ```

---

## 7. Complete Capacity & Every Single Possibility in WanGP (271+ Model Presets)

WanGP is not just a video generator—it is a complete **4-Modality Creative Studio (Video, Image, Audio/Music/TTS, and 3D)** plus an **Autonomous AI Director Agent (Deepy)** and **MCP Server**.

### A. Hardware Capacity & Output Limits
- **VRAM Floor**: Runs select models in as little as **6 GB VRAM** (GTX 10XX, RTX 20XX–50XX, AMD RDNA 2/3/4, Apple Silicon MPS).
- **Large Model Offloading (MMGP v4)**: Can run **14B, 20B, 22B, and 33B parameter models** on **8–12 GB VRAM** by streaming blocks from pinned system RAM (`Profile 4` / `Profile 5`) with RAM spilling (`vmm_spill`).
- **Quantization Formats Supported**: `BF16`, `FP16`, `FP8`, `INT8 (Quanto & ConvRot)`, `GGUF (Q2_K to Q8_0)`, `NF4`, `NVFP4 (RTX 50XX)`, and `Nunchaku INT4/FP4`.
- **Resolutions**: From `480p` (`832×480`) and `720p` (`1280×720`) up to **native `1080p`, `2K` (Prism), and `4K` (LTX-2.5 Two-Phase Tiling, SenseNova U1.5, Qwen Image 2.1, FlashVSR / SeedVR2 / DLSS 5 upscaling)**.
- **Durations**:
  - **Single-pass video**: Typically 2s to 15s (e.g., 15s of 1080p MiniMax H3 video in ~11 GB VRAM; 20s of 480p LTX-2.x video in ~28s on RTX 5090).
  - **Sliding-Window / Continuation Video**: Unlimited multi-minute sequences by chaining overlapping windows (`[/duration=5s]`, `[/new_shot]`, `[/overlap=9]`).
  - **Music tracks**: Up to **5 minutes** of full stereo song generation (`MiniMax Music 3`, `YuE2`, `ACE-Step 1.5 XL`).

---

### B. Every Video Generation & Editing Possibility
1. **Text-to-Video (T2V)**: Generate cinematic, anime, 3D, or realistic clips from text alone (`Wan 2.1/2.2`, `MiniMax H3`, `LTX-2.5`, `HunyuanVideo 1.5`, `Kandinsky 5/6`, `LongCat Video`, `Ovi 1.1`).
2. **Joint Video + Synchronized Audio Generation**: Generate video **and** lip-synced speech, sound effects, and background music in a single pass (`MiniMax H3 FL2VA/Ref2VA`, `LTX-2.5`, `Kandinsky 6 Pro 30B / Lite 3B`, `Wan 2.2 Prism`, `Ovi 1.1`).
3. **Image-to-Video (I2V) & First/Last Frame Bridging**: Animate a single image, or supply both a **Start Image** and **End Image** + **Injected Intermediate Frames** at exact timestamps to storyboard complex transitions.
4. **Multi-Reference Character, Object, Scene & Voice Consistency**:
   - **`MiniMax H3 Ref2VA`**: Feed up to **3 reference images, 3 reference videos (15s budget), and 3 reference voice clips (15s budget)**—or lock to an existing soundtrack (*Soundtrack Kept*).
   - **`LTX 2.5 MSR (Multi-Subject Reference)` & `Ingredients`**: Combine 1–5 subject, object, and background references.
   - **`JoyAI-Echo` & `JoyAI-Echo Surgical`**: Persistent character, voice, object, and location memory banks across multi-shot stories.
5. **Character Animation, Motion Transfer & Actor Replacement**:
   - **`Wan 2.2 Animate 2` & `SCAIL-2`**: Transfer body movement, facial expressions, and camera motion from a driving video onto one or multiple reference characters, with automatic relighting.
   - **`Viggle Animate`**: Replace a person or object in a video in just **3 steps** from a single edited frame.
6. **Masked Video Inpainting, Object Removal & Spatial/Temporal Outpainting**:
   - **`Wan 2.1 VACE 14B`**, **`MiniMax H3 ControlNet-Union 2.0`**, **`Bernini-R`**, **`Chrono Edit`**, **`Lucy Edit`**, **`Kiwi Edit`**: Remove objects, swap clothing/backgrounds, expand a vertical video into widescreen (**Spatial Outpainting**), or extend a clip before/after (**Temporal Outpainting**).
7. **Identity-Preserving Face Swap & Face Refinement**:
   - **`Lynx` & `VACE Lynx`**: Swap faces in video while keeping identity under changing angles and lighting.
   - **`H3 Face Refiner`**: Automatically detect, track, and restore up to **5 faces** in any generated or imported video.
8. **3D Camera Path Control**:
   - **`Fun Control-Camera 1.3B` & `Uni3C 14B`**: Fly a virtual camera (pan, orbit, dolly zoom, custom trajectory) through a single still photo using estimated 3D geometry.
   - **`Vista4D`**: Reshoot an existing dynamic video along a brand-new camera trajectory.
9. **Talking Heads, Avatars & Multi-Speaker Lip-Sync**:
   - **`InfiniteTalk` & `MultiTalk`**: Turn a portrait + 1 or 2 audio tracks into a long-form lip-synced conversation with natural hand gestures and head movement.
   - **`LongCat Avatar 1.5`**, **`Hunyuan Video Avatar / Custom Audio`**, **`Magi Human`**: Audio-driven talking or singing portraits.
10. **Professional VFX Pipelines**:
    - **`LTX-2.5 Layout-to-Render`**: Convert a rough 3D viewport playblast/blocking animation + 1 styled reference frame into a fully rendered cinematic shot.
    - **`LTX-2.5 Alpha Gen` & `Wan Alpha`**: Generate videos with transparent backgrounds or extract soft alpha mattes exported as **ProRes 4444** or **RGBA PNG ZIPs**.
    - **`LTX-2.5 SDR-to-HDR`**: Convert standard SDR video into **HDR10** video.

---

### C. Every Image Generation, Design & Layering Possibility
1. **Photorealistic & Artistic Text-to-Image**: `Flux 3 Klein 5.6B` (up to 10 references + visual layout box helper), `Flux 2 Dev / Klein 4B & 9B`, `Flux 1 Dev / Schnell / Chroma / Krea / SRPO`, `Krea 2 RAW & Turbo` (strong celebrity & landmark knowledge), `Z-Image Turbo 6B`, `HiDream O1 Dev`, `MOSS Image 5B`.
2. **Instruction-Based Image Editing & Multi-Reference Composition**:
   - **`Qwen Image 2.1 (7B)` & `Qwen Image Edit Plus (20B)`**: Edit images via natural language (*"Add a red hat, remove background people, rotate pose"*), combine up to 10 references, or paint with **LanPaint** inpainting.
   - **`Flux 1 Dev Kontext` & `DreamOmni2`**: Context-aware character and object editing.
3. **Typography, Posters, Infographics & Structured Layouts**:
   - **`SenseNova U1.5 (8B MoT)`**: Native-4K posters and infographics with crisp English/Chinese text.
   - **`Ideogram 4` & `Ming Image 0.1 Design`**: Visual **Prompt Helper** canvas to arrange text boxes, logos, and layers with exact JSON layout control.
4. **Decomposing Flat Images into Editable Transparent RGBA Layers**:
   - **`Ming Image 0.1 Design-Layer` & `Qwen Image Layered (20B)`**: Feed any finished poster or illustration and automatically split it into separate front-to-back transparent PNG layers exported as a ZIP!

---

### D. Every Audio, Voice Cloning, Radio Drama & Music Possibility
1. **Multi-Speaker Voice Cloning & Emotional Dialogue**:
   - **`Fish Audio Drama 3`**: Multi-character radio dramas with up to **3 cloned voices**, whispers, laughter, sighs, and pauses.
   - **`IndexTTS 2 / 2.5`**: Expressive voice cloning with emotion vectors, speed control, and multilingual support (EN, ZH, JA, ES, AR).
   - **`Qwen3 TTS (Base / Custom Voice / Voice Design)`**, **`OmniVoice`**, **`KugelAudio`**, **`Chatterbox`**, **`MiniMax H3 Voice Clone`**, **`DramaBox`**, **`Scenema Audio`**.
2. **Instruction-Driven Speech Editing & Cleanup**:
   - **`AuK Base & AuK Flash`**: Generate speech from text instructions, edit words inside an existing recording, remove background noise, or separate overlapping speakers.
3. **Full Song Generation (Vocals + Instrumentals)**:
   - **`YuE2` & `YuE2 Hum-to-Song`**: Generate full songs from lyrics + style tags, create covers following a source song's melody/chords, extend **ABC / MIDI musical scores**, or **hum a melody into your microphone** and turn it into a fully produced stereo song!
   - **`MiniMax Music 3`**: Studio-quality stereo songs up to 5 minutes long.
   - **`ACE-Step 1.5 / 1.5 XL Turbo`** (with 0.6B/1.7B/4B LLM musical structure planners) & **`HeartMuLa 3B`**.
4. **Sound Effects (SFX), Foley & Instrumental Ambience**:
   - **`Stable Audio 3 (Small / Medium / SFX)`**: Generate ambient soundscapes, musical loops, and custom sound effects.
   - **`MMAudio` & `PrismAudio`**: Automatically watch a silent video and synthesize synchronized foley and soundtrack audio for it.
   - **`RoFormer` & `SeedVC`**: Strip vocals to create karaoke/instrumental stems, or convert a singer/speaker's voice into another reference voice.

---

### E. Every 3D Asset, Scene Reconstruction & Mesh Animation Possibility
1. **Single Image to Textured 3D Mesh (GLB)**:
   - **`TRELLIS.2`** & **`Hunyuan3D 2.1`**: Turn one picture of an object/character into a 3D mesh with full **PBR textures** (base color, metallic, roughness), or re-texture an existing 3D mesh.
   - **`TripoSG`** (high-detail untextured geometry) & **`TripoSplat`** (3D Gaussian Splats `.ply` for fuzzy/thin structures).
2. **Multi-Part 3D Generation & Mesh Splitting**:
   - **`PartCrafter`** & **`GeoSAM2`**: Generate or segment a 3D object into separate movable sub-meshes (wheels, limbs, armor pieces) inside a single GLB.
3. **Photos / Video to 3D World Reconstruction**:
   - **`WorldMirror 2.0`**, **`MapAnything`**, **`Depth Anything 3 Points`**, **`MoGe-2`**: Convert a video walk-around or photo set into a navigable 3D Gaussian Splat scene or metric 3D point cloud with camera poses.
4. **Rig-Free Text-to-3D Mesh Animation**:
   - **`AnimateAnyMesh`**: Type a motion prompt (e.g., *"The dragon flaps its wings"*) to animate a static 3D mesh without rigging or skinning.
5. **3D-to-Video Synergy**: Rotate any 3D model in WanGP's **3D Viewer**, snap a high-res render or depth map from any camera angle, and feed it straight into an Image-to-Video model or let **Deepy** build a multi-angle **Character Turnaround Sheet**.

---

### F. Built-In Preprocessing, Upscaling & Finishing Pipeline (`Media Flow`)
- **Preprocessors**:
  - **MatAnyone & SAM3**: Interactive 1-click video/image object tracking and alpha matting.
  - **DWPose**: Human body, hand, and facial skeleton pose extraction.
  - **Depth Anything v2 / v3 & MiDaS**: Depth map extraction.
  - **RAFT Optical Flow, Canny Edge, Scribble, Rembg Background Remover, InsightFace / ArcFace, Whisper Transcription & Pyannote Speaker Diarization**.
- **Spatial Upsamplers (Low-Res → 1080p / 4K)**:
  - **`SeedVR2`**, **`FlashVSR`**, **`LTX 2.5 Detail Refiner x2`**, **`H3 VAE Upsampler x2`**, **`Chain of Zoom`**, **`DLSS 5 Neural Rendering`**, and **`Lanczos`**.
- **Temporal Upsamplers (FPS Boosting)**:
  - **`RIFE v4.26 (x2 / x3 / x4)`** and **`DLSS Multi-Frame Generation (up to x6)`** to convert 16/24 fps generations into smooth 48/60/120 fps footage.
- **Finishing**: Film grain overlay, FFmpeg lossless FLAC/MP4/MKV/ProRes remuxing, multi-track audio mixing, and soft/hard subtitle burning (`SRT`/`VTT`/`ASS`).

---

### G. Autonomous Agent (`Deepy`), Plugins & API Automation
- **Deepy Zero & Deepy Prime**: Conversational AI producer accessible in the Gradio UI, terminal (`--ask-deepy`), or mobile web app (`/deepy/`). Can write storyboards, draw SVG/canvas layouts, render 3D turnaround sheets, generate and QA video shots (checking for black frames, unwanted cuts, silent stretches, or text typos), add lower-thirds/titles, and mix final soundtracks.
- **MCP Server (`--mcp` / `--mcp-webui`)**: Exposes WanGP's full generation, inspection, and media-processing pipeline via the **Model Context Protocol (MCP v2.1)** so external AI agents (Claude Code, Codex, OpenCode, custom apps) can orchestrate video production programmatically.
- **Built-in & Community Plugin Ecosystem**: Includes **Media Flow** (batch processor), **Motion Designer**, **Video Mask Creator**, **Models Manager**, plus 20+ one-click community plugins in `plugins.json` (H3 Director timeline, LoRA Manager/Merger, CivitAI Browser, Prompt Studio, Queue Editor, Telegram/Discord Queue Notifier, Wildcards).

---

## 8. FAQ: Unlimited Generations, Making It Live (Render vs. GPU Clouds) & Dashboard Overview

### Q1: Does this repository give UNLIMITED video generations?
**Yes—100% unlimited and 100% free of API fees, credits, subscriptions, or watermarks.**
- Unlike paid SaaS tools (Runway, Kling, Sora, Pika, Luma) that charge per credit or limit your monthly generations, WanGP downloads open-source AI weights directly from Hugging Face and runs them on the GPU where WanGP is installed.
- You can generate **hundreds or thousands of videos, songs, voiceovers, images, and 3D models** with **zero per-video cost**, **zero watermarks**, and **no license fees**. Your only limit is your GPU's rendering time and storage space.

---

### Q2: Do we HAVE to deploy / make it "live" on the internet to make it work?
**No! You do NOT need to deploy it to a web host.**
- **Local PC Mode (Default):** If you have a gaming/creator PC or laptop with an NVIDIA GPU (6 GB+ VRAM) or AMD RDNA GPU, you simply run `python wgp.py` (or `scripts\run.bat`) on your computer. It opens privately in your own browser at `http://localhost:7860` without ever needing web hosting.
- **Local Wi-Fi Mode:** Run `python wgp.py --listen` and you can open the dashboard from your phone, tablet, or another laptop on your home Wi-Fi (`http://<your-PC-IP>:7860`).
- **Instant Public Share Link (No Hosting Needed):** Run `python wgp.py --share` and Gradio automatically creates a free temporary public `https://xxxx.gradio.live` link that works from anywhere in the world while the script is running.

---

### Q3: Can we make it live on Render (Render.com)?
**No — standard Render.com plans will NOT work for WanGP.**
- **Why Render fails:** Render's free and standard web services are **CPU-only** with small RAM (512 MB – 4 GB) and small disks. WanGP requires a **physical NVIDIA CUDA or AMD ROCm GPU with at least 6 GB – 24 GB of VRAM** plus **25 GB – 100 GB of disk space** to store multi-gigabyte video diffusion models (`ckpts/`). On a CPU-only platform like Render, Vercel, Railway, or Netlify, `wgp.py` cannot initialize CUDA (`torch.cuda`) and cannot run video diffusion.
- **Where SHOULD you run it if you want it live in the cloud?**
  1. **100% Free Cloud GPU Options ($0 cost):**
     - **Google Colab** (Free NVIDIA Tesla T4 with **15 GB VRAM**): Upload the included [`WanGP_Colab_Free_GPU.ipynb`](WanGP_Colab_Free_GPU.ipynb) notebook to [colab.research.google.com](https://colab.research.google.com/), set Runtime → **T4 GPU**, and click Run. It will print a live `https://xxxx.gradio.live` dashboard URL!
     - **Kaggle Notebooks** (Free **2× Tesla T4 GPUs = 30 GB VRAM** for 30 hours/week).
     - **Lightning AI Studios** (Free monthly GPU quota).
  2. **Low-Cost Dedicated GPU Cloud Providers (For 24/7 or Heavy Production):**
     - **RunPod.io** or **Vast.ai** (~$0.15–$0.39/hour for an RTX 3090 / RTX 4090 with 24 GB VRAM). WanGP already includes a [`Dockerfile`](Dockerfile) and [`entrypoint.sh`](entrypoint.sh) designed specifically for GPU Docker containers!
     - **Hugging Face Spaces** (selecting an NVIDIA T4 / L4 / A10G GPU hardware tier).

---

### Q4: Will it show a visual Dashboard?
**Yes! WanGP launches TWO full visual web dashboards out of the box:**

1. **The Main WanGP Studio Dashboard (`http://localhost:7860/` or `https://xxxx.gradio.live`)**:
   - **Top Model & Workspace Bar**: Search and switch between all **271+ Video, Audio, Image, and 3D presets**, switch project Workspaces, and pick from **5 UI color themes**.
   - **Creation Center**: Prompt editor (with microphone dictation and 1-click AI **Prompt Enhancer**), uploaders for **Start Image / End Image / Reference Images / Control Video / Reference Voices**, interactive **MatAnyone Mask Editor**, Resolution & Aspect-Ratio sliders, and LoRA pickers.
   - **Live Video Preview & Queue Manager**: Watch your video animate frame-by-frame in real time (`Tiny VAE looping video preview`) while it generates, plus a **Generation Queue** to line up batch jobs, reorder tasks, or save queues.
   - **Synchronized Media Galleries**: Built-in **Video Gallery**, **Image Gallery**, **Audio Player**, and interactive **3D Mesh/Splat Viewer** with 1-click buttons to reuse any output as the next input.
   - **Built-in Studio Apps (Tabs)**:
     - **Post-Processing & Media Flow** (Upscaling to 1080p/4K, RIFE frame interpolation, MMAudio foley, SeedVC voice swap)
     - **Motion Designer & Video Mask Creator**
     - **Plugin Manager & Checkpoints/Models Manager**
     - **Configuration Panel** (VRAM profiles, attention kernels, output formats)
2. **The Deepy Web App Dashboard (`http://localhost:7860/deepy/`)**:
   - A clean, **phone-friendly chat & storyboard dashboard** where you simply message **Deepy** (or talk via voice), attach photos/clips, watch live progress cards as it plans and renders shots, and download the finished movie.


