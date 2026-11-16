# Anima TrainFlow

Anima TrainFlow is a streamlined, one-page GUI for training LoRA on the **Anima 2B** model (made for **Anima-Base v1.0**, the version the Anima model card says LoRAs should be trained on). Optimized to run on hardware with as little as **6GB of VRAM**, it eliminates technical overhead by focusing on the essential settings that impact training results the most.

### ⚡ Everything you need in one tool
**A complete, ready-to-use workflow from start to finish.**

`Raw Images` ➔ `Smart Resize & Crop*` ➔ `Auto-Tag` ➔ `Quality Filter (optional)` ➔ `One-Click Train**` ➔ `Ready LoRA`

* **Smart Prep:** In most cases, it only performs resizing. Intelligent cropping is triggered only for extreme aspect ratios that don't fit into training buckets.
* **Fast Results:** Average training time is **~1 hour** (benchmarked on RTX 3060 12GB).

![Anima TrainFlow Interface Preview](preview.png)

## Quick Start (Portable)
> The portable archive is the upstream v1.1.0 build. This fork's tagger and quality filter need packages it does not ship (timm, dghs-imgutils): use the uv installation below.

1. [**Download Portable Version (3GB)**](https://huggingface.co/datasets/ThetaCursed/Anima-TrainFlow-Portable/resolve/main/Anima-TrainFlow-v1.1.0-Portable.7z?download=true)
2. **Extract the archive using [7-Zip](https://7-zip.org/) or WinRAR**.
3. Run `start_trainer.bat`.
4. Open the `🔧 Paths to Models <- Set Once` accordion and specify the paths to your model files.
5. Specify your **Dataset Path** (images + .txt files) and **Trigger Word**, then click **Start**.

## Installation (uv)
This fork manages its Python environment with [uv](https://docs.astral.sh/uv/): `pyproject.toml` lists the dependencies and `uv.lock` pins the exact versions (Python 3.10, PyTorch CUDA 12.8, sd-scripts requirements, tagger and aesthetic models).

1. **Install uv** (once):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/ThetaCursed/Anima-TrainFlow
   cd Anima-TrainFlow
   ```

3. **Install dependencies:**  
   Run `Install_Requirements.bat` (it runs `uv sync`, which creates `.venv`).

4. **Models:**
   *   **Anima** (DiT, text encoder, VAE) from [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima): `anima-base-v1.0.safetensors`, `qwen_3_06b_base.safetensors`, `qwen_image_vae.safetensors`. Set their paths once in `🔧 Paths to Models`.
   *   **Tagger** [animetimm/convnextv2_huge.dbv4-full](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) (auto-captioning): downloaded to the Hugging Face cache on first use. If the Hub asks for a login, accept the terms on the model page and run `uv run hf auth login`. For offline use, put `config.json`, `model.safetensors`, `selected_tags.csv` and `preprocess.json` in `models/convnextv2_huge.dbv4-full/`.
   *   **Aesthetic model** [deepghs/anime_aesthetic](https://huggingface.co/deepghs/anime_aesthetic) (quality filter): downloaded automatically on first use.
   *   **U2Net Model** (used for Smart Cropping):
       ```bash
       curl -L -o models/u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
       ```

5. **Launch the Trainer:** 
   Run `start_trainer.bat` (`uv run app.py`; it also brings `.venv` up to date with `uv.lock`). Without uv it falls back to `python_embeded\python.exe` (portable build).

## Key Features
* **Zero-Tab Interface:** All critical parameters (Trigger Word, Rank, LR, Steps) are accessible on a single screen.
* **Live Training Previews:** Watch your LoRA improve in real-time. The built-in gallery automatically updates whenever a new sample is generated.
* **AI-Powered Smart Cropping:** Integrated U2Net model automatically performs subject-aware, head-first cropping and resizes images to optimal aspect-ratio buckets via multi-threading.
* **Built-in Auto-Captioning:** [animetimm/convnextv2_huge.dbv4-full](https://huggingface.co/animetimm/convnextv2_huge.dbv4-full) (bf16 on the GPU) writes `.txt` captions using each tag's calibrated threshold (`best_threshold` from the model), with optional minimum confidences for general and character tags. Captions follow Anima's tag order (safety tag, `1girl`/`1boy`/..., characters, general tags), use spaces instead of underscores and keep parentheses unescaped (`hatsune miku (vocaloid)`), because sd-scripts trains the caption text verbatim.
* **Never-Apply Tags:** list tags (comma separated, `*` wildcard, e.g. `signature, watermark, *username`) that are left out when tagging and stripped from existing captions when training.
* **Aesthetic Quality Filter:** scores every image with deepghs `anime_aesthetic` (percentile on Danbooru's score distribution → masterpiece ≥ 95%, best ≥ 85%, great ≥ 75%, good ≥ 50%, normal ≥ 25%, low ≥ 10%, worst). Pick the labels to keep; the others are either ignored in training (files untouched) or moved to `<dataset>\_quality_filtered\<label>\` together with their caption and bucketing original (**Restore Moved Images** puts them back). Scores are cached in `<dataset>\aesthetic_scores.json`.
* **Pre-Flight Validation:** Automatically scans the dataset for missing captions, oversized images (>=2048px), and missing model paths to prevent crashes before training starts.
* **Persistent Sessions:** All UI inputs, paths, and slider positions are instantly auto-saved and restored on the next launch.
* **Portable Edition:** Includes an embedded Python environment to avoid installation or complex setup.
* **Low VRAM Friendly:** Specifically tuned for 6GB+ NVIDIA GPUs with aggressive RAM/VRAM clearing between tasks.
* **Optimized Defaults:** Pre-configured for BF16 precision and latent caching to ensure maximum performance and stability.
* **Anima-Recipe Optimizer Defaults:** AdamW8bit at 2e-5 (rank 32, alpha = rank), betas 0.9/0.99, weight decay 0.01, short warmup then a constant LR, as in the Anima author's reference config. Prodigy (adaptive LR) is still available in the dropdown.

## Dataset Preparation
Place all your training images (.png, .jpg, .webp, .bmp) in a single folder. Every image must have a matching text file with the same name containing its tags/captions (e.g., `image1.png` and `image1.txt`). You can easily generate these text files using the built-in **Auto-Caption Dataset** tool.

When training starts, TrainFlow writes the captions sd-scripts will actually encode to `training/output/<project>/configs/<project>_captions.json`: the trigger word first, then the tags of the `.txt` (never-apply tags removed, `\(`/`\)` unescaped). Only the images listed there are trained, which is also how the quality filter's "ignore" mode works.

## Anima-Base v1.0 notes
* Train on `anima-base-v1.0.safetensors` (the Anima-Aesthetic and Turbo versions are finetunes/distillations of it). It has the same architecture as the previews (28 blocks, LLM adapter inside the DiT file), so the sd-scripts loader is unchanged.
* The LLM adapter is never trained (`train_llm_adapter=False`), as the model card asks.
* Learning rate: the model card suggests starting at **2e-5 for a rank 32 LoRA** and adjusting from there, and that is the default (AdamW8bit). The rest follows the author's diffusion-pipe example config: betas 0.9/0.99, weight decay 0.01, `min(100, steps/10)` warmup steps, then a constant LR (`constant_with_warmup`; a cosine decay would halve the LR on average). Prodigy picks its own, usually much higher, step size: if you use it on the base model, watch the previews for overcooking.
* Sample prompts: the card's recommended negative prompt is the default; a good positive prefix is `masterpiece, best quality, score_7, safe, `. It works at 512²–1536², 30–50 steps, CFG 4–5.

### Fixed in this fork
* **The trigger word was never trained.** sd-scripts only applies `caption_prefix` when it tokenizes on the fly; with `cache_text_encoder_outputs` (always on here) it encodes the raw `.txt`. The trigger is now written into the captions themselves.
* **Stale text encoder cache.** sd-scripts reuses `cache_text_encoder/*.npz` as long as the file exists, so re-tagging a dataset or changing the trigger word trained on the old captions. TrainFlow now records what each cache entry was built from and re-encodes what changed (same for latents of images that changed). The first run after updating re-encodes all captions once.

## Troubleshooting

**Training stops at a random step / the step counter freezes (no error).**
The trainer runs as a separate process, and its output is drained by a dedicated
thread that never depends on the browser. Older builds read that output directly
inside the Gradio request: if the tab was hidden, reloaded or the UI was slow, the
pipe buffer filled up and the training process blocked forever inside `write()` —
the step counter simply stopped with no traceback. If you still see a stall:

* Check `training/output/<project>/train.log` — the full trainer output is always
  written there, even if the UI is not updating.
* If the log says `⏳ Trainer is alive, no output for N min`, the GPU is still
  working (sampling or saving a checkpoint) — it is not frozen.
* Press **Stop** before starting a new run. Leftover `python.exe`/accelerate
  processes keep VRAM busy, and the app now kills them automatically before a run.
* Reloading the page is safe: pressing **Start Training** re-attaches to the run
  that is already in progress instead of launching a second job.

## System Requirements
* **OS:** Windows 10/11.
* **GPU:** NVIDIA GPU (6GB+ VRAM recommended for Anima 2B training).
* **Storage:** ~6.5GB of free space (SSD recommended).

## Technical Details
* **Core:** Based on a modified version of `sd-scripts` for Anima 2B architecture.
* **UI:** Built with Gradio featuring a customized dark theme.
* **Backend:** Utilizes `accelerate launch` for optimized execution, on the same interpreter as the UI (`.venv` with uv).
* **Dataset tools:** `dataset_tools.py` (tagger, aesthetic scorer, caption building, cache invalidation).
* **Auto-Save:** All paths and configurations are automatically saved to `settings.json`.