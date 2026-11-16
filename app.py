import os
import re
import json
import platform
import subprocess
import atexit
import toml
import glob
import math
from collections import deque
from pathlib import Path
import sys
import gradio as gr
import psutil
from PIL import Image
import torch
import numpy as np
import onnxruntime as rt
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import shutil
import warnings
from typing import Generator, List, Tuple

# gradio 6.14 reads starlette's old name for HTTP 422 on every queued event, and
# starlette >= 0.48 warns about it (same value, renamed per RFC 9110). gradio
# 6.20+ uses the new name but requires huggingface-hub >= 1.2, which the pinned
# sd-scripts stack (transformers < 5) cannot use - so only this warning is muted.
warnings.filterwarnings(
    "ignore", message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated", module=r"gradio\.routes"
)

from dataset_tools import (
    CHARACTER_FLOOR_DEFAULT, FILTERED_DIR, QUALITY_FLOORS, QUALITY_LABELS,
    AestheticScorer, DbTagger, ScoreCache, TagBlacklist,
    collect_training_entries, join_caption, list_images, model_fingerprint,
    move_out_of_dataset, percentile_to_label, refresh_disk_caches, restore_moved,
    write_caption_metadata,
)

CSS = """
.gradio-container {
    position: relative !important;
}

#main-header {
    margin-bottom: -10px !important;
}

.attribution {
    position: absolute;
    right: 25px;
    top: 5px;
    z-index: 99999;
    pointer-events: none;
}

.author-text {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(187, 154, 247, 0.6);
    letter-spacing: 0.5px;
}

#log-container textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 14px !important;
    line-height: 1.3 !important;
    white-space: pre-wrap !important;
    overflow-x: hidden !important;
    height: 385px !important;
    border: 1px solid #2f334d !important;
}

*::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
*::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1) !important;
}
*::-webkit-scrollbar-thumb {
    background: #7aa2f7 !important;
    border-radius: 10px !important;
}
*::-webkit-scrollbar-thumb:hover {
    background: #bb9af7 !important;
}

footer {
    display: none !important;
}

"""

JS_SCROLL = """
function() {
    setTimeout(() => {
        const el = document.querySelector('#log-container textarea');
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    }, 50);
}
"""


LOG_BLACKLIST = [
    "triton not found",
    "flop counting will not work",
    "Lib\\site-packages\\torch\\utils\\flop_counter.py"
]


LOG_BOX__MAX_LINES = 16
GALLERY_HEIGHT = 440
MAX_LOG_LINES = 500
MAX_GALLERY_IMAGES = 24
STEP_RE = re.compile(r"(\d+)/(\d+)")

# How often the UI generator refreshes its view of the running trainer.
UI_POLL_SECONDS = 0.5
# If the trainer prints nothing for this long we say so in the log, so a slow
# sampling/saving phase is not mistaken for a frozen process.
STALL_WARN_SECONDS = 90


ROOT = Path(__file__).resolve().parent
# The trainer runs on the interpreter that runs the UI: the uv-managed .venv by
# default (start_trainer.bat -> `uv run app.py`), or python_embeded\python.exe
# when the portable build is used instead.
PYTHON_EXE = Path(sys.executable)

TRAIN_BASE = ROOT / "training"
OUTPUT_BASE = TRAIN_BASE / "output"
SETTINGS_FILE = TRAIN_BASE / "settings.json"

TRAIN_DIR = TRAIN_BASE / "sd-scripts" 
TRAIN_SCRIPT = TRAIN_DIR / "anima_train_network.py"

for d in [TRAIN_BASE, OUTPUT_BASE]:
    d.mkdir(parents=True, exist_ok=True)


QUALITY_MODE_OFF = "Off"
QUALITY_MODE_IGNORE = "Ignore them in training"
QUALITY_MODE_MOVE = "Move them out of the dataset"
QUALITY_MODES = [QUALITY_MODE_OFF, QUALITY_MODE_IGNORE, QUALITY_MODE_MOVE]

# Anima model card: "For a rank 32 LoRA, start with 2e-5 and adjust up or down
# from there." The default LR (AdamW8bit), and the one the UI restores when
# switching back from Prodigy to an AdamW optimizer.
ADAMW_DEFAULT_LR = "0.00002"
# The rest of that reference recipe (the diffusion-pipe example config of the
# model's author): betas (0.9, 0.99), weight decay 0.01, 100 warmup steps, then
# a constant LR - no decay.
ADAMW_WARMUP_STEPS = 100

# Key order = order of the components in the UI (see SETTINGS_COMPONENTS).
DEFAULT_SETTINGS = {
    "trigger_word": "",
    "dataset_path": "",
    # Anima-Base is the version the model card says LoRAs should be trained on
    # (the previews are superseded; same 28-block architecture).
    "dit_path": str(ROOT / "models" / "anima" / "dit" / "anima-base-v1.0.safetensors"),
    "qwen_path": str(ROOT / "models" / "anima" / "text_encoder" / "qwen_3_06b_base.safetensors"),
    "vae_path": str(ROOT / "models" / "anima" / "vae" / "qwen_image_vae.safetensors"),
    "network_rank": 32,
    # A light touch on the base model, as the Anima model card asks; Prodigy
    # (LR 1.0, picks its own step size) is still in the dropdown.
    "learning_rate": ADAMW_DEFAULT_LR,
    "optimizer": "AdamW8bit",
    "training_steps": 2400,
    "save_steps": 300,
    "sample_steps": 300,
    "pos_prompt": "",
    # Negative prompt recommended by the Anima model card
    "neg_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration",
    "width": 1024,
    "height": 1024,
    "sample_steps_gen": 30,
    "sample_cfg": 4.0,
    "sample_seed": 42,
    "train_seed": 42,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "side_min": 512,
    "side_max": 768,
    # dbv4 tagger: floors on top of the per-tag calibrated thresholds
    "tagger_gen_floor": 0.0,
    "tagger_char_floor": CHARACTER_FLOOR_DEFAULT,
    "tagger_blacklist": "",
    "tagger_add_safety": True,
    "tagger_overwrite": False,
    "quality_keep": list(QUALITY_LABELS),
    "quality_mode": QUALITY_MODE_OFF,
}
def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings

def save_settings(settings_dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4)

def auto_save_state(*args):
    keys = list(DEFAULT_SETTINGS.keys())
    current_state = dict(zip(keys, args))
    save_settings(current_state)


HIDDEN_SETTINGS = {
    # (the LR scheduler depends on the optimizer: see create_training_toml)
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "gradient_checkpointing": True,
    "network_module": "networks.lora_anima",
    "network_train_unet_only": True,
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "weighting_scheme": "logit_normal",
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs_to_disk": True,
    "cache_text_encoder_outputs": True,
    "sdpa": True,
    # 0 workers: sd-scripts then loads batches in the main process. On Windows a
    # spawn-based DataLoader worker can stall forever inside next(train_dataloader)
    # with no error and freeze the step counter at a random step. Latents and text
    # encoder outputs are cached to disk, so a worker buys no measurable speed.
    "max_data_loader_n_workers": 0,
    "persistent_data_loader_workers": False,
    "max_grad_norm": 1.0,
    "vae_batch_size": 1,
    "blocks_to_swap": 0,
    "sigmoid_scale": 1.3,
    # Anima model card: "Don't train the LLM adapter" - it has an outsized
    # influence on the output and is easy to degrade. sd-scripts' default is
    # already off; spelled out so a future default change cannot flip it.
    "network_args": ["train_llm_adapter=False"],
    "caption_dropout_rate": 0.05,
}


def analyze_dataset_resolution(sizes: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Training resolution from the sizes of the images that will be trained."""
    if not sizes: return 512, 768

    max_area = max(w * h for w, h in sizes)
    max_side = max(max(w, h) for w, h in sizes)

    # Calculate the side of the square that covers the area of the largest bucket
    # If the largest is 512x768, this yields 640.
    base_res = int(math.ceil(math.sqrt(max_area) / 64.0) * 64)
    max_bucket = int(math.ceil(max_side / 64.0) * 64)

    return base_res, max_bucket


def create_sample_prompts(project_name, trigger_word, pos_prompt, neg_prompt, width, height, steps_gen, cfg, seed, out_dir):
    prompt_path = out_dir / f"{project_name}_prompts.txt"
    trigger = trigger_word.strip()
    user_prompt = pos_prompt.strip().replace("\n", " ")
    
    if trigger and not user_prompt.startswith(trigger):
        actual_pos = f"{trigger}, {user_prompt}" if user_prompt else trigger
    else:
        actual_pos = user_prompt if user_prompt else trigger

    actual_neg = neg_prompt.strip().replace("\n", " ")
    prompt_str = f"{actual_pos} --n {actual_neg} --w {int(width)} --h {int(height)} --l {float(cfg)} --s {int(steps_gen)} --d {int(seed)}"
    with open(prompt_path, "w", encoding="utf-8") as f: f.write(prompt_str)
    return str(prompt_path)

def create_dataset_toml(project_name, dataset_path, metadata_path, num_images, base_res, max_bucket, out_dir, max_steps, batch_size, grad_acc):
    config_path = out_dir / f"{project_name}_dataset.toml"

    num_images = max(1, int(num_images))
    effective_batch = int(batch_size) * int(grad_acc)
    calculated_repeats = (int(max_steps) * effective_batch) / num_images
    final_repeats = max(1, math.ceil(calculated_repeats))

    # A fine-tuning style subset (metadata_file) instead of image_dir + .txt:
    # sd-scripts then trains exactly the images listed there, with the caption
    # written there. The old `caption_prefix` was how the trigger word got in,
    # but with cache_text_encoder_outputs sd-scripts encodes the raw .txt and
    # never applies the prefix - the trigger word was never trained.
    dataset_config = {
        "general": {
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": max_bucket,
            "bucket_reso_steps": 64,
            "bucket_no_upscale": True
        },
        "datasets": [{
            "resolution": base_res,
            "subsets": [{
                "metadata_file": Path(metadata_path).resolve().as_posix(),
                "image_dir": Path(dataset_path).resolve().as_posix(),
                # joins the metadata "caption" (trigger) and "tags"
                "caption_separator": ", ",
                "num_repeats": final_repeats,
                "caption_dropout_rate": HIDDEN_SETTINGS["caption_dropout_rate"],
            }]
        }]
    }
    with open(config_path, "w", encoding="utf-8") as f: 
        toml.dump(dataset_config, f)
    return str(config_path)

def create_training_toml(project_name, config_save_dir, actual_output_dir, rank, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc):
    config_path = config_save_dir / f"{project_name}_training.toml"

    network_rank = int(rank)
    network_alpha = network_rank
    
    warmup_steps = 0
    if optimizer == "Prodigy":
        scheduler = "constant"
        opt_args = ["decouple=True", "weight_decay=0.01", "d_coef=1.0", "use_bias_correction=True", "safeguard_warmup=True", "betas=0.9,0.99"]
    else:
        # Anima's reference AdamW recipe (see ADAMW_WARMUP_STEPS). The 2e-5 of
        # the model card is a constant LR: a cosine decay would halve it on average.
        scheduler = "constant_with_warmup"
        opt_args = ["weight_decay=0.01", "betas=0.9,0.99"]
        warmup_steps = min(ADAMW_WARMUP_STEPS, int(max_steps) // 10)

    training_config = {
        "pretrained_model_name_or_path": Path(models["dit_path"]).resolve().as_posix(),
        "qwen3": Path(models["qwen_path"]).resolve().as_posix(),
        "vae": Path(models["vae_path"]).resolve().as_posix(),
        "network_module": HIDDEN_SETTINGS["network_module"],
        "network_args": HIDDEN_SETTINGS["network_args"],
        "network_dim": int(rank),
        "network_alpha": network_alpha,
        "network_train_unet_only": HIDDEN_SETTINGS["network_train_unet_only"],
        "gradient_checkpointing": HIDDEN_SETTINGS["gradient_checkpointing"],
        "max_grad_norm": 1.0,
        "learning_rate": float(lr),
        "optimizer_type": optimizer,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        "lr_warmup_steps": warmup_steps,
        "max_train_steps": int(max_steps),
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "sample_sampler": "euler",
        "timestep_sampling": HIDDEN_SETTINGS["timestep_sampling"],
        "discrete_flow_shift": HIDDEN_SETTINGS["discrete_flow_shift"],
        "sigmoid_scale": HIDDEN_SETTINGS["sigmoid_scale"],
        "weighting_scheme": HIDDEN_SETTINGS["weighting_scheme"],
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "attn_mode": "sdpa",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "max_data_loader_n_workers": HIDDEN_SETTINGS["max_data_loader_n_workers"],
        "vae_chunk_size": 32,
        "vae_disable_cache": True,
        "seed": int(train_seed),
    }

    # 0 is NOT a valid value for these: sd-scripts evaluates `global_step % n` and
    # `steps % n`, so a 0 crashes with ZeroDivisionError at the first step. Only
    # write the key when the user actually wants periodic previews/checkpoints.
    if int(sample_steps) > 0:
        training_config["sample_every_n_steps"] = int(sample_steps)
    if int(save_steps) > 0:
        training_config["save_every_n_steps"] = int(save_steps)
    training_config["sample_prompts"] = Path(prompt_path).resolve().as_posix()
    with open(config_path, "w", encoding="utf-8") as f: toml.dump(training_config, f)
    return str(config_path)


def get_latest_images(sample_dir):
    if not sample_dir.exists(): return []
    images = glob.glob(str(sample_dir / "*.png")) + glob.glob(str(sample_dir / "*.jpg")) + glob.glob(str(sample_dir / "*.webp"))
    images.sort(key=os.path.getmtime, reverse=True)
    
    return [(img, Path(img).name) for img in images[:MAX_GALLERY_IMAGES]]


# ==========================================
# CROPPER CLASS (U2Net Head-First + Buckets)
# ==========================================
class SmartCropper:
    def __init__(self) -> None:
        self.session: rt.InferenceSession | None = None
        self.model_path: Path = ROOT / "models" / "u2net" / "u2net.onnx"

    def load_model(self) -> str:
        if self.session is not None: 
            return "Already loaded"
        providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        self.session = rt.InferenceSession(str(self.model_path), providers=providers)
        return "GPU" if "CUDA" in self.session.get_providers()[0] else "CPU"

    def get_valid_buckets(self, side_min: int, side_max: int) -> List[Tuple[int, int]]:
        s_min, s_max = int(side_min), int(side_max)
        buckets = {(s_min, s_min)} 
        for s in range(s_min + 64, s_max + 64, 64):
            buckets.add((s_min, s))
            buckets.add((s, s_min))
        return sorted(list(buckets), key=lambda x: (x[0] * x[1]))

    def get_best_bucket(self, w: int, h: int, buckets: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], str]:
        orig_ar = w / h
        log_orig = math.log(orig_ar)
        best_b = buckets[0]
        min_diff = float('inf')
        
        for b in buckets:
            b_ar = b[0] / b[1]
            diff = abs(math.log(b_ar) - log_orig)
            if diff < min_diff:
                min_diff = diff
                best_b = b
        
        reason = f"AR: {orig_ar:.2f} -> {best_b[0]/best_b[1]:.2f}"
        return best_b, reason

    def process_image(self, original_img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h_orig, w_orig = original_img.shape[:2]
        
        if abs((w_orig / h_orig) - (tw / th)) < 0.01:
            return cv2.resize(original_img, (tw, th), interpolation=cv2.INTER_AREA)

        low_res_scale = 1024 / max(h_orig, w_orig)
        img_sm = cv2.resize(original_img, (int(w_orig*low_res_scale), int(h_orig*low_res_scale)), interpolation=cv2.INTER_AREA)
        
        input_size = 320
        img_inp = cv2.resize(img_sm, (input_size, input_size), interpolation=cv2.INTER_AREA)
        img_inp = img_inp.astype(np.float32) / 255.0
        img_inp -= [0.485, 0.456, 0.406]
        img_inp /= [0.229, 0.224, 0.225]
        input_tensor = np.expand_dims(np.transpose(img_inp, (2, 0, 1)), 0)
        
        mask = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0][0][0]
        mask = cv2.resize(mask, (img_sm.shape[1], img_sm.shape[0]))
        
        y_idx, x_idx = np.where(mask > 0.15)
        if len(y_idx) > 0:
            top_y = int(np.min(y_idx) / low_res_scale)
            center_x = int(np.mean(x_idx) / low_res_scale)
        else:
            top_y, center_x = h_orig // 4, w_orig // 2

        scale = max(tw / w_orig, th / h_orig)
        cw, ch = int(tw / scale), int(th / scale)
        y1 = max(0, min(top_y - int(ch * 0.05), h_orig - ch))
        x1 = max(0, min(center_x - cw // 2, w_orig - cw))

        return cv2.resize(original_img[y1:y1+ch, x1:x1+cw], (tw, th), interpolation=cv2.INTER_AREA)

    def unload_model(self) -> None:
        self.session = None

smart_cropper = SmartCropper()

def generate_bucket_summary(bucket_counts: dict) -> List[str]:
    """Generates a nicely formatted text report on image distribution."""
    if not bucket_counts:
        return []
        
    lines = ["", "📊 Bucket Distribution Summary:"]
    total = 0
    # Sort keys (WxH tuples) by area for pretty printing
    for bkt in sorted(bucket_counts.keys(), key=lambda x: x[0] * x[1]):
        count = bucket_counts[bkt]
        total += count
        lines.append(f"  ├─ {bkt[0]}x{bkt[1]}: {count} images")
    lines.append(f"  └─ Total Dataset Size: {total} images")
    lines.append("")
    return lines

def run_smart_crop_ui(dataset_dir: str, side_min: float, side_max: float, current_logs: str) -> Generator[str, None, None]:
    log_lines = current_logs.split('\n') if current_logs else []
    path = Path(dataset_dir)
    
    if not dataset_dir or not path.exists():
        log_lines.append("❌ Error: Dataset path invalid!")
        yield "\n".join(log_lines)
        return

    backup_dir = path / "original_images"
    backup_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    available_buckets = smart_cropper.get_valid_buckets(int(side_min), int(side_max))
    log_lines.append(f"📋 Buckets configured: {len(available_buckets)}")
    yield "\n".join(log_lines)

    # 1. Move new original images
    for f in path.glob('*'):
        if f.is_file() and f.suffix.lower() in valid_exts and f.parent != backup_dir:
            try:
                if f.suffix.lower() == '.png':
                    with Image.open(f) as img:
                        if img.size in available_buckets:
                            continue
            except Exception: pass
            shutil.move(str(f), str(backup_dir / f.name))

    # 2. Task preparation (Idempotency + Statistics collection)
    originals = [f for f in backup_dir.glob('*') if f.suffix.lower() in valid_exts]
    tasks = []
    bucket_counts = {}
    skipped = 0

    for orig in originals:
        try:
            with Image.open(orig) as img:
                w, h = img.size
            (tw, th), reason = smart_cropper.get_best_bucket(w, h, available_buckets)
            out_p = path / f"{orig.stem}.png"
            
            if out_p.exists():
                with Image.open(out_p) as check:
                    if check.size == (tw, th):
                        skipped += 1
                        bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                        continue
            tasks.append((orig, out_p, tw, th))
        except Exception: continue

    if not tasks:
        log_lines.append(f"✅ All {skipped} images are already bucketed correctly.")
        log_lines.extend(generate_bucket_summary(bucket_counts))
        yield "\n".join(log_lines)
        return

    # 3. Multi-threaded inference
    try:
        log_lines.append(f"🚀 Model: {smart_cropper.load_model()}")
        log_lines.append(f"⚙️ Processing {len(tasks)} images (Skipped: {skipped})...")
        yield "\n".join(log_lines)

        processed = 0
        lock = threading.Lock()

        def worker(task_data: Tuple[Path, Path, int, int]) -> str | None:
            in_p, out_p, tw, th = task_data
            nonlocal processed
            try:
                img = cv2.imread(str(in_p))
                if img is not None:
                    res = smart_cropper.process_image(img, tw, th)
                    cv2.imwrite(str(out_p), res, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                
                with lock:
                    processed += 1
                    # Add successful crop to statistics
                    bucket_counts[(tw, th)] = bucket_counts.get((tw, th), 0) + 1
                return None
            except Exception as e:
                return f"⚠️ {in_p.name}: {str(e)}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, t) for t in tasks]
            
            for future in as_completed(futures):
                err = future.result()
                if err: log_lines.append(err)
                
                if processed % 5 == 0 or processed == len(tasks):
                    msg = f"🔄 Progress: {processed}/{len(tasks)}"
                    if "Progress:" in log_lines[-1]: log_lines[-1] = msg
                    else: log_lines.append(msg)
                    yield "\n".join(log_lines)

        log_lines.append(f"✅ Done! Processed: {processed}")
        
    except Exception as e:
        log_lines.append(f"❌ Error: {str(e)}")
    finally:
        smart_cropper.unload_model()
        log_lines.extend(generate_bucket_summary(bucket_counts))
        yield "\n".join(log_lines)


# ==========================================
# TAGGER (animetimm dbv4-huge)
# ==========================================
# Optional offline copy of the model repo; when it is not there the model comes
# from the Hugging Face cache (downloaded on first use).
TAGGER_LOCAL_DIR = ROOT / "models" / "convnextv2_huge.dbv4-full"
# Images per forward pass. ConvNeXtV2-huge at 512px in bf16 stays well inside
# 6 GB with 4; decoding runs in parallel threads meanwhile.
TAGGER_BATCH = 4

db_tagger = DbTagger(TAGGER_LOCAL_DIR)
aesthetic_scorer = AestheticScorer()


def _replace_progress(log_lines, prefix, msg):
    """Keep a single, updating progress line in the log box."""
    if log_lines and log_lines[-1].startswith(prefix): log_lines[-1] = msg
    else: log_lines.append(msg)


def _explain_model_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if any(s in text for s in ("Gated", "gated", "401", "403", "RepositoryNotFound")):
        text += ("\n   The tagger repo needs a Hugging Face login: accept the terms at "
                 "https://huggingface.co/animetimm/convnextv2_huge.dbv4-full, then run "
                 "`uv run hf auth login` (or copy the repo files into models/convnextv2_huge.dbv4-full).")
    if isinstance(exc, ImportError):
        text += "\n   Missing packages: run Install_Requirements.bat (uv sync)."
    return text

# ==========================================
# GUI EXECUTOR FUNCTIONS
# ==========================================

def run_auto_tagging(dataset_dir, gen_floor, char_floor, blacklist_text, add_safety, overwrite, current_logs):
    log_lines = current_logs.split('\n') if current_logs else []

    if not dataset_dir or not os.path.exists(dataset_dir):
        log_lines.append("❌ Tagger Error: Dataset path invalid!")
        yield "\n".join(log_lines)
        return

    all_files = list_images(Path(dataset_dir))

    image_files = []
    skipped = 0
    for f in all_files:
        if f.with_suffix('.txt').exists() and not overwrite:
            skipped += 1
        else:
            image_files.append(f)

    if not image_files and skipped == 0:
        log_lines.append("❌ Tagger Error: No images found!")
        yield "\n".join(log_lines)
        return

    if not image_files and skipped > 0:
        log_lines.append(f"ℹ️ All images already have captions. Skipped: {skipped}")
        yield "\n".join(log_lines)
        return

    blacklist = TagBlacklist(blacklist_text)
    log_lines.append("🏷️ Loading tagger animetimm/convnextv2_huge.dbv4-full...")
    if blacklist:
        log_lines.append(f"🚫 Never-apply list: {', '.join(blacklist.patterns)}")
    yield "\n".join(log_lines)

    try:
        mode = db_tagger.load()
        log_lines.append(f"🚀 Model loaded using: {mode}")
        log_lines.append(f"⚙️ Processing {len(image_files)} images... (Skipped: {skipped})")
        yield "\n".join(log_lines)

        processed_count = 0
        blocked_count = 0
        total_to_process = len(image_files)
        batches = [image_files[i:i + TAGGER_BATCH] for i in range(0, total_to_process, TAGGER_BATCH)]

        # Decoding/preprocessing runs in threads one batch ahead of the GPU.
        with ThreadPoolExecutor(max_workers=4) as executor:
            pending = [executor.submit(db_tagger.prepare, p) for p in batches[0]]
            for b, batch in enumerate(batches):
                current = pending
                pending = [executor.submit(db_tagger.prepare, p) for p in batches[b + 1]] if b + 1 < len(batches) else []

                ready_paths, tensors = [], []
                for img_path, future in zip(batch, current):
                    try:
                        tensors.append(future.result())
                        ready_paths.append(img_path)
                    except Exception as e:
                        log_lines.append(f"⚠️ Error {img_path.name}: {str(e)}")

                if tensors:
                    try:
                        probs = db_tagger.predict(tensors)
                    except Exception as e:
                        log_lines.append(f"⚠️ Error in batch ({', '.join(p.name for p in ready_paths)}): {str(e)}")
                        probs = []
                    for img_path, p in zip(ready_paths, probs):
                        caption, blocked = db_tagger.caption(p, gen_floor, char_floor, add_safety, blacklist)
                        blocked_count += blocked
                        with open(img_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                            f.write(caption)
                        processed_count += 1

                _replace_progress(log_lines, "🔄 Tag Progress:", f"🔄 Tag Progress: {processed_count}/{total_to_process} images.")
                yield "\n".join(log_lines)

        log_lines.append(f"✅ Tagging complete! Processed: {processed_count} | Skipped: {skipped}")
        if blocked_count:
            log_lines.append(f"🚫 Left out {blocked_count} tag(s) from the never-apply list.")

    except Exception as e:
        log_lines.append(f"❌ Critical Tagger Error: {_explain_model_error(e)}")

    finally:
        db_tagger.unload()
        log_lines.append("🧹 VRAM cleared.")
        yield "\n".join(log_lines)


# ==========================================
# AESTHETIC QUALITY FILTER
# ==========================================
def _drive(steps, render):
    """Runs a step generator, yielding render() after every step; returns the
    generator's return value (lets the UI tools and Start Training share it)."""
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value
        yield render()


def apply_quality_filter(dataset_dir: Path, images: List[Path], keep_labels, mode: str,
                         log_lines: List[str], always_score: bool = False):
    """Step generator. Scores the images (cached in aesthetic_scores.json), then
    ignores or moves the ones whose label is not in keep_labels.

    Returns the images left to train on, or None if the selection is invalid.
    """
    keep = [label for label in QUALITY_LABELS if label in (keep_labels or [])]
    filtering = mode in (QUALITY_MODE_IGNORE, QUALITY_MODE_MOVE) and len(keep) < len(QUALITY_LABELS)
    if mode != QUALITY_MODE_OFF and not keep:
        log_lines.append("❌ Quality filter: select at least one label to keep.")
        yield
        return None
    if not filtering and not always_score:
        return images

    cache = ScoreCache(dataset_dir)
    missing = [p for p in images if cache.get(p) is None]
    if missing:
        log_lines.append(f"🎨 Aesthetic scoring {len(missing)} image(s) (deepghs anime_aesthetic; {len(images) - len(missing)} cached)...")
        yield
        try:
            aesthetic_scorer.load()
            for i, img_path in enumerate(missing, 1):
                try:
                    cache.put(img_path, aesthetic_scorer.percentile(img_path))
                except Exception as e:
                    log_lines.append(f"⚠️ {img_path.name}: {e}")
                if i % 5 == 0 or i == len(missing):
                    _replace_progress(log_lines, "🔄 Score Progress:", f"🔄 Score Progress: {i}/{len(missing)} images.")
                    yield
        except Exception as e:
            log_lines.append(f"❌ Aesthetic model error: {_explain_model_error(e)}")
            yield
            return None
        finally:
            aesthetic_scorer.unload()
            cache.save()

    labels = {}
    for p in images:
        percentile = cache.get(p)
        if percentile is not None:
            labels[p] = percentile_to_label(percentile)

    counts = {label: 0 for label in QUALITY_LABELS}
    for label in labels.values():
        counts[label] += 1
    log_lines.append("📊 Aesthetic labels (percentile on Danbooru's score distribution):")
    for label in QUALITY_LABELS:
        mark = "✓" if label in keep else "✗"
        log_lines.append(f"  {mark} {label:<11} >= {QUALITY_FLOORS[label]:.2f}: {counts[label]} images")
    unscored = [p for p in images if p not in labels]
    if unscored:
        log_lines.append(f"⚠️ {len(unscored)} image(s) could not be scored and are kept.")

    rejected = [p for p in images if p in labels and labels[p] not in keep]
    kept = [p for p in images if p not in rejected]
    if not filtering:
        log_lines.append("ℹ️ Filter is Off: nothing ignored or moved.")
    elif not rejected:
        log_lines.append(f"✅ All {len(images)} images are within the selected labels.")
    elif mode == QUALITY_MODE_MOVE:
        moved_files = 0
        for p in rejected:
            try:
                moved_files += move_out_of_dataset(dataset_dir, p, labels[p])
            except Exception as e:
                log_lines.append(f"⚠️ Could not move {p.name}: {e}")
                kept.append(p)
        kept = sorted(kept)
        log_lines.append(f"📦 Moved {len(images) - len(kept)} image(s) out of the dataset ({moved_files} files "
                         f"with captions/originals) to {dataset_dir / FILTERED_DIR}")
    else:
        log_lines.append(f"🙈 {len(rejected)} image(s) outside the selected labels will be ignored in training "
                         f"({len(kept)} left).")
    yield
    return kept


def run_quality_filter_ui(dataset_dir, keep_labels, mode, current_logs):
    log_lines = current_logs.split('\n') if current_logs else []
    if not dataset_dir or not os.path.isdir(dataset_dir):
        log_lines.append("❌ Error: Dataset path invalid!")
        yield "\n".join(log_lines)
        return
    images = list_images(Path(dataset_dir))
    if not images:
        log_lines.append("❌ Error: No images found in the dataset path.")
        yield "\n".join(log_lines)
        return
    yield from _drive(
        apply_quality_filter(Path(dataset_dir), images, keep_labels, mode, log_lines, always_score=True),
        lambda: "\n".join(log_lines),
    )
    yield "\n".join(log_lines)


def restore_quality_filter_ui(dataset_dir, current_logs):
    log_lines = current_logs.split('\n') if current_logs else []
    if not dataset_dir or not os.path.isdir(dataset_dir):
        log_lines.append("❌ Error: Dataset path invalid!")
        return "\n".join(log_lines)
    restored, conflicts = restore_moved(Path(dataset_dir))
    if restored or conflicts:
        log_lines.append(f"↩️ Restored {restored} file(s) from {FILTERED_DIR} to the dataset.")
    else:
        log_lines.append("ℹ️ Nothing to restore.")
    if conflicts:
        log_lines.append(f"⚠️ Left {len(conflicts)} file(s) in {FILTERED_DIR}: a file with the same name is back "
                         f"in its place ({', '.join(conflicts[:5])}).")
    return "\n".join(log_lines)


def open_dataset_folder_ui(dataset_dir, current_logs):
    log_lines = current_logs.split('\n') if current_logs else []
    if not dataset_dir or not os.path.exists(dataset_dir):
        log_lines.append("❌ Error: Dataset path invalid or empty!")
        return "\n".join(log_lines)
    
    target_dir = Path(dataset_dir)
    if platform.system() == "Windows": os.startfile(target_dir)
    else: subprocess.Popen(["xdg-open", str(target_dir)])
    
    log_lines.append(f"📁 Opened Dataset folder.")
    return "\n".join(log_lines)


# ==========================================
# TRAINING SESSION (decoupled from the web UI)
# ==========================================
# The trainer subprocess is NOT read from inside the Gradio generator anymore.
#
# Old behaviour (the freeze the user reported):
#   the generator did `for line in iter(proc.stdout.readline, "")` and yielded the
#   whole log on every single line. The generator only advances while the browser
#   keeps consuming it, so as soon as the tab was hidden/reloaded/slow, or the
#   Gradio queue pushed back, nothing drained the child's stdout pipe. The pipe
#   buffer (64 KB on Windows) fills up, the child blocks forever inside write()
#   and the step counter just stops - no traceback, no error, GPU idle.
#   On top of that the global `training_process` was set to None on generator
#   exit, so the zombie became unreachable and the next "Start Training" happily
#   stacked a second job on the same GPU.
#
# New behaviour:
#   a dedicated daemon thread always drains stdout into a rolling buffer, while
#   the Gradio generator only *polls* that buffer. Killing the browser tab can no
#   longer stall training, and reloading the page re-attaches to the live session.

TRAIN_SESSION = None


def _kill_process_tree(pid):
    """Kill a process together with its children (taskkill /T on Windows)."""
    if not pid:
        return
    if platform.system() == "Windows":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
                timeout=20,
            )
            return
        except Exception:
            pass  # fall through to psutil
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try: child.kill()
            except psutil.NoSuchProcess: pass
        try: parent.kill()
        except psutil.NoSuchProcess: pass
    except psutil.NoSuchProcess:
        pass


def find_stray_trainers() -> List[psutil.Process]:
    """Leftover trainer processes (accelerate -> anima_train_network.py).

    A leftover silently keeps VRAM occupied and makes the next run look 2x slower
    or freeze outright, which is why stale processes have to be cleaned up.
    """
    strays = []
    own_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if proc.pid == own_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(cmdline)
            if "anima_train_network.py" in joined or "anima_train.py" in joined:
                strays.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return strays


class TrainingSession:
    """Owns the trainer subprocess and its log buffer, independent of the UI."""

    def __init__(self, sample_dir: Path):
        self.sample_dir = sample_dir
        self.process = None
        self.log_lines = deque(maxlen=MAX_LOG_LINES)
        self.log_file = None
        self.stopping = False
        self.returncode = None
        self.started_at = time.time()
        self.last_output_at = time.time()
        self._lock = threading.Lock()
        self._reader = None

    # ---------- state ----------
    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def log(self, line: str):
        self._push(line)

    def _push(self, line: str, to_file: bool = True):
        with self._lock:
            self.last_output_at = time.time()
            # drop a trailing "still alive" marker, real output replaced it
            while self.log_lines and self.log_lines[-1].startswith("⏳"):
                self.log_lines.pop()
            # tqdm writes one line per step: keep only the newest one in the box
            if "steps:" in line and STEP_RE.search(line) and self.log_lines and self.log_lines[-1].startswith("steps:"):
                self.log_lines[-1] = line
            else:
                self.log_lines.append(line)
            if to_file and self.log_file is not None:
                try:
                    self.log_file.write(line + "\n")
                    self.log_file.flush()
                except Exception:
                    pass

    def snapshot(self):
        """Return (log text, gallery images) for the current state."""
        now = time.time()
        silent_for = now - self.last_output_at
        with self._lock:
            if self.is_alive() and silent_for > STALL_WARN_SECONDS:
                msg = (
                    f"⏳ Trainer is alive, no output for {int(silent_for // 60)} min "
                    f"(sampling / saving / slow step) - still working"
                )
                while self.log_lines and self.log_lines[-1].startswith("⏳"):
                    self.log_lines.pop()
                self.log_lines.append(msg)
            log_text = "\n".join(self.log_lines)
        return log_text, get_latest_images(self.sample_dir)

    # ---------- lifecycle ----------
    def start(self, cmd, env, cwd, log_path: Path, first_lines: List[str]):
        with self._lock:
            self.log_lines.extend(first_lines)
        try:
            self.log_file = open(log_path, "a", encoding="utf-8", errors="ignore")
            self.log_file.write("\n\n===== run %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            for ln in first_lines:
                self.log_file.write(ln + "\n")
            self.log_file.flush()
        except Exception:
            self.log_file = None

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1,
            cwd=str(cwd),
            env=env,
            encoding="utf-8",
            errors="ignore",
        )
        self.started_at = self.last_output_at = time.time()
        self._reader = threading.Thread(target=self._read_loop, name="trainer-stdout", daemon=True)
        self._reader.start()

    def _read_loop(self):
        """Always drain stdout. Blocks on readline, never on the browser."""
        proc = self.process
        try:
            for raw in iter(proc.stdout.readline, ""):
                line = raw.replace("\r", "").strip()
                if not line:
                    continue
                if any(skip_word in line for skip_word in LOG_BLACKLIST):
                    continue
                self._push(line)
        except Exception as exc:
            self._push(f"❌ Log reader error: {exc}")
        finally:
            try: proc.stdout.close()
            except Exception: pass
            try:
                self.returncode = proc.wait(timeout=10)
            except Exception:
                self.returncode = proc.poll()
            with self._lock:
                self.last_output_at = time.time()

    def stop(self) -> str:
        if self.process is None:
            return "ℹ️ Not running."
        if not self.is_alive():
            self.cleanup_files()
            return "ℹ️ Process already finished."
        self.stopping = True
        self.log("🛑 Stopping training... (clearing VRAM)")
        _kill_process_tree(self.process.pid)
        try:
            self.process.wait(timeout=20)
        except Exception:
            pass
        # make sure nothing survived (accelerate children / dataloader workers)
        for stray in find_stray_trainers():
            _kill_process_tree(stray.pid)
        self.cleanup_files()
        self.log("🛑 Training stopped. VRAM should be free now.")
        return "🛑 Training stopped. VRAM should be free now."

    def cleanup_files(self):
        with self._lock:
            if self.log_file is not None:
                try: self.log_file.close()
                except Exception: pass
                self.log_file = None


@atexit.register
def _shutdown_training_session():
    """Never leave an orphaned trainer behind that keeps the GPU busy."""
    if TRAIN_SESSION is not None and TRAIN_SESSION.is_alive():
        try:
            TRAIN_SESSION.stop()
        except Exception:
            pass


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def start_training(trigger_word, dataset_path, dit_p, qwen_p, vae_p, rank, lr, optimizer, t_steps, save_steps, sample_steps, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, train_seed, batch_size, grad_acc, blacklist_text, quality_keep, quality_mode):
    global TRAIN_SESSION

    # Re-attach FIRST: the page may have been reloaded while a run is going on, and
    # in that case the form values must not be re-validated or re-submitted.
    if TRAIN_SESSION is not None and TRAIN_SESSION.is_alive():
        # The page was reloaded / the tab was switched while training kept going.
        # Re-attach to the live session instead of refusing or, worse, starting a
        # second job on the same GPU.
        session = TRAIN_SESSION
        session.log("🔁 Re-attached to the training that is already running.")
        last_text, last_key = None, None
        while session.is_alive():
            text, images = session.snapshot()
            key = tuple(name for _, name in images)
            if text != last_text or key != last_key:
                last_text, last_key = text, key
                yield text, (images if key else gr.update())
            time.sleep(UI_POLL_SECONDS)
        text, images = session.snapshot()
        yield text, images
        return

    model_errors = []
    if not dit_p or not os.path.isfile(dit_p): model_errors.append(f"❌ DiT file not found: {dit_p}")
    if not qwen_p or not os.path.isfile(qwen_p): model_errors.append(f"❌ Qwen3 file not found: {qwen_p}")
    if not vae_p or not os.path.isfile(vae_p): model_errors.append(f"❌ VAE file not found: {vae_p}")
    
    dataset_errors = []
    image_files = []
    if not dataset_path or not os.path.isdir(dataset_path):
        dataset_errors.append(f"❌ Dataset path not found: {dataset_path}")
    else:
        image_files = list_images(Path(dataset_path))
        if not image_files:
            dataset_errors.append(f"❌ No valid images found in the dataset path.")

    if model_errors or dataset_errors:
        full_error = ""
        if model_errors:
            full_error += "\n".join(model_errors)
            full_error += "\n\n⚠️ ERROR: Please check and set the correct model paths in the section:\n'🔧 Paths to Models <- Set Once'\n\n"
        if dataset_errors:
            full_error += "\n".join(dataset_errors)
            
        yield full_error.strip(), gr.update()
        return

    if not torch.cuda.is_available():
        cuda_error = (
            "❌ ERROR: NVIDIA GPU not detected or CUDA drivers are not installed!\n\n"
            "Technical details:\n"
            "- PyTorch cannot initialize CUDA.\n"
            "- Training on CPU is extremely slow and is not supported by this script.\n\n"
            "Please update your NVIDIA drivers and restart the app."
        )
        yield cuda_error, gr.update()
        return

    trigger = trigger_word.strip()
    project_name = re.sub(r'[^a-zA-Z0-9]', '_', trigger).strip('_') or "untitled"
    dataset_dir = Path(dataset_path)

    log_lines = [f"🚀 Preparing: {project_name}..."]
    render = lambda: ("\n".join(log_lines), gr.update())

    yield render()

    # 1. Aesthetic quality filter (no-op when Off)
    image_files = yield from _drive(
        apply_quality_filter(dataset_dir, image_files, quality_keep, quality_mode, log_lines), render)
    if image_files is None:
        yield render()
        return
    if not image_files:
        log_lines.append("❌ No images left to train on after the quality filter.")
        yield render()
        return

    # 2. Captions and sizes of exactly the images that will be trained
    log_lines.append(f"🔍 Analyzing {len(image_files)} dataset images...")
    yield render()
    blacklist = TagBlacklist(blacklist_text)
    report = collect_training_entries(image_files, trigger, blacklist)
    if report.errors:
        log_lines.extend(report.errors)
        yield render()
        return
    entries = report.entries

    project_out_dir = OUTPUT_BASE / project_name
    sample_dir = project_out_dir / "sample"
    project_configs_dir = project_out_dir / "configs"

    for d in [project_out_dir, sample_dir, project_configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # A leftover trainer from a killed run keeps VRAM busy and makes the new run
    # freeze or crawl, so clean the GPU before starting.
    strays = find_stray_trainers()
    if strays:
        pids = ", ".join(str(p.pid) for p in strays)
        log_lines.append(f"🧹 Killing leftover trainer process(es) from a previous run: {pids}")
        yield render()
        for stray in strays:
            _kill_process_tree(stray.pid)
        time.sleep(3)

    base_res, max_bucket = analyze_dataset_resolution([e.size for e in entries])
    log_lines.append(f"📐 Auto-Resolution Set: Base {base_res}px, Max Bucket {max_bucket}px")

    # 3. The captions sd-scripts will encode: "<trigger>, <tags>"
    metadata_path = project_configs_dir / f"{project_name}_captions.json"
    write_caption_metadata(metadata_path, entries, trigger)
    log_lines.append(f"📝 Captions for {len(entries)} images"
                     + (f", trigger '{trigger}' first" if trigger else " (no trigger word)")
                     + f" -> {metadata_path.name}")
    if report.blacklisted:
        log_lines.append(f"🚫 Removed {report.blacklisted} tag(s) of the never-apply list from the captions.")
    if report.empty:
        log_lines.append(f"⚠️ {len(report.empty)} caption(s) are empty" + (" (trigger word only)." if trigger else "."))

    # 4. Drop cached text encoder outputs / latents that no longer match
    te_fingerprint = f"qwen={model_fingerprint(qwen_p)}|dropout={HIDDEN_SETTINGS['caption_dropout_rate']}|v1"
    latent_fingerprint = f"vae={model_fingerprint(vae_p)}|v1"
    try:
        te_dropped, latent_dropped = refresh_disk_caches(
            dataset_dir, {e.image: join_caption(trigger, e.tags) for e in entries}, te_fingerprint, latent_fingerprint)
        if te_dropped or latent_dropped:
            log_lines.append(f"♻️ Captions/images changed since the last cache: re-encoding {te_dropped} caption(s)"
                             f" and {latent_dropped} latent(s).")
    except Exception as e:
        log_lines.append(f"⚠️ Could not check the disk caches ({e}); delete the cache_text_encoder folder by hand if captions changed.")

    models = {"dit_path": dit_p, "qwen_path": qwen_p, "vae_path": vae_p}
    prompt_path = create_sample_prompts(project_name, trigger_word, pos, neg, w, h, s_steps_gen, s_cfg, s_seed, project_configs_dir)
    dataset_toml = create_dataset_toml(project_name, dataset_path, metadata_path, len(entries), base_res, max_bucket, project_configs_dir, t_steps, batch_size, grad_acc)
    training_toml = create_training_toml(project_name, project_configs_dir, project_out_dir, rank, lr, optimizer, t_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc)

    cmd = [
        str(PYTHON_EXE), "-m", "accelerate.commands.launch", "--num_processes=1", "--mixed_precision=bf16", "--dynamo_backend=no",
        TRAIN_SCRIPT.resolve().as_posix(), 
        "--config_file", Path(training_toml).resolve().as_posix(), 
        "--dataset_config", Path(dataset_toml).resolve().as_posix()
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRAIN_DIR.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    env["KMP_WARNINGS"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["ACCELERATE_USE_CPU"] = "False"

    session = TrainingSession(sample_dir)
    TRAIN_SESSION = session

    log_lines.append("⏳ Launching trainer... (full log is also written to train.log in the checkpoint folder)")

    try:
        session.start(cmd, env, TRAIN_DIR, project_out_dir / "train.log", log_lines)
    except Exception as e:
        session.log(f"❌ Could not start the trainer: {e}")
        session.cleanup_files()
        yield session.snapshot()
        return

    # Poll the buffer instead of driving the training loop from this request:
    # the browser can now be closed, hidden or reloaded without stalling training.
    last_text, last_key = None, None
    while session.is_alive():
        text, images = session.snapshot()
        key = tuple(name for _, name in images)
        if text != last_text or key != last_key:
            last_text, last_key = text, key
            yield text, (images if key else gr.update())
        time.sleep(UI_POLL_SECONDS)

    session.cleanup_files()
    returncode = session.process.poll() if session.process is not None else None
    if session.stopping:
        pass  # stop() already logged the message
    elif returncode == 0:
        session.log("✅ Training finished. LoRA saved to the checkpoint folder.")
    else:
        session.log(
            f"❌ Training process exited with code {returncode}. Check the log above "
            f"or {project_out_dir / 'train.log'} for details."
        )
    text, images = session.snapshot()
    yield text, images


def stop_training():
    if TRAIN_SESSION is not None and TRAIN_SESSION.is_alive():
        return TRAIN_SESSION.stop()

    # Nothing tracked, but a zombie from a crashed/reloaded run may still hold VRAM.
    strays = find_stray_trainers()
    for stray in strays:
        _kill_process_tree(stray.pid)
    if strays:
        return f"🧹 Killed {len(strays)} leftover trainer process(es). VRAM should be free."
    return "ℹ️ Not running."

def open_output_folder(trigger_word):
    proj = re.sub(r'[^a-zA-Z0-9]', '_', trigger_word.strip()).strip('_')
    target_dir = OUTPUT_BASE / proj if proj else OUTPUT_BASE
    if not target_dir.exists(): target_dir = OUTPUT_BASE
    if platform.system() == "Windows": os.startfile(target_dir)
    else: subprocess.Popen(["xdg-open", str(target_dir)])
    return "📁 Folder opened."

def handle_optimizer_change(opt, current_lr, saved_adam_lr):
    if opt == "Prodigy": return "1.0", current_lr
    return (saved_adam_lr if current_lr == "1.0" else current_lr), saved_adam_lr


# ==========================================
# UI BUILDER
# ==========================================
cs = load_settings()

with gr.Blocks(title="Anima TrainFlow: Easy LoRA Trainer for Anima 2B") as ui:
    gr.Markdown(
        "# Anima TrainFlow\n"
        '<div class="attribution"><span class="author-text">Created by ThetaCursed</span></div>',
        elem_id="main-header"
    )
    
    saved_adam_lr = gr.State(value=ADAMW_DEFAULT_LR)

    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    trigger_word = gr.Textbox(label="Trigger Word / Project Name", value=cs.get("trigger_word", ""), placeholder="e.g., unique_style")
                    dataset_path = gr.Textbox(label="Dataset Path (Images + .txt)", value=cs.get("dataset_path", ""), placeholder="C:/Images/MyDataset")
                with gr.Accordion("🔧 Paths to Models <- Set Once", open=False):
                    dit_input = gr.Textbox(label="DiT", value=cs.get("dit_path", ""))
                    qwen_input = gr.Textbox(label="Qwen3", value=cs.get("qwen_path", ""))
                    vae_input = gr.Textbox(label="VAE", value=cs.get("vae_path", ""))
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary")
                    stop_btn = gr.Button("🛑 Stop", variant="stop")
                    folder_btn = gr.Button("📁 Checkpoint Folder", variant="secondary")
            with gr.Column(scale=1):
                with gr.Row():
                    rank_input = gr.Number(label="Network Rank", value=cs.get("network_rank", 16), precision=0)
                    lr_input = gr.Textbox(label="Learning Rate", value=cs.get("learning_rate", ADAMW_DEFAULT_LR))
                    optimizer_input = gr.Dropdown(label="Optimizer", choices=["AdamW8bit", "AdamW", "Prodigy"], value=cs.get("optimizer", "AdamW8bit"))
                    train_seed_val = gr.Number(value=cs.get("train_seed", 42), visible=False)
                    batch_size_input = gr.Number(label="Batch Size", value=cs.get("train_batch_size", 1), precision=0)
                with gr.Row():
                    steps_input = gr.Number(label="Training Steps", value=cs.get("training_steps", 2400), precision=0)
                    save_steps_input = gr.Number(label="Save Every n Steps", value=cs.get("save_steps", 300), precision=0)
                    sample_steps_input = gr.Number(label="Preview Every n Steps", value=cs.get("sample_steps", 300), precision=0)
                    grad_acc_input = gr.Number(label="Gradient Accumulation", value=cs.get("gradient_accumulation_steps", 1), precision=0)

    with gr.Row():
        with gr.Column(scale=1):
            output_log = gr.Textbox(label="Logs", lines=LOG_BOX__MAX_LINES, max_lines=LOG_BOX__MAX_LINES, interactive=False, autoscroll=True, elem_id="log-container")
            
            with gr.Group():
                gr.Markdown("### Smart Aspect Ratio Bucketing")
                with gr.Row():
                    side_min_input = gr.Number(label="Min Side (Base)", value=cs.get("side_min", 512), precision=0)
                    side_max_input = gr.Number(label="Max Side (Limit)", value=cs.get("side_max", 768), precision=0)
                with gr.Row():
                    crop_btn = gr.Button("✂️ Start Bucketing", variant="secondary")
                    open_ds_btn = gr.Button("📂 Open Dataset", variant="secondary")
                    
            # --- AUTO CAPTION ---
            with gr.Group():
                gr.Markdown("### Auto-Caption Dataset (animetimm dbv4-huge)")
                with gr.Row():
                    tagger_gen_floor = gr.Slider(0.0, 1.0, value=cs.get("tagger_gen_floor", 0.0), step=0.01, label="General Tags: Min Confidence",
                                                 info="0 = the model's calibrated threshold of each tag")
                    tagger_char_floor = gr.Slider(0.0, 1.0, value=cs.get("tagger_char_floor", CHARACTER_FLOOR_DEFAULT), step=0.01, label="Character Tags: Min Confidence",
                                                  info="On top of the calibrated threshold")
                tagger_blacklist = gr.Textbox(label="Never Apply These Tags", lines=2, value=cs.get("tagger_blacklist", ""),
                                              placeholder="e.g. signature, watermark, artist name, *username, simple background",
                                              info="Comma separated, * = wildcard. Left out when tagging and removed from existing captions when training.")
                with gr.Row():
                    tagger_btn = gr.Button("Create .txt captions", variant="secondary")
                    tagger_overwrite = gr.Checkbox(label="Overwrite existing .txt", value=cs.get("tagger_overwrite", False))
                    tagger_add_safety = gr.Checkbox(label="Add safety tag (safe/sensitive/nsfw/explicit)", value=cs.get("tagger_add_safety", True))


        with gr.Column(scale=1):
            preview_gallery = gr.Gallery(label="Previews", columns=2, rows=2, height=GALLERY_HEIGHT, object_fit="contain")
            with gr.Group():
                pos_prompt = gr.Textbox(label="Prompt (Trigger word added automatically)", lines=2, value=cs.get("pos_prompt", ""),
                                        placeholder="e.g. masterpiece, best quality, score_7, safe, 1girl, solo, upper body")
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=cs.get("neg_prompt", ""))
                with gr.Row():
                    width_input = gr.Number(label="Width", value=cs.get("width", 1024), precision=0, min_width=80)
                    height_input = gr.Number(label="Height", value=cs.get("height", 1024), precision=0, min_width=80)
                    sample_steps_gen_input = gr.Number(label="Steps", value=cs.get("sample_steps_gen", 30), precision=0, min_width=80)
                    sample_cfg_input = gr.Number(label="CFG", value=cs.get("sample_cfg", 4.0), min_width=80)
                    sample_seed_input = gr.Number(label="Seed", value=cs.get("sample_seed", 42), precision=0, min_width=80)

            # --- AESTHETIC QUALITY FILTER ---
            with gr.Group():
                gr.Markdown("### Aesthetic Quality Filter (deepghs anime_aesthetic)")
                quality_keep = gr.CheckboxGroup(
                    choices=QUALITY_LABELS, value=cs.get("quality_keep", list(QUALITY_LABELS)), label="Keep images rated",
                    info="Percentile on Danbooru's score distribution: masterpiece ≥ 95%, best ≥ 85%, great ≥ 75%, "
                         "good ≥ 50%, normal ≥ 25%, low ≥ 10%, worst < 10%")
                quality_mode = gr.Radio(
                    choices=QUALITY_MODES, value=cs.get("quality_mode", QUALITY_MODE_OFF), label="Images outside the selection",
                    info=f"Applied when training starts. 'Move' puts them in the dataset's {FILTERED_DIR} folder (not trained, can be restored).")
                with gr.Row():
                    quality_btn = gr.Button("📊 Score & Apply Now", variant="secondary")
                    quality_restore_btn = gr.Button("↩️ Restore Moved Images", variant="secondary")

    training_inputs = [
        trigger_word, dataset_path, dit_input, qwen_input, vae_input,
        rank_input, lr_input, optimizer_input,
        steps_input, save_steps_input, sample_steps_input,
        pos_prompt, neg_prompt, width_input, height_input,
        sample_steps_gen_input, sample_cfg_input, sample_seed_input, train_seed_val, batch_size_input, grad_acc_input,
        tagger_blacklist, quality_keep, quality_mode,
    ]

    # Every persisted setting, by its key in settings.json. auto_save_state and
    # load_state_on_refresh pair values with keys by position, so the list is
    # built from DEFAULT_SETTINGS' order and checked against it.
    SETTINGS_COMPONENTS = {
        "trigger_word": trigger_word, "dataset_path": dataset_path,
        "dit_path": dit_input, "qwen_path": qwen_input, "vae_path": vae_input,
        "network_rank": rank_input, "learning_rate": lr_input, "optimizer": optimizer_input,
        "training_steps": steps_input, "save_steps": save_steps_input, "sample_steps": sample_steps_input,
        "pos_prompt": pos_prompt, "neg_prompt": neg_prompt, "width": width_input, "height": height_input,
        "sample_steps_gen": sample_steps_gen_input, "sample_cfg": sample_cfg_input, "sample_seed": sample_seed_input,
        "train_seed": train_seed_val, "train_batch_size": batch_size_input, "gradient_accumulation_steps": grad_acc_input,
        "side_min": side_min_input, "side_max": side_max_input,
        "tagger_gen_floor": tagger_gen_floor, "tagger_char_floor": tagger_char_floor,
        "tagger_blacklist": tagger_blacklist, "tagger_add_safety": tagger_add_safety, "tagger_overwrite": tagger_overwrite,
        "quality_keep": quality_keep, "quality_mode": quality_mode,
    }
    assert set(SETTINGS_COMPONENTS) == set(DEFAULT_SETTINGS), "SETTINGS_COMPONENTS and DEFAULT_SETTINGS differ"
    all_settings_list = [SETTINGS_COMPONENTS[k] for k in DEFAULT_SETTINGS]

    def load_state_on_refresh():
        current_settings = load_settings()
        return [current_settings.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS.keys()]

    ui.load(fn=load_state_on_refresh, inputs=None, outputs=all_settings_list)

    output_log.change(None, None, None, js=JS_SCROLL)
    optimizer_input.change(fn=handle_optimizer_change, inputs=[optimizer_input, lr_input, saved_adam_lr], outputs=[lr_input, saved_adam_lr])
    
    for comp in all_settings_list: 
        comp.change(fn=auto_save_state, inputs=all_settings_list)

    crop_btn.click(
        fn=run_smart_crop_ui,
        inputs=[dataset_path, side_min_input, side_max_input, output_log],
        outputs=[output_log]
    )
    
    open_ds_btn.click(
        fn=open_dataset_folder_ui,
        inputs=[dataset_path, output_log],
        outputs=[output_log]
    )
    
    tagger_btn.click(
        fn=run_auto_tagging,
        inputs=[dataset_path, tagger_gen_floor, tagger_char_floor, tagger_blacklist, tagger_add_safety, tagger_overwrite, output_log],
        outputs=[output_log]
    )

    quality_btn.click(
        fn=run_quality_filter_ui,
        inputs=[dataset_path, quality_keep, quality_mode, output_log],
        outputs=[output_log]
    )
    quality_restore_btn.click(
        fn=restore_quality_filter_ui,
        inputs=[dataset_path, output_log],
        outputs=[output_log]
    )

    start_btn.click(fn=start_training, inputs=training_inputs, outputs=[output_log, preview_gallery])
    stop_btn.click(fn=stop_training, outputs=output_log)
    folder_btn.click(fn=open_output_folder, inputs=[trigger_word], outputs=output_log)

if __name__ == "__main__":
     ui.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)