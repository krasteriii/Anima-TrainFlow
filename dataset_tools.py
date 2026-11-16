"""Dataset tools for Anima TrainFlow.

Kept out of app.py so the UI file stays about the UI and the trainer process:

* DbTagger - auto-captioning with animetimm/convnextv2_huge.dbv4-full (timm /
  PyTorch). Uses the per-tag calibrated thresholds shipped with the model and
  writes captions in the tag order Anima was trained on.
* AestheticScorer + ScoreCache - deepghs anime_aesthetic (via dghs-imgutils):
  every image gets its percentile on the Danbooru score distribution, which the
  quality filter buckets into worst .. masterpiece.
* Training captions - the "never apply" tag list, the caption sd-scripts really
  encodes (trigger word included) and the invalidation of stale disk caches.

Everything heavy (torch, timm, imgutils) is imported lazily, so importing this
module costs nothing until a tool is actually used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# sd-scripts refuses nothing, but anything this big must go through the
# Smart Aspect Ratio Bucketing tool first (VRAM / bucket limits).
MAX_IMAGE_SIDE = 2048


def list_images(folder: Path) -> list[Path]:
    """The training images: image files directly inside the dataset folder.

    Subfolders (original_images, latent_cache, cache_text_encoder, the quality
    filter folder) are never part of the dataset, same as for sd-scripts.
    """
    if not folder.is_dir():
        return []
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def file_signature(path: Path) -> str:
    """Cheap change detector for an image file (size + modification time)."""
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def flatten_rgb(image: Image.Image) -> Image.Image:
    """Transparency over white. convert('RGB') alone would turn it black."""
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


# ==========================================
# Tags
# ==========================================
# Kaomoji are drawn with their underscores, so they keep them. Anima's model
# card: "Use lowercase for tags, and spaces instead of underscores. Score tags
# are the only tags that use underscores."
KAOMOJIS = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}
SCORE_TAG_RE = re.compile(r"score_\d+")


def to_caption_tag(danbooru_name: str) -> str:
    """Danbooru tag name -> how Anima's captions spell it."""
    if danbooru_name in KAOMOJIS or SCORE_TAG_RE.fullmatch(danbooru_name):
        return danbooru_name
    return danbooru_name.replace("_", " ")


def unescape_tag(tag: str) -> str:
    r"""'hatsune miku \(vocaloid\)' -> 'hatsune miku (vocaloid)'.

    The backslash is prompt syntax of A1111/Forge/ComfyUI, which strip it before
    the text encoder sees the prompt. sd-scripts feeds Anima's text encoder the
    caption verbatim, so an escaped caption trains a literal backslash that no
    prompt will ever reproduce.
    """
    return tag.replace("\\(", "(").replace("\\)", ")")


def tag_key(tag: str) -> str:
    """Comparison key: case, underscores vs spaces and escapes don't matter."""
    return " ".join(unescape_tag(tag).replace("_", " ").lower().split())


class TagBlacklist:
    """Tags that must never end up in a caption.

    Entries are separated by commas or new lines and compared with tag_key(), so
    'artist_name' also blocks 'artist name'. '*' is a wildcard: '*username'
    blocks 'twitter username' and 'patreon username'. No other character is
    special, so tags such as '?' or ':>' can be listed as they are.
    """

    def __init__(self, text: str | None):
        self.patterns: list[str] = []
        self._regexes: list[re.Pattern] = []
        for raw in re.split(r"[,\n]", text or ""):
            key = tag_key(raw)
            if not key or key in self.patterns:
                continue
            self.patterns.append(key)
            self._regexes.append(re.compile(".*".join(re.escape(part) for part in key.split("*"))))

    def __bool__(self) -> bool:
        return bool(self._regexes)

    def blocks(self, tag: str) -> bool:
        key = tag_key(tag)
        return any(regex.fullmatch(key) for regex in self._regexes)


# ==========================================
# Auto-captioning: animetimm dbv4
# ==========================================
DBV4_REPO = "animetimm/convnextv2_huge.dbv4-full"
DBV4_FILES = ("config.json", "model.safetensors", "selected_tags.csv", "preprocess.json")
# Character tags calibrate low for rare characters and then hallucinate them
# (bench of the tagger project: P 0.94 / R 0.29 with this floor). The model
# card suggests ~0.5 for the whole category.
CHARACTER_FLOOR_DEFAULT = 0.50
# dbv4 rating -> Anima's safety tags (model card: safe, sensitive, nsfw, explicit)
RATING_TO_SAFETY = {"general": "safe", "sensitive": "sensitive", "questionable": "nsfw", "explicit": "explicit"}
# The "[1girl/1boy/1other etc]" section of Anima's tag order
COUNT_TAG_RE = re.compile(r"\d\+?(?:girls?|boys?|others?)|multiple (?:girls|boys|others)|no humans")


class DbTagger:
    """animetimm/convnextv2_huge.dbv4-full: 9225 general + 3247 character tags.

    Thresholds: every tag has its own `best_threshold` (the one that maximises
    that tag's F1 on the test split, "Macro@Best" in the model card). The UI
    floors are applied on top of it, never below.
    """

    def __init__(self, local_dir: Path):
        # Optional offline copy of the repo; otherwise the Hugging Face cache.
        self.local_dir = local_dir
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.dtype = None
        self.device_label = ""

    def model_dir(self) -> Path:
        if all((self.local_dir / f).is_file() for f in DBV4_FILES):
            return self.local_dir
        from huggingface_hub import snapshot_download

        # Cache first: works offline and skips the round trip to the Hub.
        try:
            folder = Path(snapshot_download(DBV4_REPO, allow_patterns=list(DBV4_FILES), local_files_only=True))
            if all((folder / f).is_file() for f in DBV4_FILES):
                return folder
        except Exception:
            pass
        return Path(snapshot_download(DBV4_REPO, allow_patterns=list(DBV4_FILES)))

    def load(self) -> str:
        if self.model is not None:
            return self.device_label
        import torch
        import timm
        from imgutils.preprocess import create_torchvision_transforms

        folder = self.model_dir()
        model = timm.create_model(f"local-dir:{folder.as_posix()}", pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 halves the VRAM (1.4 GB instead of 2.8 GB) with the same tags as
        # fp32. fp16 is NOT an option: ConvNeXtV2's GRN layers overflow to NaN.
        use_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        with open(folder / "preprocess.json", encoding="utf-8") as f:
            self.transform = create_torchvision_transforms(json.load(f)["test"])
        self._load_tags(folder / "selected_tags.csv")
        self.device_label = ("GPU" if self.device == "cuda" else "CPU (slow)") + (", bf16" if use_bf16 else ", fp32")
        return self.device_label

    def _load_tags(self, csv_path: Path) -> None:
        names, categories, thresholds = [], [], []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.append(row["name"])
                categories.append(int(row["category"]))
                thresholds.append(float(row["best_threshold"]))
        categories = np.array(categories)
        self.names = names
        self.thresholds = np.array(thresholds, dtype=np.float32)
        self.general_idx = np.flatnonzero(categories == 0)
        self.character_idx = np.flatnonzero(categories == 4)
        self.rating_idx = np.flatnonzero(categories == 9)

    def prepare(self, image_path: Path):
        """Decode + preprocess one image (thread-safe, runs in a worker pool)."""
        with Image.open(image_path) as img:
            rgb = flatten_rgb(img)
        return self.transform(rgb)

    def predict(self, tensors: list) -> np.ndarray:
        import torch

        batch = torch.stack(tensors).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            return torch.sigmoid(self.model(batch)).float().cpu().numpy()

    def caption(self, probs: np.ndarray, general_floor: float = 0.0,
                character_floor: float = CHARACTER_FLOOR_DEFAULT, add_safety: bool = True,
                blacklist: TagBlacklist | None = None) -> tuple[str, int]:
        """Caption in Anima's order: safety, people count, characters, general.

        (The card's full order is quality/meta/year/safety, 1girl/1boy/1other,
        character, series, artist, general; dbv4 predicts no series or artist.)
        Returns the caption and how many tags the blacklist removed.
        """

        def pick(indexes: np.ndarray, floor: float) -> list[str]:
            hit = indexes[probs[indexes] >= np.maximum(self.thresholds[indexes], floor)]
            hit = hit[np.argsort(-probs[hit], kind="stable")]  # most confident first
            return [to_caption_tag(self.names[i]) for i in hit]

        general = pick(self.general_idx, float(general_floor))
        characters = pick(self.character_idx, float(character_floor))
        tags = []
        if add_safety:
            rating = self.names[self.rating_idx[int(np.argmax(probs[self.rating_idx]))]]
            tags.append(RATING_TO_SAFETY.get(rating, rating))
        tags += [t for t in general if COUNT_TAG_RE.fullmatch(t)]
        tags += characters
        tags += [t for t in general if not COUNT_TAG_RE.fullmatch(t)]
        kept = [t for t in tags if not (blacklist and blacklist.blocks(t))]
        return ", ".join(kept), len(tags) - len(kept)

    def unload(self) -> None:
        if self.model is None:
            return
        import gc
        import torch

        self.model = None
        self.transform = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==========================================
# Quality filter: deepghs anime_aesthetic
# ==========================================
AESTHETIC_REPO = "deepghs/anime_aesthetic"
AESTHETIC_MODEL = "swinv2pv3_v0_448_ls0.2_x"  # imgutils' default
AESTHETIC_ID = f"{AESTHETIC_REPO}/{AESTHETIC_MODEL}"
# Best label first. Floors are percentiles of the Danbooru score distribution,
# the same table imgutils uses (dbaesthetic._DEFAULT_LABEL_MAPPING): an image
# is "best" when it scores at least as well as 85% of Danbooru.
QUALITY_LABELS = ["masterpiece", "best", "great", "good", "normal", "low", "worst"]
QUALITY_FLOORS = {"masterpiece": 0.95, "best": 0.85, "great": 0.75, "good": 0.50,
                  "normal": 0.25, "low": 0.10, "worst": 0.0}
SCORES_FILE = "aesthetic_scores.json"
FILTERED_DIR = "_quality_filtered"
MOVES_FILE = "moved.json"
ORIGINALS_DIR = "original_images"  # Smart Aspect Ratio Bucketing backups


def percentile_to_label(percentile: float) -> str:
    for label in QUALITY_LABELS:
        if percentile >= QUALITY_FLOORS[label]:
            return label
    return QUALITY_LABELS[-1]


class AestheticScorer:
    def __init__(self):
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from imgutils.metrics.dbaesthetic import AestheticModel

            self._model = AestheticModel(AESTHETIC_REPO)

    def percentile(self, image_path: Path) -> float:
        _, percentile = self._model.get_aesthetic(str(image_path), AESTHETIC_MODEL, fmt=("label", "percentile"))
        return float(percentile)

    def unload(self) -> None:
        if self._model is not None:
            self._model.clear()
            self._model = None


class ScoreCache:
    """<dataset>/aesthetic_scores.json: percentile per file name.

    An entry only counts while the file is unchanged (size + mtime), so a
    re-cropped image is scored again. Moving a file keeps its mtime, which is
    why a restored image does not need to be scored twice.
    """

    def __init__(self, dataset_dir: Path):
        self.path = dataset_dir / SCORES_FILE
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            data = {}
        self.images = data.get("images", {}) if data.get("model") == AESTHETIC_ID else {}

    def get(self, image: Path) -> float | None:
        entry = self.images.get(image.name)
        if entry and entry.get("sig") == file_signature(image):
            return entry["percentile"]
        return None

    def put(self, image: Path, percentile: float) -> None:
        self.images[image.name] = {
            "percentile": round(percentile, 6),
            "label": percentile_to_label(percentile),
            "sig": file_signature(image),
        }

    def save(self) -> None:
        payload = {"model": AESTHETIC_ID, "images": self.images}
        self.path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def _unique_path(path: Path) -> Path:
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _load_moves(dataset_dir: Path) -> list[dict]:
    try:
        return json.loads((dataset_dir / FILTERED_DIR / MOVES_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return []


def _save_moves(dataset_dir: Path, moves: list[dict]) -> None:
    path = dataset_dir / FILTERED_DIR / MOVES_FILE
    if moves:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(moves, indent=1, ensure_ascii=False), encoding="utf-8")
    elif path.exists():
        path.unlink()


def move_out_of_dataset(dataset_dir: Path, image: Path, label: str) -> int:
    """Moves an image, its caption and its bucketing original to
    <dataset>/_quality_filtered/<label>/. Returns how many files moved.

    The original has to go too: Smart Aspect Ratio Bucketing re-creates every
    crop whose original is still in original_images, which would bring the
    filtered image right back into the dataset.
    """
    dest = dataset_dir / FILTERED_DIR / label
    pairs = [(image, dest / image.name)]
    caption = image.with_suffix(".txt")
    if caption.exists():
        pairs.append((caption, dest / caption.name))
    originals = dataset_dir / ORIGINALS_DIR
    if originals.is_dir():
        for original in originals.iterdir():
            if original.is_file() and original.stem == image.stem and original.suffix.lower() in IMAGE_EXTS:
                pairs.append((original, dest / ORIGINALS_DIR / original.name))

    moves = _load_moves(dataset_dir)
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = _unique_path(dst)
        shutil.move(str(src), str(dst))
        moves.append({"from": src.relative_to(dataset_dir).as_posix(), "to": dst.relative_to(dataset_dir).as_posix()})
    _save_moves(dataset_dir, moves)
    return len(pairs)


def restore_moved(dataset_dir: Path) -> tuple[int, list[str]]:
    """Undo move_out_of_dataset. Never overwrites: a file whose old place is
    taken again stays where it is and is reported. Returns (restored, conflicts)."""
    moves = _load_moves(dataset_dir)
    restored, conflicts, remaining = 0, [], []
    for move in reversed(moves):
        src, dst = dataset_dir / move["to"], dataset_dir / move["from"]
        if not src.exists():
            continue  # deleted by hand since then
        if dst.exists():
            conflicts.append(move["from"])
            remaining.append(move)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        restored += 1
    _save_moves(dataset_dir, list(reversed(remaining)))
    # drop the folders that are now empty, deepest first
    root = dataset_dir / FILTERED_DIR
    if root.is_dir():
        for folder in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if not any(folder.iterdir()):
                folder.rmdir()
        if not any(root.iterdir()):
            root.rmdir()
    return restored, conflicts


# ==========================================
# Training captions
# ==========================================
def read_caption(txt_path: Path) -> str | None:
    """The caption sd-scripts would use: first line of the .txt, stripped.
    None when there is no .txt; UnicodeDecodeError when it is not UTF-8."""
    try:
        text = txt_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    lines = text.splitlines()
    return lines[0].strip() if lines else ""


def build_caption_tags(raw: str, trigger: str, blacklist: TagBlacklist | None) -> tuple[str, int]:
    """Tag part of a training caption (the trigger goes in front separately).

    Unescapes \\( \\), drops blacklisted tags and duplicates, and drops the
    trigger word if the caption already has it, since it is re-added in front.
    Returns the tags and how many the blacklist removed.
    """
    trigger_key = tag_key(trigger) if trigger else None
    kept, seen, removed = [], set(), 0
    for part in raw.split(","):
        tag = unescape_tag(part.strip())
        if not tag:
            continue
        key = tag_key(tag)
        if key == trigger_key or key in seen:
            continue
        if blacklist and blacklist.blocks(tag):
            removed += 1
            continue
        seen.add(key)
        kept.append(tag)
    return ", ".join(kept), removed


def join_caption(trigger: str, tags: str) -> str:
    """What sd-scripts encodes for a metadata entry {"caption": trigger,
    "tags": tags} with caption_separator ", " (FineTuningDataset)."""
    return ", ".join(part for part in (trigger, tags) if part)


@dataclass
class TrainingEntry:
    image: Path
    tags: str
    size: tuple[int, int]


@dataclass
class CaptionReport:
    entries: list[TrainingEntry]
    errors: list[str]
    empty: list[str]
    blacklisted: int


def _names(items: list[str], limit: int = 5) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" ... (+{len(items) - limit})" if len(items) > limit else "")


def collect_training_entries(images: Iterable[Path], trigger: str, blacklist: TagBlacklist | None) -> CaptionReport:
    """Reads caption and size of every image that will be trained and checks
    everything that would otherwise crash or silently corrupt the run."""
    entries, missing, unreadable, oversized, empty = [], [], [], [], []
    blacklisted = 0
    by_stem: dict[str, list[str]] = {}
    for image in images:
        # Windows file names are case-insensitive, and so are the cache names
        by_stem.setdefault(image.stem.lower(), []).append(image.name)
        try:
            raw = read_caption(image.with_suffix(".txt"))
        except (UnicodeDecodeError, OSError):
            unreadable.append(image.with_suffix(".txt").name)
            continue
        if raw is None:
            missing.append(image.name)
            continue
        try:
            with Image.open(image) as img:
                size = img.size
        except Exception:
            unreadable.append(image.name)
            continue
        if max(size) >= MAX_IMAGE_SIDE:
            oversized.append(image.name)
        tags, removed = build_caption_tags(raw, trigger, blacklist)
        blacklisted += removed
        if not tags:
            empty.append(image.name)
        entries.append(TrainingEntry(image, tags, size))

    errors = []
    if missing:
        errors.append(f"❌ {len(missing)} image(s) without a .txt caption: {_names(missing)}. "
                      "Use 'Create .txt captions' first.")
    if unreadable:
        errors.append(f"❌ Could not read {len(unreadable)} file(s) (broken image or caption not in UTF-8): "
                      f"{_names(unreadable)}")
    if oversized:
        errors.append(f"❌ {len(oversized)} image(s) are too large (>= {MAX_IMAGE_SIDE}px): {_names(oversized)}. "
                      "Use the 'Smart Aspect Ratio Bucketing' tool to resize them before training.")
    clashes = [" / ".join(names) for names in by_stem.values() if len(names) > 1]
    if clashes:
        errors.append(f"❌ Images that only differ by extension share their cache files: {_names(clashes)}. "
                      "Rename one of each pair.")
    return CaptionReport(entries, errors, empty, blacklisted)


def write_caption_metadata(path: Path, entries: list[TrainingEntry], trigger: str) -> None:
    """sd-scripts fine-tuning metadata: exactly the images to train, with their
    captions. The trigger goes in "caption" and the tags in "tags", which keeps
    the tag frequency statistics that sd-scripts stores in the LoRA."""
    data = {
        entry.image.resolve().as_posix(): {
            "caption": trigger,
            "tags": entry.tags,
            "image_size": [int(entry.size[0]), int(entry.size[1])],
        }
        for entry in entries
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


# ==========================================
# Disk cache invalidation
# ==========================================
# Where sd-scripts' Anima strategies cache to (library/strategy_anima.py)
TE_CACHE_DIR = "cache_text_encoder"
TE_CACHE_SUFFIX = "_anima_te.npz"
LATENT_CACHE_DIR = "latent_cache"
LATENT_CACHE_RE = re.compile(r"(.+)_\d+x\d+_anima\.npz")
CACHE_MANIFEST = "trainflow_cache_manifest.json"


def model_fingerprint(path: str) -> str:
    p = Path(path)
    try:
        return f"{p.name}:{p.stat().st_size}"
    except OSError:
        return p.name


def refresh_disk_caches(dataset_dir: Path, captions: dict[Path, str], te_fingerprint: str,
                        latent_fingerprint: str) -> tuple[int, int]:
    """Deletes the cache files that no longer match what they were built from.

    sd-scripts only checks that a cached .npz has the expected keys and shape,
    never that it came from the current caption or image. Without this, a
    re-tagged dataset, a new trigger word or a re-cropped image would silently
    train on the old cache. A manifest next to the text encoder cache records
    what every entry was built from.

    Text encoder entries without history are dropped too: older versions of
    TrainFlow cached captions without the trigger word. Latents without history
    are kept (the image is the same file, as far as anyone can tell).
    Returns (text encoder entries dropped, latent entries dropped).
    """
    te_dir = dataset_dir / TE_CACHE_DIR
    latent_dir = dataset_dir / LATENT_CACHE_DIR
    manifest_path = te_dir / CACHE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        manifest = {}

    latents: dict[str, list[Path]] = {}
    if latent_dir.is_dir():
        for npz in latent_dir.iterdir():
            match = LATENT_CACHE_RE.fullmatch(npz.name)
            if match:
                latents.setdefault(match.group(1).lower(), []).append(npz)

    te_dropped = latent_dropped = 0
    for image, caption in captions.items():
        old = manifest.get(image.name, {})
        te_sig = hashlib.sha1(f"{te_fingerprint}\n{caption}".encode("utf-8")).hexdigest()
        latent_sig = f"{latent_fingerprint}|{file_signature(image)}"
        if old.get("te") != te_sig:
            npz = te_dir / f"{image.stem}{TE_CACHE_SUFFIX}"
            if npz.exists():
                npz.unlink()
                te_dropped += 1
        if "latent" in old and old["latent"] != latent_sig:
            for npz in latents.get(image.stem.lower(), []):
                npz.unlink()
                latent_dropped += 1
        manifest[image.name] = {"te": te_sig, "latent": latent_sig}

    te_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return te_dropped, latent_dropped
